#!/usr/bin/env python3
"""
STEP 2: score every light curve for "weirdness".

Reads the .npz files made by step1, turns each light curve into ~25 numbers
(how spiky, how smooth, how periodic, any big dips or steps, ...), then scores
every star with three different outlier detectors and averages their ranks.
Stars are also flagged for things that are usually instrument problems rather
than real stars (crowded pixels, faint stars, glitches shared by many stars).

Outputs (in output/):
  ranked.csv                 every star, best (weirdest) first
  crossmatch_input.csv       top clean candidates, for step3
  plots/                     a picture for each of the top candidates
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import signal, stats
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import RobustScaler

MIN_POINTS = 5000        # a normal sector has about 15,000 good points
BIN_DAYS = 1.0 / 48.0    # 30-minute bins for the slow-timescale features

FEATURES = [
    "log_mad", "rms_over_mad", "skew", "kurt", "zmax", "zmin",
    "frac_hi", "frac_lo", "dip_run", "n_flares", "n_dips", "eta", "hf_ratio",
    "ls_peak", "ls_logP", "ls_second", "spec_slope",
    "half_shift", "poly_amp", "step_max", "range_99_1", "slope_abs",
]


def _runs(mask):
    """Lengths of consecutive True runs in a boolean array."""
    if not mask.any():
        return np.array([], dtype=int)
    m = np.concatenate(([False], mask, [False]))
    edges = np.flatnonzero(m[1:] != m[:-1])
    return edges[1::2] - edges[0::2]


def load_star(path):
    z = np.load(path)
    t, f, q = z["t"], z["f"], z["q"]
    ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
    if ok.sum() < MIN_POINTS:
        return None
    t, f = t[ok], f[ok]
    med = np.median(f)
    if not np.isfinite(med) or med <= 0:
        return None
    meta = {k: float(z[k]) for k in ["tic", "sector", "ra", "dec", "tmag", "teff", "radius", "crowdsap"]}
    meta["n_points"] = int(ok.sum())
    return t, f / med, meta


def bin_curve(t, fn):
    idx = np.floor((t - t[0]) / BIN_DAYS).astype(int)
    n = idx.max() + 1
    cnt = np.bincount(idx, minlength=n).astype(float)
    sm = np.bincount(idx, weights=fn, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        fb = sm / cnt
    fb[cnt < 5] = np.nan
    tb = t[0] + (np.arange(n) + 0.5) * BIN_DAYS
    good = np.isfinite(fb)
    return tb[good], fb[good]


def features(t, fn):
    med = np.median(fn)
    mad = 1.4826 * np.median(np.abs(fn - med))
    mad = max(mad, 1e-6)
    zn = np.clip((fn - med) / mad, -60, 60)

    tb, fb = bin_curve(t, fn)
    if len(fb) < 60:
        return None, None
    medb = np.median(fb)
    madb = max(1.4826 * np.median(np.abs(fb - medb)), 1e-6)
    zb = np.clip((fb - medb) / madb, -60, 60)

    out = {}
    out["log_mad"] = np.log10(mad)
    out["rms_over_mad"] = min(np.std(fn) / mad, 60)
    out["skew"] = float(np.clip(stats.skew(zn), -30, 30))
    out["kurt"] = float(np.clip(stats.kurtosis(zn), -5, 300))
    out["zmax"] = float(zn.max())
    out["zmin"] = float(zn.min())
    out["frac_hi"] = float(np.mean(zn > 3))
    out["frac_lo"] = float(np.mean(zn < -3))
    out["dip_run"] = float(_runs(zn < -2).max(initial=0)) / len(zn)
    out["n_flares"] = float(np.sum(_runs(zn > 3) >= 3))
    out["n_dips"] = float(np.sum(_runs(zb < -3) >= 2))
    d = np.diff(zb)
    out["eta"] = float(np.sum(d ** 2) / ((len(zb) - 1) * max(np.var(zb), 1e-9)))
    out["hf_ratio"] = float(1.4826 * np.median(np.abs(np.diff(fn))) / np.sqrt(2) / mad)

    # Lomb-Scargle periodogram on the 30-min binned curve
    periods = np.logspace(np.log10(0.15), np.log10(12.0), 300)
    freq = 2 * np.pi / periods
    p = signal.lombscargle(tb, zb - zb.mean(), freq, normalize=True)
    k = int(np.argmax(p))
    out["ls_peak"] = float(p[k])
    out["ls_logP"] = float(np.log10(periods[k]))
    far = np.abs(np.log(periods / periods[k])) > 0.25
    out["ls_second"] = float(p[far].max() / max(p[k], 1e-9)) if far.any() else 0.0
    out["spec_slope"] = float(np.polyfit(np.log(1.0 / periods), np.log(p + 1e-9), 1)[0])

    # slow trends and steps
    half = len(zb) // 2
    out["half_shift"] = float(abs(np.median(zb[half:]) - np.median(zb[:half])))
    x = (tb - tb.mean()) / (tb.max() - tb.min())
    out["poly_amp"] = float(np.std(np.polyval(np.polyfit(x, zb, 3), x)))
    steps = []
    for s in np.linspace(0.15, 0.85, 15):
        c = int(len(zb) * s)
        steps.append(abs(np.median(zb[c:]) - np.median(zb[:c])))
    out["step_max"] = float(max(steps))
    out["range_99_1"] = float(np.percentile(zn, 99) - np.percentile(zn, 1))
    out["slope_abs"] = float(abs(np.polyfit(x, zb, 1)[0]))

    # how many strong outliers sit right next to a data gap or the ends (typical instrument ramps)
    gap_after = np.where(np.diff(tb) > 0.5)[0]
    edges = [tb[0], tb[-1]] + [tb[i] for i in gap_after] + [tb[i + 1] for i in gap_after]
    near_edge = np.zeros(len(tb), bool)
    for te in edges:
        near_edge |= np.abs(tb - te) < 0.5
    strong = np.abs(zb) > 4
    out["n_strong"] = float(strong.sum())
    out["edge_frac"] = float((strong & near_edge).sum() / max(strong.sum(), 1))

    i = int(np.argmax(np.abs(zn)))
    out["_t_peak"] = float(t[i])
    return out, (tb, fb)


def guess_label(r):
    """Very rough human-readable hint. It is only a hint, not a classification."""
    tags = []
    if r["ls_peak"] > 0.6:
        tags.append("strongly periodic")
    if r["n_flares"] >= 2 or r["zmax"] > 15 and r["skew"] > 3:
        tags.append("flare/outburst-like")
    if r["n_dips"] >= 1 and r["ls_peak"] < 0.6:
        tags.append("non-periodic dips")
    if r["step_max"] > 4 and r["ls_peak"] < 0.5:
        tags.append("level shift")
    return "; ".join(tags) if tags else "unclear"


def score_models(X, seed=0):
    Xs = RobustScaler().fit_transform(X)
    Xs = np.clip(Xs, -8, 8)

    iso = IsolationForest(n_estimators=400, random_state=seed).fit(Xs)
    s_iso = -iso.score_samples(Xs)

    k = min(20, max(5, len(Xs) // 20))
    lof = LocalOutlierFactor(n_neighbors=k).fit(Xs)
    s_lof = -lof.negative_outlier_factor_

    ae = MLPRegressor(hidden_layer_sizes=(16, 6, 16), max_iter=600, random_state=seed,
                      learning_rate_init=0.003)
    ae.fit(Xs, Xs)
    s_ae = np.mean((ae.predict(Xs) - Xs) ** 2, axis=1)

    return s_iso, s_lof, s_ae


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="output")
    ap.add_argument("--top", type=int, default=100)
    ap.add_argument("--plots", type=int, default=40)
    a = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(a.data, "**", "*.npz"), recursive=True))
    print("Found %d light curve files." % len(paths))
    rows, curves = [], {}
    for pth in paths:
        s = load_star(pth)
        if s is None:
            continue
        t, fn, meta = s
        res = features(t, fn)
        if res[0] is None:
            continue
        feats, binned = res
        row = dict(meta)
        row.update(feats)
        rows.append(row)
        curves[(int(meta["tic"]), int(meta["sector"]))] = (t, fn, binned)
    df = pd.DataFrame(rows)
    df["tic"] = df["tic"].astype(int)
    df["sector"] = df["sector"].astype(int)
    print("Usable stars: %d" % len(df))
    if len(df) < 60:
        raise SystemExit("Too few usable stars to score. Download more with step1.")

    X = df[FEATURES].to_numpy(dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=60, neginf=-60)
    s_iso, s_lof, s_ae = score_models(X)
    df["s_iso"], df["s_lof"], df["s_ae"] = s_iso, s_lof, s_ae
    df["score"] = (
        pd.Series(s_iso).rank(pct=True) + pd.Series(s_lof).rank(pct=True) + pd.Series(s_ae).rank(pct=True)
    ).to_numpy() / 3.0

    # ---- flags for things that are probably the instrument, not the star ----
    flags = [[] for _ in range(len(df))]
    for i, r in df.iterrows():
        if np.isfinite(r["crowdsap"]) and r["crowdsap"] < 0.8:
            flags[i].append("crowded")
        if np.isfinite(r["tmag"]) and r["tmag"] > 14.5:
            flags[i].append("faint")
        if r["n_points"] < 9000:
            flags[i].append("gappy")
        if r["n_strong"] >= 3 and r["edge_frac"] > 0.6:
            flags[i].append("edge_ramp")
    # glitches shared by many stars at the same moment
    tp = df["_t_peak"].to_numpy()
    for sec, grp in df.groupby("sector"):
        bins = np.floor(grp["_t_peak"].to_numpy() / 0.25).astype(int)
        vals, counts = np.unique(bins, return_counts=True)
        span = max(1, int(bins.max() - bins.min() + 1))
        expected = len(grp) / span          # how many peaks per bin if times were random
        thresh = expected + 6.0 * np.sqrt(expected) + 3.0
        bad = set(vals[counts >= thresh])
        for ix, b in zip(grp.index, bins):
            if b in bad:
                flags[ix].append("shared_glitch")
    df["flags"] = [",".join(f) for f in flags]
    df["clean"] = df["flags"] == ""
    df["hint"] = [guess_label(r) for _, r in df.iterrows()]

    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    os.makedirs(a.out, exist_ok=True)
    keep = ["rank", "tic", "sector", "ra", "dec", "tmag", "teff", "radius", "crowdsap",
            "score", "s_iso", "s_lof", "s_ae", "clean", "flags", "hint"] + FEATURES
    df[keep].to_csv(os.path.join(a.out, "ranked.csv"), index=False)
    df[df["clean"]].head(a.top)[["rank", "tic", "sector", "ra", "dec", "tmag", "score", "hint"]].to_csv(
        os.path.join(a.out, "crossmatch_input.csv"), index=False)

    # ---- pictures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pdir = os.path.join(a.out, "plots")
    os.makedirs(pdir, exist_ok=True)
    for _, r in df[df["clean"]].head(a.plots).iterrows():
        t, fn, (tb, fb) = curves[(int(r["tic"]), int(r["sector"]))]
        fig, ax = plt.subplots(figsize=(11, 3.6))
        ax.plot(t, fn, ".", ms=1.2, alpha=0.35, color="#7a8aa0")
        ax.plot(tb, fb, "-", lw=1.2, color="#d1495b")
        ax.set_title("#%d  TIC %d  sector %d  Tmag %.1f  score %.3f  [%s]" % (
            r["rank"], r["tic"], r["sector"], r["tmag"], r["score"], r["hint"]), fontsize=9)
        ax.set_xlabel("time (BTJD days)")
        ax.set_ylabel("normalized flux")
        fig.tight_layout()
        fig.savefig(os.path.join(pdir, "rank%03d_TIC%d.png" % (r["rank"], r["tic"])), dpi=110)
        plt.close(fig)

    print("Wrote %s/ranked.csv, crossmatch_input.csv and %d plots." % (a.out, min(a.plots, int(df["clean"].sum()))))
    print("Clean stars: %d of %d" % (int(df["clean"].sum()), len(df)))


if __name__ == "__main__":
    main()
