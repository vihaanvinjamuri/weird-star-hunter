#!/usr/bin/env python3
"""
Step 7: measure the binary properly, with error bars.

What it produces (this is the table every confirmation of our claim needs):
  * the time of primary minimum light (an "epoch") and the orbital period, each with an uncertainty
  * where the secondary eclipse falls (phase 0.5 means a circular orbit)
  * eclipse depths per sector with error bars (bootstrap over orbital cycles)
  * a light-ephemeris line:  Min I = BJD(TDB) T0 + P * E

Method (the standard "eclipse timing" idea):
  1. Fold all sectors at our period to build an average eclipse shape (the template).
  2. For every single eclipse in the data, slide the template in time until it fits best.
     That gives one measured time of minimum per eclipse.
  3. Fit a straight line (time against cycle number). Slope = period, intercept = epoch.
     The scatter of the points around the line gives the honest error bars.

Usage:
    python3 step7_ephemeris.py --tic 359791122 --period 0.363081 --local followup/_tmp_qlp
"""
import argparse
import glob
import os

import numpy as np

from step5_other_sectors import load_qlp


# ---------------------------------------------------------------- template

def circ_smooth(y, k=1):
    out = np.empty_like(y)
    n = len(y)
    for i in range(n):
        out[i] = np.mean([y[(i + j) % n] for j in range(-k, k + 1)])
    return out


def build_template(t, f, P, T0, nb=100):
    ph = ((t - T0) / P) % 1.0
    idx = (ph * nb).astype(int)
    prof = np.full(nb, np.nan)
    for b in range(nb):
        x = f[idx == b]
        if len(x) >= 3:
            prof[b] = np.median(x)
    good = np.isfinite(prof)
    xs = (np.arange(nb) + 0.5) / nb
    prof = np.interp(xs, xs[good], prof[good], period=1.0)
    return xs, circ_smooth(prof, 1)


def make_g(xs, prof):
    """Template as a function of orbital phase (wraps around at 1)."""
    xe = np.r_[xs[-1] - 1.0, xs, xs[0] + 1.0]
    ye = np.r_[prof[-1], prof, prof[0]]

    def g(phase):
        p = np.asarray(phase) % 1.0
        p = np.where(p > xs[-1], p - 1.0, p)
        return np.interp(p, xe, ye)
    return g


def parabola_min(x, y):
    """Position of the minimum of the 3-point parabola through (x, y)."""
    a, b, _ = np.polyfit(x, y, 2)
    return -b / (2 * a) if a > 0 else x[1]


def find_minima(xs, prof):
    nb = len(xs)
    i1 = int(np.argmin(prof))
    p1 = parabola_min(np.array([-1.0, 0.0, 1.0]), prof[[(i1 - 1) % nb, i1, (i1 + 1) % nb]])
    ph1 = (xs[i1] + p1 / nb) % 1.0
    far = [b for b in range(nb) if 0.35 <= ((xs[b] - ph1) % 1.0) <= 0.65]
    i2 = far[int(np.argmin(prof[far]))]
    p2 = parabola_min(np.array([-1.0, 0.0, 1.0]), prof[[(i2 - 1) % nb, i2, (i2 + 1) % nb]])
    ph2 = (xs[i2] + p2 / nb) % 1.0
    return ph1, ph2


# ---------------------------------------------------------------- timing of single eclipses

def fit_one(t, f, tc, g, T0, P, half, cadence, grid=0.012, step=0.0002):
    m = np.abs(t - tc) < half
    if m.sum() < 0.8 * (2 * half / cadence):
        return None
    tt, ff = t[m], f[m]
    shifts = np.arange(-grid, grid + step / 2, step)
    sse = np.empty(len(shifts))
    for i, d in enumerate(shifts):
        gg = g((tt - d - T0) / P)
        A = np.vstack([np.ones_like(gg), gg]).T
        coef, res, *_ = np.linalg.lstsq(A, ff, rcond=None)
        sse[i] = np.sum((ff - A @ coef) ** 2)
    j = int(np.argmin(sse))
    if j == 0 or j == len(shifts) - 1:
        return None
    d0 = parabola_min(shifts[j - 1:j + 2], sse[j - 1:j + 2])
    curv = (sse[j - 1] - 2 * sse[j] + sse[j + 1]) / step ** 2
    s2 = sse[j] / max(len(tt) - 3, 1)
    if curv <= 0:
        return None
    sig = np.sqrt(2 * s2 / curv)
    return d0, sig


def wlin(E, T, sig):
    """Weighted straight-line fit T = Tm + P (E - Em), with errors scaled by the real scatter."""
    Em = float(np.round(np.mean(E)))
    x = E - Em
    keep = np.ones(len(x), bool)
    for _ in range(2):
        w = 1.0 / np.maximum(sig[keep], 1e-5)
        coef, cov = np.polyfit(x[keep], T[keep], 1, w=w, cov="unscaled")
        res = T - np.polyval(coef, x)
        scat = np.std(res[keep])
        keep = np.abs(res) < 3.5 * max(scat, 1e-5)
    w = 1.0 / np.maximum(sig[keep], 1e-5)
    coef, cov = np.polyfit(x[keep], T[keep], 1, w=w, cov="unscaled")
    res = T[keep] - np.polyval(coef, x[keep])
    chi2 = np.sum((res / np.maximum(sig[keep], 1e-5)) ** 2) / max(keep.sum() - 2, 1)
    scale = max(chi2, 1.0)
    P, Tm = coef
    return dict(P=P, Tm=Tm, sP=np.sqrt(cov[0, 0] * scale), sT=np.sqrt(cov[1, 1] * scale), Em=Em,
                n_used=int(keep.sum()), n_all=len(E), rms_min=np.std(res) * 1440, keep=keep, res_all=T - np.polyval(coef, x))


# ---------------------------------------------------------------- depths with bootstrap

def depths_from(t, f, P, T0, nb=100):
    ph = ((t - T0) / P) % 1.0
    idx = (ph * nb).astype(int)
    prof = np.array([np.median(f[idx == b]) if (idx == b).sum() >= 3 else np.nan for b in range(nb)])
    if np.isnan(prof).any():
        return np.nan, np.nan
    sm = circ_smooth(prof, 1)
    xs = (np.arange(nb) + 0.5) / nb
    quad = (np.abs(xs - 0.25) < 0.05) | (np.abs(xs - 0.75) < 0.05)
    level = np.median(prof[quad])
    prim = sm[(xs < 0.1) | (xs > 0.9)].min()
    sec = sm[np.abs(xs - 0.5) < 0.1].min()
    return 100 * (1 - prim / level), 100 * (1 - sec / level)


def boot_depths(t, f, P, T0, nboot=150, seed=1):
    rng = np.random.default_rng(seed)
    cyc = np.floor((t - T0) / P).astype(int)
    ucyc = np.unique(cyc)
    groups = {c: np.where(cyc == c)[0] for c in ucyc}
    base = depths_from(t, f, P, T0)
    out = []
    for _ in range(nboot):
        pick = rng.choice(ucyc, size=len(ucyc), replace=True)
        ii = np.concatenate([groups[c] for c in pick])
        d = depths_from(t[ii], f[ii], P, T0)
        if np.isfinite(d[0]) and np.isfinite(d[1]):
            out.append(d)
    out = np.array(out)
    return base, out.std(axis=0) if len(out) > 10 else (np.nan, np.nan)


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--period", type=float, required=True, help="starting period (days), e.g. the step 5 combined value")
    ap.add_argument("--local", default="followup/_tmp_qlp", help="folder holding the QLP *_llc.fits files")
    ap.add_argument("--out", default="followup")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.local, "**", "*llc.fits"), recursive=True))
    if not files:
        raise SystemExit("No *_llc.fits files under %s. Run step 5 first." % a.local)
    data = {}
    for p in files:
        sec, t, f = load_qlp(p)
        if len(t) > 200:
            data[sec] = (t, f)
    secs = sorted(data)
    T = np.concatenate([data[s][0] for s in secs])
    F = np.concatenate([data[s][1] for s in secs])
    P0 = a.period
    cad = float(np.median(np.diff(T[T < T.min() + 20])))
    print("Sectors %s, %d points, cadence %.0f s" % (secs, len(T), cad * 86400))

    # first guess of the zero point: deepest part of the folded curve
    Tref = float(T.min())
    xs, prof = build_template(T, F, P0, Tref)
    ph1, ph2 = find_minima(xs, prof)
    Tref = Tref + ph1 * P0
    xs, prof = build_template(T, F, P0, Tref)           # rebuild with the primary minimum at phase 0
    ph1, ph2 = find_minima(xs, prof)
    if ph1 > 0.5:
        ph1 -= 1.0
    Tref = Tref + ph1 * P0
    xs, prof = build_template(T, F, P0, Tref)
    ph1, ph2 = find_minima(xs, prof)
    g = make_g(xs, prof)
    half = 0.12 * P0

    results = {}
    for name, off in (("primary", 0.0), ("secondary", ph2)):
        Es, Ts, Ss = [], [], []
        n_lo = int(np.floor((T.min() - Tref) / P0)) - 1
        n_hi = int(np.ceil((T.max() - Tref) / P0)) + 1
        for n in range(n_lo, n_hi + 1):
            tc = Tref + (n + off) * P0
            if tc < T.min() or tc > T.max():
                continue
            r = fit_one(T, F, tc, g, Tref, P0, half, cad)
            if r is None:
                continue
            d0, sig = r
            Es.append(n)
            Ts.append(tc + d0)
            Ss.append(sig)
        Es, Ts, Ss = np.array(Es), np.array(Ts), np.array(Ss)
        if len(Es) < 10:
            raise SystemExit("Only %d usable %s eclipses; cannot fit." % (len(Es), name))
        results[name] = (Es, Ts, Ss, wlin(Es, Ts, Ss))

    pr = results["primary"][3]
    se = results["secondary"][3]
    # secondary phase from the two fits evaluated at the same mean epoch
    Em = pr["Em"]
    Tp = pr["Tm"]
    Ts_at = se["Tm"] + se["P"] * (Em - se["Em"])
    sec_phase = ((Ts_at - Tp) / pr["P"]) % 1.0
    sec_phase_err = np.sqrt(pr["sT"] ** 2 + se["sT"] ** 2 + (se["sP"] * (Em - se["Em"])) ** 2) / pr["P"]
    sec_phase_err = float(np.sqrt(sec_phase_err ** 2 + 0.002 ** 2))   # method floor found by testing on fake data

    lines = []

    def say(s=""):
        print(s)
        lines.append(s)

    say("\nTIC %d ephemeris from QLP sectors %s" % (a.tic, secs))
    say("-" * 70)
    say("Primary eclipses timed    : %d used of %d" % (pr["n_used"], pr["n_all"]))
    say("Secondary eclipses timed  : %d used of %d" % (se["n_used"], se["n_all"]))
    say("Scatter of primary times  : %.1f minutes (rms around the straight line)" % pr["rms_min"])
    say("")
    say("Period (primary timing)   : %.7f +- %.7f d" % (pr["P"], pr["sP"]))
    say("Period (secondary timing) : %.7f +- %.7f d" % (se["P"], se["sP"]))
    say("Time of primary minimum   : BJD(TDB) %.5f +- %.5f   (epoch E = %d)" % (pr["Tm"] + 2457000, pr["sT"], pr["Em"]))
    say("Secondary eclipse phase   : %.4f +- %.4f   (0.5000 = circular orbit; error includes a 0.002 method floor)" % (sec_phase, sec_phase_err))
    say("")
    say("Light ephemeris           : Min I = BJD(TDB) %.5f + %.7f x E" % (pr["Tm"] - pr["Em"] * pr["P"] + 2457000, pr["P"]))
    say("")
    say("Eclipse depths per sector (percent of out-of-eclipse flux, bootstrap errors)")
    say("%8s %16s %16s" % ("sector", "primary %", "secondary %"))
    for s in secs:
        t, f = data[s]
        (d1, d2), (e1, e2) = boot_depths(t, f, pr["P"], pr["Tm"] - pr["Em"] * pr["P"])
        say("%8d %9.2f +- %4.2f %9.2f +- %4.2f" % (s, d1, e1, d2, e2))
    say("\nA period difference between primary and secondary timing larger than about 3 sigma, or a")
    say("secondary phase far from 0.5, would be unusual for a contact binary. Tell me if you see that.")

    os.makedirs(a.out, exist_ok=True)
    txt = os.path.join(a.out, "TIC%d_ephemeris.txt" % a.tic)
    with open(txt, "w") as fh:
        fh.write("\n".join(lines) + "\n")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    T0 = pr["Tm"] - pr["Em"] * pr["P"]
    ph = ((T - T0) / pr["P"] + 0.25) % 1.0 - 0.25
    ax[0].scatter(ph, F, s=1.5, alpha=0.15, color="tab:blue")
    ax[0].scatter(ph + 1.0, F, s=1.5, alpha=0.15, color="tab:blue")
    ax[0].set_xlim(-0.25, 1.25)
    ax[0].set_xlabel("orbital phase (primary minimum = 0)")
    ax[0].set_ylabel("normalised flux")
    ax[0].set_title("TIC %d, sectors %s, P = %.6f d" % (a.tic, secs, pr["P"]), fontsize=9)
    Es, Ts, Ss, fit = results["primary"]
    ax[1].errorbar(Es, fit["res_all"] * 1440, yerr=Ss * 1440, fmt=".", ms=3, alpha=0.5, label="primary")
    Es2, Ts2, Ss2, fit2 = results["secondary"]
    ax[1].errorbar(Es2, fit2["res_all"] * 1440, yerr=Ss2 * 1440, fmt=".", ms=3, alpha=0.5, label="secondary")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("cycle number")
    ax[1].set_ylabel("observed minus fitted time (minutes)")
    ax[1].set_title("eclipse timing residuals (flat = steady period)", fontsize=9)
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    png = os.path.join(a.out, "TIC%d_ephemeris.png" % a.tic)
    plt.savefig(png, dpi=130)
    print("\nWrote %s and %s" % (txt, png))


if __name__ == "__main__":
    main()
