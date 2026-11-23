#!/usr/bin/env python3
"""
Step 5: does the signal come back in OTHER TESS sectors?

Sector 80 gave us a period of about 0.36307 d. If this is a real binary, the same
eclipses must show up in sectors 91 and 92 (about a year later), at the same period,
with the same depths, and with the eclipses lining up in phase.
We use the QLP light curves (made from the full-frame images) from MAST.

Usage:
    python3 step5_other_sectors.py --tic 359791122 --ra 270.70257 --dec -13.88538 --period 0.363069
"""
import argparse
import os
import re
import glob

import numpy as np
from astropy.io import fits


# ---------------------------------------------------------------- loading

def running_median(t, f, window=2.0):
    """Slow-trend removal. A 2 day window spans several 0.36 d cycles, so it does not eat the signal."""
    out = np.empty_like(f)
    lo = np.searchsorted(t, t - window / 2)
    hi = np.searchsorted(t, t + window / 2)
    for i in range(len(t)):
        out[i] = np.median(f[lo[i]:hi[i]])
    return out


def load_qlp(path):
    with fits.open(path) as h:
        d = h[1].data
        names = d.columns.names
        t = np.array(d["TIME"], float)
        col = "KSPSAP_FLUX" if "KSPSAP_FLUX" in names else "SAP_FLUX"
        f = np.array(d[col], float)
        q = np.array(d["QUALITY"])
        sec = h[0].header.get("SECTOR")
    if sec is None:
        m = re.search(r"s(\d{4})", os.path.basename(path))
        sec = int(m.group(1)) if m else 0
    ok = np.isfinite(t) & np.isfinite(f) & (q == 0)
    t, f = t[ok], f[ok]
    o = np.argsort(t)
    t, f = t[o], f[o]
    f = f / np.median(f)
    f = f / running_median(t, f)
    # drop big upward outliers (cosmic rays, flares); keep dips, they are the signal
    mad = 1.4826 * np.median(np.abs(f - np.median(f)))
    keep = f < 1 + 5 * mad
    return int(sec), t[keep], f[keep]


# ---------------------------------------------------------------- period search (PDM)

def theta(t, f, P, nb=60):
    """Phase dispersion minimisation. Lower = the folded curve is tighter = better period."""
    idx = (((t / P) % 1.0) * nb).astype(int)
    n = np.bincount(idx, minlength=nb).astype(float)
    s = np.bincount(idx, f, nb)
    s2 = np.bincount(idx, f * f, nb)
    ok = n > 1
    within = (s2[ok] - s[ok] ** 2 / n[ok]).sum() / (n[ok].sum() - ok.sum())
    return within / f.var(ddof=1)


def scan(t, f, p0, half_width, step, nb=60):
    ps = np.arange(p0 - half_width, p0 + half_width, step)
    th = np.array([theta(t, f, p, nb) for p in ps])
    i = int(np.argmin(th))
    return ps, th, ps[i], th[i]


def fold_profile(t, f, P, nb=50):
    idx = (((t / P) % 1.0) * nb).astype(int)
    prof = np.full(nb, np.nan)
    for b in range(nb):
        x = f[idx == b]
        if len(x) >= 3:
            prof[b] = np.median(x)
    return prof


def eclipse_summary(prof):
    """Depth of the deeper and the shallower eclipse (%), and the phase of the deeper one."""
    nb = len(prof)
    level = np.nanpercentile(prof, 70)
    smooth = np.array([np.nanmean(prof[[(b - 1) % nb, b, (b + 1) % nb]]) for b in range(nb)])
    b1 = int(np.nanargmin(smooth))
    far = [b for b in range(nb) if 0.3 <= ((b - b1) % nb) / nb <= 0.7]
    b2 = far[int(np.nanargmin(smooth[far]))]
    d1 = 100 * (level - smooth[b1])
    d2 = 100 * (level - smooth[b2])
    return d1, d2, (b1 + 0.5) / nb, (b2 + 0.5) / nb


# ---------------------------------------------------------------- analysis

def analyze(files, p0, tic, out):
    data = {}
    for p in files:
        sec, t, f = load_qlp(p)
        if len(t) < 200:
            print("  sector %s: only %d good points, skipped" % (sec, len(t)))
            continue
        data[sec] = (t, f)
    if not data:
        raise SystemExit("No usable light curves.")
    secs = sorted(data)

    rows = []
    best_single = {}
    for s in secs:
        t, f = data[s]
        ps, th, pb, thb = scan(t, f, p0, 2e-3, 2e-6)
        best_single[s] = pb
        rows.append(dict(sector=s, n=len(t), span_d=t[-1] - t[0], p_best=pb, theta_best=thb,
                         theta_at_p0=theta(t, f, p0)))

    # one combined period from all sectors together (long baseline = very precise period)
    T = np.concatenate([data[s][0] for s in secs])
    F = np.concatenate([data[s][1] for s in secs])
    pc = p0
    th_comb = np.nan
    if len(secs) > 1:
        _, _, pc, th_comb = scan(T, F, p0, 3e-4, 1e-7)
    else:
        pc = best_single[secs[0]]

    for r in rows:
        t, f = data[r["sector"]]
        prof = fold_profile(t, f, pc)
        d1, d2, ph1, ph2 = eclipse_summary(prof)
        r.update(depth_deep_pct=d1, depth_shallow_pct=d2, phase_deep=ph1, phase_shallow=ph2)

    print("\nPer-sector results (QLP full-frame light curves)")
    hdr = "%6s %7s %8s %11s %11s %13s %9s %10s %10s" % (
        "sector", "n", "span_d", "period_d", "theta_best", "theta_at_P0", "deep_%", "shallow_%", "deep_phase")
    print(hdr)
    for r in rows:
        print("%6d %7d %8.1f %11.6f %11.3f %13.3f %9.2f %10.2f %10.2f" % (
            r["sector"], r["n"], r["span_d"], r["p_best"], r["theta_best"], r["theta_at_p0"],
            r["depth_deep_pct"], r["depth_shallow_pct"], r["phase_deep"]))
    print("(theta: 1.0 = nothing repeats, lower = a tighter repeating curve. Real signals give similar low values in every sector.)")

    if len(secs) > 1:
        print("\nCombined period from sectors %s : %.7f d   (theta %.3f)" % (secs, pc, th_comb))
        print("Phases of the deeper eclipse in each sector at that period should agree if the signal is coherent.")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    for k, s in enumerate(secs):
        t, f = data[s]
        ph = (t / pc) % 1.0
        ax[0].scatter(ph, f - 0.02 * k, s=2, alpha=0.25, label="S%d" % s)
        nb = 50
        prof = fold_profile(t, f, pc, nb)
        ax[0].plot((np.arange(nb) + 0.5) / nb, prof - 0.02 * k, "k-", lw=1)
    ax[0].set_xlabel("orbital phase (period %.6f d)" % pc)
    ax[0].set_ylabel("normalised flux (sectors offset)")
    ax[0].set_title("TIC %d: every sector folded at the same period" % tic, fontsize=9)
    ax[0].legend(fontsize=8, markerscale=4)
    for s in secs:
        t, f = data[s]
        ps, th, _, _ = scan(t, f, p0, 2e-3, 2e-6)
        ax[1].plot(ps, th, label="S%d" % s)
    ax[1].axvline(pc, color="r", ls=":")
    ax[1].set_xlabel("trial period (d)")
    ax[1].set_ylabel("theta (lower = better)")
    ax[1].set_title("period search per sector", fontsize=9)
    ax[1].legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(out, exist_ok=True)
    png = os.path.join(out, "TIC%d_multisector.png" % tic)
    plt.savefig(png, dpi=130)
    csv = os.path.join(out, "TIC%d_multisector.csv" % tic)
    keys = list(rows[0].keys())
    with open(csv, "w") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join(str(r[k]) for k in keys) + "\n")
    print("\nWrote %s and %s" % (png, csv))


# ---------------------------------------------------------------- network part

def neighbours(ra, dec, tic):
    """Nearby stars from the TESS Input Catalog: could one of them be the real source of the dips?"""
    try:
        from astroquery.mast import Catalogs
        from astropy.coordinates import SkyCoord
        import astropy.units as u
        c = SkyCoord(ra, dec, unit="deg")
        t = Catalogs.query_region(c, radius=45 * u.arcsec, catalog="TIC")
        t = t[np.argsort(np.array(t["dstArcSec"], float))]
        print("\nNearby stars in the TESS Input Catalog (TESS pixels are 21 arcsec wide):")
        print("%12s %8s %8s %8s" % ("TIC", "sep_as", "Tmag", "dTmag"))
        t0 = None
        for r in t:
            if int(r["ID"]) == tic:
                t0 = float(r["Tmag"])
        for r in t[:12]:
            tm = float(r["Tmag"]) if np.isfinite(float(r["Tmag"])) else np.nan
            print("%12d %8.1f %8.2f %8.2f" % (int(r["ID"]), float(r["dstArcSec"]), tm,
                                               tm - t0 if t0 is not None else np.nan))
    except Exception as e:
        print("(neighbour lookup failed: %s)" % e)


def fetch(tic, ra, dec, dl):
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    print("Asking MAST for QLP light curves of TIC %d ..." % tic)
    obs = Observations.query_criteria(coordinates=SkyCoord(ra, dec, unit="deg"), radius=0.003 * u.deg,
                                      obs_collection="HLSP", provenance_name="QLP",
                                      target_name=str(tic))
    print("MAST lists %d QLP observations." % len(obs))
    if len(obs) == 0:
        raise SystemExit("Nothing found.")
    prods = Observations.get_product_list(obs)
    names = [str(x) for x in prods["productFilename"]]
    sel = [i for i, n in enumerate(names) if n.endswith("llc.fits")]
    if not sel:
        print("No *_llc.fits files. Files offered were:")
        for n in names:
            print("  ", n)
        raise SystemExit("Send me this list.")
    prods = prods[sel]
    print("Downloading %d light-curve files ..." % len(prods))
    m = Observations.download_products(prods, download_dir=dl)
    return sorted(str(p) for p in m["Local Path"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--period", type=float, required=True)
    ap.add_argument("--out", default="followup")
    ap.add_argument("--local", default=None, help="folder that already holds the QLP *_llc.fits files")
    a = ap.parse_args()

    dl = os.path.join(a.out, "_tmp_qlp")
    if a.local:
        files = sorted(glob.glob(os.path.join(a.local, "**", "*llc.fits"), recursive=True))
    else:
        files = fetch(a.tic, a.ra, a.dec, dl)
    print("Using %d files" % len(files))
    analyze(files, a.period, a.tic, a.out)
    if not a.local:
        neighbours(a.ra, a.dec, a.tic)


if __name__ == "__main__":
    main()
