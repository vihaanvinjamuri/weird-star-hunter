#!/usr/bin/env python3
"""
STEP 4 (run this on your Mac, it needs internet).

Follow-up on ONE candidate star. Downloads every TESS 2-minute light curve MAST has
for it (all sectors), then checks:
  1. Is the same period found in every sector?            (a real signal repeats)
  2. How deep are the two eclipses in each sector?         (stable depths = real, stable system)
  3. Does the star's image centroid move during eclipses?  (a rough check on whether the
     signal comes from the target star or from a neighbour inside the same TESS pixels)
  4. What is the best combined period using all sectors?

Output: followup/TIC<id>_summary.csv, followup/TIC<id>_folded.png and a printed table.

Example
  python3 step4_followup.py --tic 359791122 --period 0.3631
"""
import argparse
import os
import shutil
import sys

import numpy as np


def pdm_theta(t, f, P, nb=100):
    """Phase dispersion: smaller means the folded curve is tighter (better period)."""
    b = ((t / P) % 1.0 * nb).astype(int)
    n = np.bincount(b, minlength=nb).astype(float)
    s = np.bincount(b, weights=f, minlength=nb)
    s2 = np.bincount(b, weights=f * f, minlength=nb)
    ok = n > 3
    ss = (s2[ok] - s[ok] ** 2 / n[ok]).sum()
    return ss / max(n[ok].sum(), 1.0)


def best_period(t, f, lo, hi, steps):
    grid = np.linspace(lo, hi, steps)
    th = np.array([pdm_theta(t, f, P) for P in grid])
    return float(grid[int(np.argmin(th))])


def fold_levels(t, f, P, nb=60):
    """Out-of-eclipse level and the depth of the two eclipses (folded at 2P)."""
    ph = (t / (2 * P)) % 1.0
    b = (ph * nb).astype(int)
    med = np.array([np.median(f[b == i]) if (b == i).sum() > 5 else np.nan for i in range(nb)])
    out_level = np.nanpercentile(med, 90)
    d1 = out_level - np.nanmin(med[: nb // 2])
    d2 = out_level - np.nanmin(med[nb // 2:])
    return float(out_level), float(d1), float(d2)


def centroid_shift(t, f, c, P):
    """Mean centroid offset in-eclipse minus out-of-eclipse, in milli-pixels, with its error."""
    from scipy.ndimage import median_filter
    ok = np.isfinite(c)
    if ok.sum() < 2000:
        return np.nan, np.nan
    t, f, c = t[ok], f[ok], c[ok]
    c = c - median_filter(c, size=min(1081, (len(c) // 2) * 2 - 1), mode="nearest")   # remove slow drift
    nb = 100
    ph = (t / P) % 1.0
    b = (ph * nb).astype(int)
    med = np.array([np.median(f[b == i]) if (b == i).sum() > 5 else np.nan for i in range(nb)])
    ph0 = (np.nanargmin(med) + 0.5) / nb
    dphi = (ph - ph0 + 0.5) % 1.0 - 0.5
    inn = np.abs(dphi) < 0.07
    out = np.abs(dphi) > 0.2
    if inn.sum() < 50 or out.sum() < 50:
        return np.nan, np.nan
    shift = c[inn].mean() - c[out].mean()
    err = np.sqrt(c[inn].var() / inn.sum() + c[out].var() / out.sum())
    return float(shift * 1000), float(err * 1000)


def analyze_fits(path, p0):
    """Analyse one SPOC light-curve FITS file. Returns (summary dict, t, flux)."""
    from astropy.io import fits
    with fits.open(path) as hdul:
        h0, h1, d = hdul[0].header, hdul[1].header, hdul[1].data
        t = np.array(d["TIME"], dtype=float)
        f = np.array(d["PDCSAP_FLUX"], dtype=float)
        q = np.array(d["QUALITY"], dtype=int)
        cols = d.columns.names
        c1 = np.array(d["MOM_CENTR1"], dtype=float) if "MOM_CENTR1" in cols else np.full(len(t), np.nan)
        c2 = np.array(d["MOM_CENTR2"], dtype=float) if "MOM_CENTR2" in cols else np.full(len(t), np.nan)
        sector = int(h0["SECTOR"])
        crowd = float(h1.get("CROWDSAP", np.nan))
    m = np.isfinite(t) & np.isfinite(f) & (q == 0)
    t, f, c1, c2 = t[m], f[m], c1[m], c2[m]
    f = f / np.median(f)
    P = best_period(t, f, p0 * 0.99, p0 * 1.01, 1200)
    P = best_period(t, f, P - 3e-4, P + 3e-4, 600)
    level, d1, d2 = fold_levels(t, f, P)
    s1, e1 = centroid_shift(t, f, c1, P)
    s2, e2 = centroid_shift(t, f, c2, P)
    row = dict(sector=sector, n_points=len(t), period_d=P, depth1_pct=100 * d1, depth2_pct=100 * d2,
               crowdsap=crowd, centroid_col_shift_mpix=s1, centroid_col_err=e1,
               centroid_row_shift_mpix=s2, centroid_row_err=e2)
    return row, t, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--period", type=float, required=True, help="rough period in days (from the scan)")
    ap.add_argument("--out", default="followup")
    a = ap.parse_args()

    import socket
    socket.setdefaulttimeout(60)
    try:
        from astroquery.mast import Observations
        import pandas as pd
    except ImportError:
        sys.exit("Missing packages. Run:  python3 -m pip install astroquery astropy numpy pandas scipy matplotlib")

    os.makedirs(a.out, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp_fits")
    shutil.rmtree(tmp, ignore_errors=True)

    print("Asking MAST for all 2-minute light curves of TIC %d ..." % a.tic)
    obs = Observations.query_criteria(obs_collection="TESS", dataproduct_type="timeseries",
                                      target_name=str(a.tic), provenance_name="SPOC")
    print("MAST lists %d observations." % len(obs))
    if len(obs) == 0:
        sys.exit("No observations found for this TIC.")
    prods = Observations.get_product_list(obs)
    keep = [i for i, nme in enumerate(prods["productFilename"]) if str(nme).endswith("_lc.fits")]
    prods = prods[keep]
    print("Downloading %d light-curve files (about 2 MB each) ..." % len(prods))
    Observations.download_products(prods, download_dir=tmp, mrp_only=False)

    rows, T, F, S = [], [], [], []
    for root, _d, files in os.walk(tmp):
        for fn in sorted(files):
            if fn.endswith("_lc.fits"):
                try:
                    row, t, f = analyze_fits(os.path.join(root, fn), a.period)
                except Exception as exc:
                    print("  skipped %s (%s)" % (fn, str(exc)[:80]))
                    continue
                rows.append(row)
                T.append(t); F.append(f); S.append(np.full(len(t), row["sector"]))
    shutil.rmtree(tmp, ignore_errors=True)
    if not rows:
        sys.exit("Could not analyse any file.")

    df = pd.DataFrame(rows).sort_values("sector").reset_index(drop=True)
    t = np.concatenate(T); f = np.concatenate(F); s = np.concatenate(S)
    pmed = float(df["period_d"].median())
    # combined fit across sectors: first a coarse check around the median, then a fine one
    Pc = best_period(t, f, pmed - 2e-4, pmed + 2e-4, 2000)
    Pc = best_period(t, f, Pc - 2e-6, Pc + 2e-6, 800)

    csv = os.path.join(a.out, "TIC%d_summary.csv" % a.tic)
    df.to_csv(csv, index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for sec in sorted(np.unique(s)):
        m = s == sec
        ax[0].plot((t[m] / (2 * Pc)) % 1.0, f[m], ".", ms=1.2, alpha=0.35, label="S%d" % sec)
    ax[0].set_title("TIC %d folded at 2 x %.6f d (curve repeats twice; check whether the true period is 1x or 2x)" % (a.tic, Pc), fontsize=9)
    ax[0].set_xlabel("orbital phase"); ax[0].set_ylabel("normalised flux")
    ax[0].legend(markerscale=6, fontsize=7, ncol=2)
    ax[1].errorbar(df["sector"], df["period_d"], fmt="o", color="#d1495b")
    ax[1].set_title("period found in each sector (days)", fontsize=9)
    ax[1].set_xlabel("sector")
    fig.tight_layout()
    png = os.path.join(a.out, "TIC%d_folded.png" % a.tic)
    fig.savefig(png, dpi=110)

    pd.set_option("display.width", 200)
    print("\nPer-sector results")
    print(df.round(5).to_string(index=False))
    print("\nSectors analysed      : %s" % list(df["sector"]))
    print("Period spread (d)     : min %.6f  max %.6f" % (df["period_d"].min(), df["period_d"].max()))
    print("Combined period (d)   : %.7f   (CAUTION: check the plot. If the fold at 2x shows the curve repeating twice, this is already the orbital period)" % Pc)
    print("Depths (%%)            : primary %.2f +- %.2f, secondary %.2f +- %.2f" % (
        df["depth1_pct"].mean(), df["depth1_pct"].std(), df["depth2_pct"].mean(), df["depth2_pct"].std()))
    print("\nWrote %s and %s" % (csv, png))


if __name__ == "__main__":
    main()
