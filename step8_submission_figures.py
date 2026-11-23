#!/usr/bin/env python3
"""
Step 8: make the plots a VSX new-star submission asks for.

VSX wants (their data manual): a time-versus-magnitude plot, a phase plot for periodic stars,
and a finding chart (north and east marked, scale shown) when confusion is possible.

Outputs go to followup/submission/ :
    TIC<id>_lightcurve_time.png    brightness against time, one panel per TESS sector
    TIC<id>_phase.png              folded at the period, brightness in TESS magnitudes
    TIC<id>_finding_chart.png      sky image around the star with N and E arrows and a 1 arcmin scale bar
    TIC<id>_photometry.csv         the numbers behind the plots (time in BJD(TDB), TESS-band magnitude)

Usage:
    python3 step8_submission_figures.py --tic 359791122 --ra 270.70257 --dec -13.88538 --tmag 10.326
"""
import argparse
import glob
import os

import numpy as np

from step5_other_sectors import load_qlp

# from step 7 (BJD(TDB) of primary minimum and period in days)
T0_JD = 2460480.11071
PERIOD = 0.3630809


def to_mag(flux, tmag):
    """Differential TESS magnitudes: the star's catalog magnitude plus the change in brightness."""
    return tmag - 2.5 * np.log10(flux)


def binned(ph, y, nb=60):
    idx = (ph * nb).astype(int)
    x = (np.arange(nb) + 0.5) / nb
    m = np.array([np.median(y[idx == b]) if (idx == b).sum() > 3 else np.nan for b in range(nb)])
    return x, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--tmag", type=float, required=True)
    ap.add_argument("--period", type=float, default=PERIOD)
    ap.add_argument("--t0", type=float, default=T0_JD, help="BJD(TDB) of a primary minimum")
    ap.add_argument("--local", default="followup/_tmp_qlp")
    ap.add_argument("--out", default="followup/submission")
    a = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(a.out, exist_ok=True)
    files = sorted(glob.glob(os.path.join(a.local, "**", "*llc.fits"), recursive=True))
    if not files:
        raise SystemExit("No QLP files under %s. Run step 5 first." % a.local)
    data = {}
    for p in files:
        sec, t, f = load_qlp(p)
        if len(t) > 200:
            data[sec] = (t, f)
    secs = sorted(data)

    # ---- time series
    fig, axs = plt.subplots(len(secs), 1, figsize=(9, 2.6 * len(secs)), sharey=True)
    axs = np.atleast_1d(axs)
    for ax, s in zip(axs, secs):
        t, f = data[s]
        ax.plot(t + 2457000, to_mag(f, a.tmag), ".", ms=1.5, alpha=0.5, color="k")
        ax.set_ylabel("TESS mag")
        ax.set_title("TESS Sector %d (QLP full-frame photometry)" % s, fontsize=9, loc="left")
    axs[0].invert_yaxis()
    axs[-1].set_xlabel("BJD (TDB)")
    plt.tight_layout()
    fig.savefig(os.path.join(a.out, "TIC%d_lightcurve_time.png" % a.tic), dpi=140)
    plt.close(fig)

    # ---- phase plot
    t0 = a.t0 - 2457000
    T = np.concatenate([data[s][0] for s in secs])
    F = np.concatenate([data[s][1] for s in secs])
    ph = ((T - t0) / a.period + 0.25) % 1.0 - 0.25
    mag = to_mag(F, a.tmag)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k, s in enumerate(secs):
        t, f = data[s]
        p = ((t - t0) / a.period + 0.25) % 1.0 - 0.25
        ax.plot(p, to_mag(f, a.tmag), ".", ms=1.5, alpha=0.25, label="Sector %d" % s)
    pb = ((T - t0) / a.period) % 1.0
    x, m = binned(pb, mag)
    ax.plot(np.r_[x, x + 1.0] - 0.0, np.r_[m, m], "k-", lw=1.3, label="binned median")
    ax.set_xlim(-0.25, 1.25)
    ax.invert_yaxis()
    ax.set_xlabel("phase (P = %.7f d, primary minimum BJD %.5f)" % (a.period, a.t0))
    ax.set_ylabel("TESS mag")
    ax.legend(fontsize=8, markerscale=5, loc="lower right")
    plt.tight_layout()
    fig.savefig(os.path.join(a.out, "TIC%d_phase.png" % a.tic), dpi=140)
    plt.close(fig)

    # ---- numbers
    base = float(np.nanpercentile(mag, 20))
    faint = float(np.nanmax(m))
    bright = float(np.nanmin(m))
    with open(os.path.join(a.out, "TIC%d_photometry.csv" % a.tic), "w") as fh:
        fh.write("BJD_TDB,TESS_mag,sector\n")
        for s in secs:
            t, f = data[s]
            for ti, mi in zip(t + 2457000, to_mag(f, a.tmag)):
                fh.write("%.5f,%.4f,%d\n" % (ti, mi, s))
    print("\nBrightness range in the TESS band (binned medians): %.3f (brightest) to %.3f (faintest)" % (bright, faint))
    print("Amplitude about %.3f mag. Catalog TESS magnitude of the star: %.3f" % (faint - bright, a.tmag))
    print("Wrote time-series plot, phase plot and photometry table to %s" % a.out)

    # ---- finding chart (needs internet; skipped cleanly if it fails)
    try:
        from astroquery.skyview import SkyView
        from astropy.coordinates import SkyCoord
        from astropy.wcs import WCS
        from astropy.visualization import ZScaleInterval
        import astropy.units as u
        c = SkyCoord(a.ra, a.dec, unit="deg")
        img = None
        for survey in ("DSS2 Red", "DSS"):
            try:
                img = SkyView.get_images(position=c, survey=[survey], radius=2.0 * u.arcmin, pixels=700)[0][0]
                used = survey
                break
            except Exception:
                img = None
        if img is None:
            raise RuntimeError("no sky image could be downloaded")
        wcs = WCS(img.header)
        arr = np.array(img.data, float)
        lo, hi = ZScaleInterval().get_limits(arr)
        fig = plt.figure(figsize=(6.2, 6.2))
        ax = fig.add_subplot(111, projection=wcs)
        ax.imshow(arr, origin="lower", cmap="gray_r", vmin=lo, vmax=hi)
        x0, y0 = wcs.world_to_pixel(c)
        ax.plot([x0 - 60, x0 - 25], [y0, y0], color="red", lw=1.2)
        ax.plot([x0 + 25, x0 + 60], [y0, y0], color="red", lw=1.2)
        ax.plot([x0, x0], [y0 - 60, y0 - 25], color="red", lw=1.2)
        ax.plot([x0, x0], [y0 + 25, y0 + 60], color="red", lw=1.2)
        scale = np.abs(wcs.proj_plane_pixel_scales()[0].to(u.arcsec).value) if hasattr(wcs.proj_plane_pixel_scales()[0], "to") \
            else abs(wcs.proj_plane_pixel_scales()[0]) * 3600.0
        n60 = 60.0 / scale
        ny, nx = arr.shape
        bx, by = nx * 0.08, ny * 0.07
        ax.plot([bx, bx + n60], [by, by], color="k", lw=3)
        ax.text(bx, by + ny * 0.015, "1 arcmin", fontsize=9)
        ax.annotate("", xy=(nx * 0.90, ny * 0.93), xytext=(nx * 0.90, ny * 0.80),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
        ax.annotate("", xy=(nx * 0.76, ny * 0.80), xytext=(nx * 0.90, ny * 0.80),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.5))
        ax.text(nx * 0.90, ny * 0.95, "N", ha="center", fontsize=11)
        ax.text(nx * 0.735, ny * 0.80, "E", ha="right", va="center", fontsize=11)
        ax.set_title("TIC %d  (%s, north up, east left)" % (a.tic, used), fontsize=9)
        ax.set_xlabel("RA")
        ax.set_ylabel("Dec")
        fig.savefig(os.path.join(a.out, "TIC%d_finding_chart.png" % a.tic), dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("Wrote finding chart (%s)." % used)
    except Exception as e:
        print("Finding chart skipped: %s" % str(e)[:120])


if __name__ == "__main__":
    main()
