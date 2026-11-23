#!/usr/bin/env python3
"""
STEP 1 (run this on your Mac, which has normal internet access).

Downloads TESS 2-minute SPOC light curves for ONE sector, keeps only the small
arrays we need (time, flux, quality + star info) as .npz files in data/, and
deletes the big FITS files as it goes so your disk does not fill up.
It is resumable: if you stop it and run it again, it skips stars it already has.

Examples
  python3 step1_download.py --sector 80 --test        # grab 3 stars to check everything works
  python3 step1_download.py --sector 80 --n 1500      # the real run
"""
import argparse
import os
import random
import shutil
import sys
import time

import numpy as np


def _num(header, key, cast=float):
    """Read a FITS header value, returning NaN if it is missing."""
    try:
        v = header.get(key)
        return cast(v) if v is not None else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def extract(fits_path, out_dir):
    """Turn one SPOC light-curve FITS file into a small .npz. Returns the npz path or None."""
    from astropy.io import fits

    with fits.open(fits_path) as hdul:
        h0, h1, d = hdul[0].header, hdul[1].header, hdul[1].data
        tic = int(h0["TICID"])
        sector = int(h0["SECTOR"])
        t = np.array(d["TIME"], dtype=np.float64)
        f = np.array(d["PDCSAP_FLUX"], dtype=np.float32)
        e = np.array(d["PDCSAP_FLUX_ERR"], dtype=np.float32)
        q = np.array(d["QUALITY"], dtype=np.int32)
        meta = dict(
            tic=tic,
            sector=sector,
            ra=_num(h0, "RA_OBJ"),
            dec=_num(h0, "DEC_OBJ"),
            tmag=_num(h0, "TESSMAG"),
            teff=_num(h0, "TEFF"),
            radius=_num(h0, "RADIUS"),
            crowdsap=_num(h1, "CROWDSAP"),
        )
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "tic%d_s%d.npz" % (tic, sector))
    np.savez_compressed(out, t=t, f=f, e=e, q=q, **meta)
    return out


def harvest(tmp_dir, out_dir):
    """Convert every FITS file found in tmp_dir to .npz (also rescues files left by an interrupted run)."""
    n = 0
    for root, _dirs, files in os.walk(tmp_dir):
        for fn in files:
            if fn.endswith("_lc.fits"):
                try:
                    extract(os.path.join(root, fn), out_dir)
                    n += 1
                except Exception as exc:  # a half-downloaded file can fail; skip it
                    print("  skipped %s (%s)" % (fn, exc))
    return n


def count_saved(out_dir):
    return len([x for x in os.listdir(out_dir) if x.endswith(".npz")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", type=int, required=True)
    ap.add_argument("--n", type=int, default=1500, help="how many stars to download")
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--test", action="store_true", help="only download 3 stars")
    a = ap.parse_args()

    try:
        from astroquery.mast import Observations
    except ImportError:
        sys.exit("Missing packages. Run:  python3 -m pip install astroquery astropy numpy pandas")

    n_target = 3 if a.test else a.n
    out_dir = os.path.join(a.out, "sector%d" % a.sector)
    os.makedirs(out_dir, exist_ok=True)
    tmp = os.path.join(a.out, "_tmp_fits")

    print("Asking MAST for sector %d light curves..." % a.sector)
    obs = Observations.query_criteria(
        obs_collection="TESS",
        dataproduct_type="timeseries",
        sequence_number=a.sector,
        provenance_name="SPOC",
    )
    print("MAST lists %d observations for this sector." % len(obs))
    if len(obs) == 0:
        sys.exit("No observations found. Try a different --sector (for example 70 or 90).")

    idx = list(range(len(obs)))
    random.Random(a.seed).shuffle(idx)

    if os.path.isdir(tmp):
        harvest(tmp, out_dir)           # rescue anything left by an interrupted run
        shutil.rmtree(tmp, ignore_errors=True)
    have = count_saved(out_dir)
    print("Already have %d stars. Target %d." % (have, n_target))
    pos = 0
    while have < n_target and pos < len(idx):
        batch_idx = idx[pos:pos + a.batch]
        pos += a.batch
        try:
            prods = Observations.get_product_list(obs[batch_idx])
            keep = [
                i for i, name in enumerate(prods["productFilename"])
                if str(name).endswith("_lc.fits")
            ]
            if not keep:
                continue
            prods = prods[keep]
            shutil.rmtree(tmp, ignore_errors=True)
            Observations.download_products(prods, download_dir=tmp, mrp_only=False)
        except Exception as exc:
            print("  hiccup (%s); keeping what was downloaded and carrying on" % str(exc)[:80])
            time.sleep(5)
        harvest(tmp, out_dir)           # saves whatever arrived, even from a partly failed batch
        shutil.rmtree(tmp, ignore_errors=True)
        have = count_saved(out_dir)
        print("  %d / %d stars saved" % (have, n_target))

    print("Done. %d stars in %s" % (have, out_dir))


if __name__ == "__main__":
    main()
