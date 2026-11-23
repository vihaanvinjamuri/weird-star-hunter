#!/usr/bin/env python3
"""
Self-test helper (NOT real data). Builds fake TESS-like light curves, including a few
deliberately weird ones, so we can check the pipeline finds them before using real data.
Also builds one fake SPOC FITS file to test the FITS-reading code in step1.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def make_t(rng):
    t1 = np.arange(0, 13.2, 2 / 1440.0)
    t2 = np.arange(14.2, 27.0, 2 / 1440.0)
    t = np.concatenate([t1, t2]) + 3000.0
    keep = rng.random(len(t)) > 0.03
    return t[keep]


def synth(kind, rng):
    t = make_t(rng)
    x = t - t[0]
    noise = rng.choice([4e-4, 8e-4, 2e-3, 6e-3])
    f = 1 + rng.normal(0, noise, len(t))
    if kind == "rotator":
        P = rng.uniform(0.5, 12)
        f += rng.uniform(0.002, 0.03) * np.sin(2 * np.pi * x / P + rng.uniform(0, 6))
    elif kind == "pulsator":
        P = rng.uniform(0.05, 0.3)
        f += rng.uniform(0.005, 0.05) * np.sin(2 * np.pi * x / P)
    elif kind == "eclipsing":
        P = rng.uniform(0.6, 8)
        ph = (x / P) % 1.0
        f -= rng.uniform(0.05, 0.4) * (np.abs(ph - 0.5) < 0.04)
    elif kind == "flare":
        for _ in range(rng.integers(2, 8)):
            c = rng.uniform(1, 26)
            amp = rng.uniform(0.01, 0.1)
            d = x - c
            f += np.where((d > 0) & (d < 0.3), amp * np.exp(-np.clip(d, 0, None) / 0.02), 0)
    elif kind == "weird_bigdip":      # one deep, asymmetric, non-repeating dip
        c = rng.uniform(8, 20)
        d = x - c
        f -= 0.18 * np.exp(-0.5 * ((d + 0.5 * (d > 0) * d) / 0.35) ** 2)
    elif kind == "weird_step":        # sudden permanent change in brightness
        f += 0.03 * (x > rng.uniform(8, 20))
    elif kind == "weird_burst":       # irregular bursts, no flare shape
        for _ in range(6):
            c = rng.uniform(1, 26)
            f += 0.04 * np.exp(-0.5 * ((x - c) / rng.uniform(0.2, 0.7)) ** 2)
    elif kind == "weird_growing":     # oscillation that keeps growing
        f += 0.0005 * (1 + x / 2) * np.sin(2 * np.pi * x / 0.9)
    elif kind == "weird_multidip":    # several irregular dips of different depth
        for _ in range(5):
            c = rng.uniform(1, 26)
            f -= rng.uniform(0.02, 0.08) * np.exp(-0.5 * ((x - c) / rng.uniform(0.1, 0.4)) ** 2)
    return t, f.astype(np.float32)


def build(outdir, n_normal=700, seed=1):
    rng = np.random.default_rng(seed)
    os.makedirs(outdir, exist_ok=True)
    kinds = (["flat"] * int(n_normal * 0.5) + ["rotator"] * int(n_normal * 0.25) +
             ["pulsator"] * int(n_normal * 0.06) + ["eclipsing"] * int(n_normal * 0.1) +
             ["flare"] * int(n_normal * 0.09))
    weird = ["weird_bigdip", "weird_step", "weird_burst", "weird_growing", "weird_multidip"] * 2
    kinds += weird
    truth = {}
    for i, k in enumerate(kinds):
        t, f = synth(k, rng)
        tic = 900000 + i
        np.savez_compressed(
            os.path.join(outdir, "tic%d_s80.npz" % tic),
            t=t, f=f, e=np.full(len(t), 1e-3, dtype=np.float32),
            q=np.zeros(len(t), dtype=np.int32),
            tic=tic, sector=80, ra=rng.uniform(0, 360), dec=rng.uniform(-80, 80),
            tmag=rng.uniform(8, 13.5), teff=5500.0, radius=1.0, crowdsap=0.97)
        truth[tic] = k
    return truth


def build_fake_fits(path):
    from astropy.io import fits
    rng = np.random.default_rng(5)
    t, f = synth("rotator", rng)
    cols = [
        fits.Column(name="TIME", format="D", array=t),
        fits.Column(name="PDCSAP_FLUX", format="E", array=f * 1000),
        fits.Column(name="PDCSAP_FLUX_ERR", format="E", array=np.full(len(t), 1.0, dtype=np.float32)),
        fits.Column(name="QUALITY", format="J", array=np.zeros(len(t), dtype=np.int32)),
    ]
    h0 = fits.PrimaryHDU()
    h0.header["TICID"] = 123456
    h0.header["SECTOR"] = 80
    h0.header["RA_OBJ"] = 150.1
    h0.header["DEC_OBJ"] = -30.2
    h0.header["TESSMAG"] = 10.5
    h0.header["TEFF"] = 5600.0
    h0.header["RADIUS"] = 1.1
    h1 = fits.BinTableHDU.from_columns(cols)
    h1.header["CROWDSAP"] = 0.95
    fits.HDUList([h0, h1]).writeto(path, overwrite=True)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "synthetic_data")
    truth = build(out)
    import json
    with open(os.path.join(out, "truth.json"), "w") as fh:
        json.dump(truth, fh)
    print("wrote %d synthetic stars to %s" % (len(truth), out))
