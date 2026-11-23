#!/usr/bin/env python3
"""
Step 6: is our star already listed in any variable-star catalog?

We ask VizieR (the big astronomy catalog service) for every row within a few arcseconds of our star,
in the main variable-star catalogs, then do a wide keyword search for any other variable-star or
eclipsing-binary catalog that has a row there.

Usage:
    python3 step6_catalog_sweep.py --tic 359791122 --ra 270.70257 --dec -13.88538

A match close to our star (about 1-2 arcsec) is a possible identification of the same star.
A match a few arcsec away may be the faint neighbour TIC 1657033712 (2.6 arcsec away), not our star.
"""
import argparse
import os

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord

SERVERS = ["vizier.cds.unistra.fr", "vizier.cfa.harvard.edu"]

# (VizieR catalog id, plain name). If an id is wrong the script just reports "error", it does not stop.
CATALOGS = [
    ("B/vsx/vsx", "AAVSO VSX (International Variable Star Index)"),
    ("B/gcvs", "GCVS (General Catalogue of Variable Stars)"),
    ("I/358/veb", "Gaia DR3 eclipsing binary candidates"),
    ("I/358/vclassre", "Gaia DR3 variability classification"),
    ("II/366", "ASAS-SN catalog of variable stars"),
    ("J/ApJS/249/18", "ZTF periodic variables (Chen+ 2020)"),
    ("J/AJ/156/241", "ATLAS variable stars (Heinze+ 2018)"),
    ("J/ApJS/213/9", "CRTS periodic variables (Drake+ 2014)"),
    ("J/AJ/155/39", "KELT variable stars (Oelkers+ 2018)"),
    ("II/264", "ASAS-3 variable stars"),
    ("J/ApJS/237/28", "WISE variable stars (Chen+ 2018)"),
]

LINES = []


def say(s=""):
    print(s)
    LINES.append(s)


def make_vizier(server, **kw):
    from astroquery.vizier import Vizier
    v = Vizier(columns=["*", "_RAJ2000", "_DEJ2000"], row_limit=3, timeout=40, **kw)
    v.VIZIER_SERVER = server
    return v


def sep_arcsec(row, centre):
    try:
        ra, dec = row["_RAJ2000"], row["_DEJ2000"]
        if isinstance(ra, (str, bytes)):
            c = SkyCoord(str(ra), str(dec), unit=(u.hourangle, u.deg))
        else:
            c = SkyCoord(float(ra), float(dec), unit="deg")
        return c.separation(centre).arcsec
    except Exception:
        return float("nan")


def short(tab):
    cols = [c for c in tab.colnames if not c.startswith("_")][:9]
    r = tab[0]
    return ", ".join("%s=%s" % (c, str(r[c])[:20]) for c in cols)


def pick_server(centre):
    for s in SERVERS:
        try:
            v = make_vizier(s)
            v.query_region(centre, radius=5 * u.arcsec, catalog="B/vsx/vsx")
            return s
        except Exception as e:
            say("(server %s not usable: %s)" % (s, str(e)[:80]))
    return None


def sweep(centre, server, radius_arcsec):
    v = make_vizier(server)
    found = []
    say("\nExplicit catalogs (radius %g arcsec)" % radius_arcsec)
    say("-" * 78)
    for cat, name in CATALOGS:
        try:
            res = v.query_region(centre, radius=radius_arcsec * u.arcsec, catalog=cat)
        except Exception as e:
            say("%-44s ERROR (%s)" % (name, str(e)[:60]))
            continue
        hits = [(k, res[k]) for k in res.keys() if len(res[k])]
        if not hits:
            say("%-44s no match" % name)
            continue
        for k, t in hits:
            s = sep_arcsec(t[0], centre)
            say("%-44s MATCH in %s: %d row(s), nearest %.1f arcsec" % (name, k, len(t), s))
            say("      " + short(t))
            found.append((name, s))
    return found


def _flatten(x):
    if isinstance(x, str):
        yield x
    elif isinstance(x, dict):
        for k, v in x.items():
            yield from _flatten(k)
            yield from _flatten(v)
    elif isinstance(x, (list, tuple, set)):
        for v in x:
            yield from _flatten(v)


def find_keywords(server):
    """Ask VizieR which keywords exist, keep the ones about variables / eclipsing binaries."""
    try:
        from astroquery.vizier import Vizier
        v = Vizier()
        v.VIZIER_SERVER = server
        allk = set(_flatten(v.get_available_keywords()))
        keep = sorted(k for k in allk if any(w in k.lower() for w in ("eclips", "variable", "contact")))
        if keep:
            return keep
    except Exception as e:
        say("(could not list VizieR keywords: %s)" % str(e)[:70])
    return ["Binaries: eclipsing", "Stars: variable"]


def keyword_sweep(centre, server, radius_arcsec):
    say("\nWide net: any other catalog about variable stars or eclipsing binaries with a row nearby")
    say("-" * 78)
    kws = find_keywords(server)
    say("keywords used: %s" % ", ".join(kws))
    seen = set()
    for kw in kws:
        try:
            v = make_vizier(server, keywords=[kw])
            res = v.query_region(centre, radius=radius_arcsec * u.arcsec)
        except Exception as e:
            say("keyword %s: ERROR (%s)" % (kw, str(e)[:60]))
            continue
        for k in res.keys():
            t = res[k]
            if k in seen or not len(t):
                continue
            seen.add(k)
            desc = str(t.meta.get("description", ""))[:70] if hasattr(t, "meta") else ""
            say("%-28s %d row(s), nearest %.1f arcsec  %s" % (k, len(t), sep_arcsec(t[0], centre), desc))
    if not seen:
        say("nothing found")
    return seen


def control(centre, server):
    """Sanity check: 'no match' is only meaningful if the catalog id is real and covers this area.
    Sparse catalogs need a wide circle, so we use 1 degree."""
    say("\nControl: does each catalog have any rows within 1 degree? (proves the id is real)")
    say("-" * 78)
    v = make_vizier(server)
    v.ROW_LIMIT = 1
    for cat, name in CATALOGS:
        try:
            res = v.query_region(centre, radius=1 * u.deg, catalog=cat)
            n = sum(len(res[k]) for k in res.keys())
            say("%-44s %s" % (name, "ok, has data nearby" if n else "EMPTY: wrong id or no coverage here (inconclusive)"))
        except Exception as e:
            say("%-44s ERROR (%s)" % (name, str(e)[:60]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tic", type=int, required=True)
    ap.add_argument("--ra", type=float, required=True)
    ap.add_argument("--dec", type=float, required=True)
    ap.add_argument("--radius", type=float, default=5.0, help="search radius in arcsec")
    ap.add_argument("--out", default="followup")
    a = ap.parse_args()
    centre = SkyCoord(a.ra, a.dec, unit="deg")

    say("Catalog sweep for TIC %d at RA %.5f Dec %.5f" % (a.tic, a.ra, a.dec))
    server = pick_server(centre)
    if server is None:
        raise SystemExit("No VizieR server reachable right now. Try again later or send me this message.")
    say("Using VizieR server: %s" % server)

    found = sweep(centre, server, a.radius)
    control(centre, server)

    say("\nSummary")
    say("-" * 78)
    if found:
        for name, s in found:
            say("  %s: match %.1f arcsec away" % (name, s))
        say("A match means we must check whether it is our star or the neighbour before claiming anything.")
    else:
        say("No explicit-catalog match within %g arcsec." % a.radius)
    say("Reminder: 'not found' only covers the catalogs listed above, not the whole literature.")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "TIC%d_catalog_sweep.txt" % a.tic)
    with open(path, "w") as fh:
        fh.write("\n".join(LINES) + "\n")
    print("\nWrote %s" % path)


if __name__ == "__main__":
    main()
