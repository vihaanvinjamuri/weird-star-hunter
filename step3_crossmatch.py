#!/usr/bin/env python3
"""
STEP 3 (run this on your Mac, it needs internet).

For each top candidate from step2, asks public catalogs whether anyone has
already flagged the star:
  - SIMBAD        (the main astronomy database; gives the object type)
  - VSX           (AAVSO's Variable Star Index, via VizieR)
  - TOI           (NASA's official TESS planet candidates)
  - CTOI          (community TESS planet candidates on ExoFOP)

IMPORTANT: "no match" here only means "not in these catalogs". Before claiming
anything is new you still need to do a manual literature check (links are in the output).

Input : output/crossmatch_input.csv   (made by step2)
Output: output/crossmatch.csv
"""
import argparse
import io
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import requests

SIMBAD = ["https://simbad.cds.unistra.fr/simbad/sim-tap/sync",
          "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync"]      # mirror
VIZIER = ["https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync",
          "https://tapvizier.cfa.harvard.edu/TAPVizieR/tap/sync"]    # mirror
DEAD = set()          # endpoints that could not even be reached (skipped afterwards)
LAST_ERR = {}         # first full error per service, printed at the end
TOI_URL = ("https://exoplanetarchive.ipac.caltech.edu/TAP/sync?"
           "query=select+tid,toi,tfopwg_disp+from+toi&format=csv")
CTOI_URL = "https://exofop.ipac.caltech.edu/tess/download_ctoi.php?sort=ctoi&output=csv"

# SIMBAD object-type codes that mean "known variable / eclipsing / flaring star"
VAR_TYPES = {
    "V*", "EB*", "El*", "Al*", "bL*", "WU*", "RR*", "Ce*", "Pu*", "dS*", "gD*", "bC*", "RV*",
    "Mi*", "Ir*", "Or*", "TT*", "Fl*", "Er*", "Ro*", "BY*", "RS*", "Psr", "WV*", "LP*", "AB*",
    "Be*", "CV*", "No*", "SN*", "Sy*", "XB*", "HXB", "LXB", "Pe*", "SB*", "**",
}


def tap(urls, query, label):
    last = None
    for url in urls:
        if url in DEAD:
            continue
        try:
            r = requests.post(url, data={"REQUEST": "doQuery", "LANG": "ADQL", "FORMAT": "csv", "QUERY": query},
                              timeout=(8, 45))
            r.raise_for_status()
            return pd.read_csv(io.StringIO(r.text))
        except requests.exceptions.ReadTimeout as exc:
            last = exc                         # server is slow, not gone: allow a retry on the next star
            LAST_ERR.setdefault(label + " " + url, repr(exc)[:300])
        except (requests.ConnectionError, requests.Timeout) as exc:
            DEAD.add(url)                      # unreachable from here: do not waste time on it again
            last = exc
            LAST_ERR.setdefault(label + " " + url, repr(exc)[:300])
        except Exception as exc:
            last = exc
            if getattr(getattr(exc, "response", None), "status_code", 0) == 404:
                DEAD.add(url)                  # endpoint moved or gone
            LAST_ERR.setdefault(label + " " + url, repr(exc)[:300] + " | " + getattr(getattr(exc, "response", None), "text", "")[:200])
    raise last or RuntimeError("no reachable %s endpoint" % label)


def simbad_lookup(tic, ra, dec):
    q = ("SELECT b.main_id, b.otype FROM basic AS b JOIN ident AS i ON i.oidref = b.oid "
         "WHERE i.id = 'TIC %d'" % tic)
    try:
        df = tap(SIMBAD, q, 'SIMBAD')
        if len(df):
            return df.iloc[0]["main_id"], str(df.iloc[0]["otype"])
        q = ("SELECT main_id, otype FROM basic WHERE CONTAINS(POINT('ICRS', ra, dec), "
             "CIRCLE('ICRS', %f, %f, %f)) = 1" % (ra, dec, 10.0 / 3600))
        df = tap(SIMBAD, q, 'SIMBAD')
        if len(df):
            return df.iloc[0]["main_id"], str(df.iloc[0]["otype"])
    except Exception as exc:
        return "ERROR", str(exc)[:60]
    return "", ""


def vsx_aavso(ra, dec):
    r = requests.get("https://www.aavso.org/vsx/index.php",
                     params={"view": "api.list", "ra": ra, "dec": dec, "radius": 10.0 / 3600, "format": "json"},
                     timeout=(8, 30))
    r.raise_for_status()
    objs = r.json().get("VSXObjects", {})
    objs = objs.get("VSXObject", []) if isinstance(objs, dict) else objs
    if isinstance(objs, dict):
        objs = [objs]
    if objs:
        return str(objs[0].get("Name", "?")), str(objs[0].get("VariabilityType", "?"))
    return "", ""


def vsx_astroquery(ra, dec, server):
    from astroquery.vizier import Vizier
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    v = Vizier(columns=["Name", "Type"], timeout=30)
    v.VIZIER_SERVER = server
    res = v.query_region(SkyCoord(ra, dec, unit="deg"), radius=10 * u.arcsec, catalog="B/vsx/vsx")
    if len(res) and len(res[0]):
        return str(res[0][0]["Name"]), str(res[0][0]["Type"])
    return "", ""


def vsx_lookup(ra, dec):
    q = ('SELECT "Name","Type" FROM "B/vsx/vsx" WHERE CONTAINS(POINT(\'ICRS\',"RAJ2000","DEJ2000"),'
         "CIRCLE('ICRS',%f,%f,%f))=1" % (ra, dec, 10.0 / 3600))
    try:
        df = tap(VIZIER, q, 'VSX')
        if len(df):
            return str(df.iloc[0]["Name"]), str(df.iloc[0]["Type"])
        return "", ""
    except Exception:
        pass
    for name, server in [("vizier-cds", "vizier.cds.unistra.fr"), ("vizier-cfa", "vizier.cfa.harvard.edu")]:
        if name in DEAD:
            continue
        try:
            return vsx_astroquery(ra, dec, server)
        except Exception as exc:
            DEAD.add(name)                         # do not retry a route that failed
            LAST_ERR.setdefault("VSX " + name, repr(exc)[:300])
    return "ERROR", "all VSX routes failed"


def ebcat_lookup(tic):
    """Is the star in the TESS Eclipsing Binary catalog (Villanova)? True, False, or None if unreachable."""
    try:
        r = requests.get("https://tessebs.villanova.edu/%010d" % tic, timeout=(8, 30))
        if r.status_code == 404 or "not in the catalog" in r.text:
            return False
        r.raise_for_status()
        return ("Orbital period" in r.text) or ("Time of superior conjunction" in r.text)
    except Exception as exc:
        LAST_ERR.setdefault("EBcat", repr(exc)[:300])
        return None


def load_tic_set(url, tic_col_candidates, label):
    try:
        r = requests.get(url, timeout=90)
        r.raise_for_status()
        df = pd.read_csv(io.StringIO(r.text))
        for c in df.columns:
            if c.strip().lower() in tic_col_candidates:
                return set(pd.to_numeric(df[c], errors="coerce").dropna().astype(int)), True
        print("  could not find a TIC column in %s list; skipping it" % label)
    except Exception as exc:
        print("  %s list unavailable (%s); skipping it" % (label, str(exc)[:80]))
    return set(), False


def status(row):
    if row["in_toi"]:
        return "TOI (official TESS planet candidate)"
    if row["in_ctoi"]:
        return "CTOI (community planet candidate)"
    if row.get("in_ebcat") is True:
        return "Known eclipsing binary (TESS EB catalog)"
    if row["vsx_name"] not in ("", "ERROR"):
        return "Known variable (VSX)"
    if row["simbad_type"] in VAR_TYPES:
        return "Known variable (SIMBAD)"
    if row["simbad_id"] == "ERROR":
        return "CHECK FAILED (rerun)"
    note = " (VSX unreachable)" if row["vsx_name"] == "ERROR" else ""
    if row["simbad_id"]:
        return "In SIMBAD, no variability type%s: check literature" % note
    return "No SIMBAD/VSX/TOI match%s: check literature" % note


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="output/crossmatch_input.csv")
    ap.add_argument("--out", default="output/crossmatch.csv")
    ap.add_argument("--top", type=int, default=100, help="only check the best N candidates")
    ap.add_argument("--workers", type=int, default=3, help="parallel lookups")
    a = ap.parse_args()
    if not os.path.exists(a.inp):
        sys.exit("Missing %s. Run step2 first." % a.inp)
    df = pd.read_csv(a.inp).head(a.top)
    print("Cross-matching %d candidates (%d at a time)..." % (len(df), a.workers))

    toi_set, ok1 = load_tic_set(TOI_URL, {"tid", "tic id", "tic"}, "TOI")
    ctoi_set, ok2 = load_tic_set(CTOI_URL, {"tic id", "tic", "tid"}, "CTOI")
    print("Planet-candidate lists loaded. Now checking each star...")

    def work(r):
        tic = int(r["tic"])
        sid, sty = simbad_lookup(tic, float(r["ra"]), float(r["dec"]))
        vn, vt = vsx_lookup(float(r["ra"]), float(r["dec"]))
        rec = dict(r)
        rec.update(simbad_id=sid, simbad_type=sty, vsx_name=vn, vsx_type=vt,
                   in_toi=tic in toi_set, in_ctoi=tic in ctoi_set, in_ebcat=ebcat_lookup(tic))
        rec["status"] = status(rec)
        rec["simbad_link"] = "https://simbad.cds.unistra.fr/simbad/sim-id?Ident=TIC+%d" % tic
        rec["exofop_link"] = "https://exofop.ipac.caltech.edu/tess/target.php?id=%d" % tic
        rec["ads_link"] = "https://ui.adsabs.harvard.edu/search/q=%22TIC%20{}%22".format(tic)
        return rec

    rows = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for k, rec in enumerate(ex.map(work, [r for _, r in df.iterrows()])):
            rows.append(rec)
            if (k + 1) % 5 == 0 or k + 1 == len(df):
                print("  %d / %d  (%.0fs)" % (k + 1, len(df), time.time() - t0))
                pd.DataFrame(rows).to_csv(a.out + ".partial", index=False)   # progress is saved as we go

    for k, v in LAST_ERR.items():
        print("  first error from %s: %s" % (k, v))
    out = pd.DataFrame(rows)
    out.to_csv(a.out, index=False)
    print("\nWrote %s" % a.out)
    print(out["status"].value_counts().to_string())
    if not (ok1 and ok2):
        print("\nNote: the TOI and/or CTOI list could not be loaded, so planet-candidate status is incomplete.")


if __name__ == "__main__":
    main()
