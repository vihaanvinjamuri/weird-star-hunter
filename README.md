# Weird Star Hunter

Unsupervised anomaly detection on public NASA TESS light curves, and the follow-up of one candidate, TIC 359791122.

## What this project does

TESS records the brightness of huge numbers of stars. This pipeline turns each light curve into 22 numbers, scores every star with three outlier detectors (Isolation Forest, Local Outlier Factor and a small autoencoder), and averages their ranks to find the most unusual stars. It was run on 1,003 two-minute light curves from TESS Sector 80.

The 60 highest-ranked clean stars were checked against SIMBAD, VSX, the TESS planet-candidate lists and the TESS Eclipsing Binary catalog. 37 were already known. Of the 23 with no variability label, one showed clear eclipses: TIC 359791122 (TYC 5687-417-1), a star of TESS magnitude 10.3 in Serpens.

## The candidate

TIC 359791122 is an eclipsing binary with an orbital period of 0.36308095 +/- 0.00000011 days (8.71 hours). Its eclipses come back unchanged in Sectors 91 and 92, and its image centroid does not move during eclipse. It was not found in VSX, the Gaia DR3 variability tables, ZTF, ATLAS, ASAS-SN, the TESS Eclipsing Binary catalog, SIMBAD or an ADS full-text search.

This is a statement about the sources searched, not proof that nobody has noticed the star. Six of the catalogs in the sweep could not be checked properly, and survey catalogs are incomplete.

## How to run it

One-time setup, in Terminal from this folder:

    python3 -m pip install -r requirements.txt

1. Download the light curves (needs internet; resumable, safe to stop and rerun)

       python3 step1_download.py --sector 80 --n 1500

2. Score every star

       python3 step2_analyze.py

3. Cross-match the top candidates against catalogs (needs internet)

       python3 step3_crossmatch.py

4. Follow up one candidate in the 2-minute SPOC data

       python3 step4_followup.py --tic 359791122 --period 0.3631

5. Check other sectors with QLP full-frame light curves

       python3 step5_other_sectors.py --tic 359791122 --ra 270.70257 --dec -13.88538 --period 0.363069

6. Sweep the variable-star catalogs around the candidate

       python3 step6_catalog_sweep.py --tic 359791122 --ra 270.70257 --dec -13.88538

7. Measure the ephemeris from individual eclipse times

       python3 step7_ephemeris.py --tic 359791122 --period 0.363081

8. Make the figures and photometry table for a VSX submission

       python3 step8_submission_figures.py --tic 359791122 --ra 270.70257 --dec -13.88538 --tmag 10.326

## What is in this repository

- `step1_download.py` to `step8_submission_figures.py`: the pipeline
- `tests/make_synthetic.py`: builds fake light curves with 10 planted oddities among 700 normal stars. All 10 landed in the top 25 of 710. This only shows the code works, not that it finds new stars.
- `output/ranked.csv`: all 1,003 stars ranked by anomaly score
- `output/crossmatch.csv`: catalog status of the top stars
- `output/plots/`: light-curve plots of the top candidates
- `followup/`: results for TIC 359791122, including the ephemeris, catalog sweep and multi-sector timings
- `followup/submission/`: the photometry table (BJD(TDB) and TESS-band magnitude) and the figures prepared for VSX

The raw TESS light curves are public at MAST and are not stored here. Step 1 downloads them.

## Limits

- Most top-ranked stars in real data are known variables. That is expected, and it is why the cross-match step exists.
- "Not found" means not found in the sources searched.
- The eclipse type (probably a contact binary) is tentative. I have no spectra.

## Credits

Data are from NASA's TESS mission, served by MAST. QLP light curves are from the MIT Quick-Look Pipeline. Catalog queries use SIMBAD, VizieR, VSX and the Gaia archive. I used Claude (Anthropic) to help write the code and draft the text, and I checked the results myself.

## License

MIT
