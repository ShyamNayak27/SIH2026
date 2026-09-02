# What the real data actually says

Written after building `ner_landslide_v1.csv` from live sources on 2026-09-02.
Read this before writing any modelling code. The headline is not good news, and
it changes what the team should build.

## 1. The open inventory cannot support a terrain-based susceptibility model

Univariate separation on the 1,029-row dataset, positive median vs negative
median, with AUC of that single feature:

| feature | positives | negatives | AUC |
|---|---|---|---|
| **slope_deg** | 13.4° | 13.5° | **0.507** |
| tri | 18.8 | 18.9 | 0.506 |
| twi | 13.6 | 13.6 | 0.478 |
| relief_500m | 244 m | 228 m | 0.535 |
| elevation | 902 m | 648 m | 0.558 |
| **rain_3d** | **49.3 mm** | **30.0 mm** | **0.623** |
| rain_1d | 16.2 mm | 8.5 mm | 0.611 |
| api | 118 mm | 105 mm | 0.576 |

An AUC of 0.507 on slope means the recorded landslide locations are, in slope
terms, indistinguishable from random points in the same eight states.

The cause is location accuracy, and the catalogue is honest about it. Of the 344
usable NER events, only **4 are "exact" and 41 are accurate to 1 km**; 185 are
accurate to 10–100 km. These are media-derived reports, so the coordinate is
typically a town, a highway marker or a district centroid — not the failure
scar. At 10 km, the recorded point and the actual slope have no terrain in
common.

**The rule this gives us: a feature is only usable if its own resolution is
coarser than the inventory's location error.** IMD rainfall at 0.25° (~28 km) is
coarser than a 10 km location error, so rainfall features survive. Terrain at
90 m does not, and its columns are essentially noise in this build.

## 2. The rainfall signal is real

Positives saw roughly twice the same-day rainfall of negatives (16.2 mm vs
8.5 mm) and 1.6× the 3-day antecedent total. That is the physical relationship
we set out to capture, and it holds on data nobody tuned.

## 3. Be suspicious of the headline AUC

Baseline on the spatially-blocked test set (172 rows, base rate 0.297):

| features | logistic regression | gradient boosting |
|---|---|---|
| terrain only | AUC 0.614 / AP 0.418 | AUC 0.749 / AP 0.537 |
| rainfall only | AUC 0.491 / AP 0.343 | AUC 0.637 / AP 0.470 |
| terrain + rainfall | AUC 0.602 / AP 0.384 | AUC 0.749 / AP 0.507 |

Terrain-only boosting reaches 0.749 while every individual terrain feature sits
at ~0.50. A model cannot conjure signal from noise, so it is not learning slope
physics — it is using the multivariate terrain signature as a fingerprint for
*where in the region it is*, and landslide reports cluster in particular
districts. That is **reporting bias**, not hazard. Quote this number in the
deck and you will be asked, correctly, why slope alone gives 0.507.

## 4. Consequences for each track

- **Charu + Shourya** — do not present a 90 m susceptibility map from this
  inventory. Build the model at the resolution the labels support: district ×
  day, with rainfall as the primary driver. Report AP alongside AUC and always
  quote the terrain-only baseline beside the full model, so the reporting-bias
  gap is visible rather than hidden.
- **Rhea** — the horizon question is answered: 344 events, every one dated, over
  2007–2017, which is enough for rainfall-threshold triggers but far too few for
  a sequence model. Recommend intensity-duration thresholds over a static
  susceptibility layer.
- **Neil** — this raises the value of the CV track, not lowers it. Satellite scar
  detection produces polygons with metre-accurate geometry, which is exactly what
  the inventory lacks. Landslide4Sense fine-tuning is the highest-leverage thing
  on the board.
- **Stuti** — explanations must be built on rainfall and exposure, because those
  are the columns that carry signal. "180 mm in 3 days over a district with 40
  km of hill highway" is defensible; "this 90 m pixel is steep" is not, from this
  data.
- **Shyam (me)** — the GSI Bhukosh inventory moves from nice-to-have to the
  critical path. Its polygons are surveyed, not media-derived, and they are what
  makes the terrain half of the platform real.

## 5. A rejected design, recorded so nobody re-adds it

The first build sampled negatives only from slope ≥ 8°, to avoid the classic
"flat is safe" shortcut. Inside the eight NER states that inverted the
relationship: the steep population is dominated by very steep Arunachal ridges,
so negatives came out **steeper** than positives (22.5° vs 13.3°), which would
have taught the model that steeper ground is safer. The floor was removed. These
states are mostly hill terrain, so the shortcut it guarded against does not
arise; the guard was worse than the disease.

## 6. Known limitations of v1

- Negatives are *unlabelled*, not confirmed stable. A steep site on a wet day may
  be an unreported landslide. This is a positive-unlabelled problem.
- Rainfall is IMD's land-only India grid. 110 of the 692 bbox events fell outside
  the mask (Bangladesh, Myanmar, Nepal, Bhutan) and were dropped with the
  non-India scope.
- 78 Darjeeling-area events were dropped: same hills, West Bengal, outside the
  eight-state brief. Worth reconsidering if the brief allows.
- `dist_road_m`, `dist_major_road_m`, `dist_stream_m` are pending — the Overpass
  extraction was still running when v1 was cut. They arrive in v1.1.
- Lithology, faults, land cover, NDVI and soil moisture are absent; each needs a
  login or a portal that cannot be scripted. See `PENDING` in `src/schema_v1.py`.
