# Data and characterization sources

Targets/caches belong at `<shared-data-root>/error-qcbm/`. The bundled noise
JSON is static model configuration, not a dataset or a complete Hub response.
See [README credits](README.md#reference-and-attribution) and [BibTeX references](README.md#citation-and-license) for credits
and notice boundaries. PDFs and the author source cache are not redistributed.

## Synthetic Gaussian target

There is no external Gaussian training dataset. `lib/data.py::gaussian_target`
evaluates two Gaussian densities at integer coordinates and normalizes their sum.
For the 495-bin experiment, the means are 141 and 353 and both widths are 61.
This follows the [pinned author code](https://github.com/Quandela/photonic-qcbm/tree/d593d209128e403644b9aee595c474ce551391ae).
It is a known target probability vector, not density estimation from a finite
collection of training samples. Generated cache files belong outside this paper.

Coordinates are integers `0..494`, not finite-width bin integrals. Coordinate
`j` maps to collision-free state `j` in ascending binary lexicographic order,
equivalent to the author's mixed-radix ordering. Physical evolution still
includes collisions before detection/conditioning.

From the flat workspace root:

```bash
python implementation.py --paper error-qcbm --config configs/prepare_gaussian.json
```

This generates `data/error-qcbm/gaussian/{target.npz,metadata.json}` and a
timestamped run-level `data_preparation.json`; it does not sample or train.
`--data-root` overrides the shared parent. Metadata includes the source commit,
parameters and a float64 probability-vector hash for the generating environment.

## Financial target

The reproduction uses daily ECB USD-per-EUR reference rates, series
`EXR.D.USD.EUR.SP00.A`, requested from 2004-01-01 through 2023-12-31.
[Official CSV query](https://data-api.ecb.europa.eu/service/data/EXR/D.USD.EUR.SP00.A?startPeriod=2004-01-01&endPeriod=2023-12-31&format=csvdata).

Place an unchanged CSV at
`<shared-data-root>/error-qcbm/fx/ecb_usd_per_eur_2004_2023.csv`.
The frozen configuration expects SHA256
`2d45d8e704309af8ec25f53bdaab38f17b7d731022662b5708cbeedebca89794`.
A different response fails explicitly; do not silently substitute a new dataset.
The frozen file is not redistributed. Explicit acquisition and offline
preprocessing both use the shared runtime:

```bash
# Explicit official download, frozen-hash verification, then preprocessing
python implementation.py --paper error-qcbm --config configs/prepare_fx_download.json

# Offline verification/preprocessing of an existing frozen CSV
python implementation.py --paper error-qcbm --config configs/prepare_fx.json
```

The downloader requires the official HTTPS query and a finite timeout. It retains
the candidate at `fx/downloads/<SHA256>.csv` and records URL, response metadata,
timestamp and hash before validation. Mismatches fail without replacing a
different source, editing the scientific config or starting training. A new
ECB response/version is not substituted automatically. The exact frozen response
may be necessary if the live API changes values, metadata or encoding.

Live API acquisition timed out during the 2026-10-27 review, including the
explicit preparation command. Fresh downloaded-byte equivalence is unverified.
Frozen-source preprocessing was checked offline; promotion/mismatch/cache
preservation and timeout behavior were tested with explicit fixtures. The
original CSV acquisition time was not preserved; file timestamps are not used
to invent it.

The saved source contained 5,124 valid rates (2004-01-02–2023-12-29) and 37 missing
rows. Successive available observations produce 5,123 natural-log returns.
There is no interpolation or forward filling. A 495-bin equal-width histogram
over the observed return range defines the empirical target.
The paper does not publish its exact provider, dates or bin edges; these are
explicit reproduction choices. No held-out forecasting/generalization claim is made.

The frozen target occupies 231 bins. FX NPZ contains rates/dates, returns/dates,
histogram counts/edges, probabilities and output-state mapping. Metadata credits
the European Central Bank and states that reproduction authors calculated the
log returns/histogram, which are not ECB-produced derived data. Both preparation
and training obtain `source_credit`, `source_terms_url` and
`transformation_notice` from `lib/data.py::financial_target`. Acquisition
details remain specific to the preparation path. The exported financial
PNG/SVG also carries the ECB source and reproduction-author processing notice.
ECB's [copyright conditions](https://www.ecb.europa.eu/services/using-our-site/disclaimer/html/index.en.html)
require source attribution and disclosure of modifications; consult them for
other conditions. Software licensing does not replace data terms. The target
shape need not match paper panel (d); synthetic returns or changed bins are not
used to imitate it.

## Belenos-characteristic simulation

The original internal snapshot was read on 2026-10-17T11:23:43.820379+00:00 and
reported total transmission 2.8% and HOM visibility 72.9%. The local model uses
T=.028 and I=.729 directly, without an extra square root at the API boundary.
The original snapshot SHA256 is
`e94ff99c712f0f844a55ea709c98b261bd033a283171c625584dceecbcbaa90e`.

The complete Hub response, credentials and server metadata are not distributed.
Public configs explicitly select
[`configs/noise/belenos_20261017.json`](configs/noise/belenos_20261017.json)
through `noise_profile`, resolved relative to this paper directory. It contains
only the selected percentages, source-field mapping, query time, original hash
and modeling assumptions. Its schema is `error_qcbm.noise_profile.v1`; each run
also records the public-profile file hash. No Hub account, key, original snapshot,
device query or QPU submission is required. Missing/ambiguous inputs, invalid
percentages and unknown schemas fail without fallback.

The calibration timestamp was not identified and is `null`: query time is not
a calibration date. `Transmittance (%) / 100` gives T≈.028, so photon loss is
1-T≈.972. `HOM (%) / 100` gives I=.729. Brightness is fixed at one because T
already includes total efficiency. I is passed directly at the API boundary;
the internal amplitude-overlap convention uses sqrt(I). Independent two-photon
HOM controls test this mapping. This chosen overlap model is not source tomography.

Local assumptions are brightness=1, g2=0 and threshold detection. Available
profiles include ideal (1,1), paper loss-only (.2,1), Belenos loss-only (.028,1)
and Belenos loss-plus-partial-overlap (.028,.729). Additional lossless/I=.84
controls are explicit choices and do not alter the profile's measured fields.
Unmodeled effects include multiphoton emission/nonzero g2, dark counts,
mode-dependent losses, phase errors, drift and correlated noise. Hardware clock
and per-channel calibration fields are excluded; clock does not predict simulator
runtime. This is not a digital twin or an Altair/Belenos QPU result.

```bash
python implementation.py --paper error-qcbm --config configs/belenos_profile_smoke.json
```

This small offline native-MerLin check uses four modes/two photons, three
profiles, 2k/4k recorded shots and 1k warm-up per profile. It tests counts and a
sampling/theory bound, not full-circuit training or recycling-sector adequacy.
`belenos_smoke.json` still means the expensive 50M/20-update training check.

I=.84 is an additional model input motivated by the paper's approximate
indistinguishability measurement .84±.03. The measurement convention is not
fully specified, so it is not asserted to be a corrected Belenos HOM value.
The commented .92 example in the author repository is not substituted for .84.

## Legacy inputs and test scope

Explicit legacy `noise_snapshot`, or `snapshot_file` in native/phase-one
settings, refers to `<shared-data-root>/error-qcbm/hub/`. Legacy physical mapping
and provenance remain unchanged for frozen checkpoint signatures. Public-profile
provenance is different even if physical values match. Do not relabel an existing
checkpoint signature. For legacy continuation keep its original config; when
overlaying a public config, explicitly set `noise_profile` to `null` and supply
`noise_snapshot`, preserving other scientific settings and sampler evidence.
See [TRAINING.md](lib/TRAINING.md).

Public tests use bundled profiles and synthetic legacy fixtures without original
Hub replies, ECB CSV, author caches or historical results. Offline qualification
refers to the reviewed installed environment, not a fresh dependency installation.
