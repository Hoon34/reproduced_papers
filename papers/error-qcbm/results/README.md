# Figure 2 counterparts and curated numeric results

These three PNG/SVG pairs are generated reproduction figures, not
copies of the article's artwork or author numerical data. The financial figure
includes its ECB source and processing notice. `figure_sha256` in the result
manifest identifies the committed images. The public bundle redraws all PNGs with identical pixels
in the tested environment; SVG geometry/styles/text also match after removing
timestamps and generated identifiers. Rendering in a different environment may
change fonts or metadata without changing the selected numeric trajectories.

- **gaussian_fig2ab:** Gaussian-mixture counterparts of Fig.2(a,b), based on
  20 seeds and lossless/post-selection/recycling controls, 200k initial shots,
  1,000 SPSA updates. The target is the known 495-bin probability vector.
  Panel (a) shows mean +/- population SD (`ddof=0`) across all 20 original
  runs per condition. Panel (b) shows five learning curves each for lossless
  and recycling, chosen by lowest final finite-shot KL separately per condition
  with seed as the tie-break. Lossless seeds: **16,18,2,7,1**; recycling:
  **6,15,1,4,14**. This is not a histogram or a matched-best-seed comparison.
  The author's exact best-five selection rule was not supplied; this is an
  explicit reproduction choice. Post-selection is shown in (a).
- **financial_fig2cd:** ECB 2004–2023 adaptation, TVD, 60k initial shots.
  Source: European Central Bank (ECB), series `EXR.D.USD.EUR.SP00.A`.
  Log returns and histograms were calculated by the reproduction authors;
  they are not ECB-produced derived data. This notice also appears inside
  the exported PNG/SVG.
  Panel (c) shows mean +/- population SD over 20 originals. Panel (d) uses
  finite-shot samples of one trained lossless model (fixed seed 0, independent
  of performance),
  with 5,123 model samples and 5,123 target returns. Its coarse histogram bins are
  a visualization choice: sum five adjacent training bins, 495 to 99. The model
  histogram is sampled data, not exact probabilities. It selects 5,123 items
  without replacement from **39,219 stored native four-click samples**, using
  subsample seed **20261018**. This is data subsampling, not a replacement
  photonic sampler or new photon shots. Original financial observations and
  display boundaries were unavailable, so histogram shape can differ.
- **belenos_fig2ef_original:** Belenos-characteristic MerLin simulation, 50M
  initial shots per evaluation. Panel (e) keeps three individual training
  instances per condition, seeds 0/1/2, without a mean, shaded band or interruption
  marker. Panel (f) keeps seed 6 across all six curves: noisy recycling I=.729,
  lossless I=.729, lossless ideal I=1, loss-only recycling I=1, noisy recycling
  I=.84 and lossless I=.84. Seed 6 was selected by the original noisy recycling
  final finite-shot KL among 19 complete original runs; it is not reselected
  for each added condition. There is no actual QPU Experiment curve.

Panel (e)'s three seeds have different initializations; the paper's three
hardware repeats used one common initialization. This is a simulation extension
with 20 matched initializations, not a hardware reproduction. In (f), T=.028
applies to lossy curves; the labeled lossless controls use T=1. I=.84 is a
paper-motivated model input, not a newly measured Belenos value.

| Panel (f), original selected seed 6 | Final finite-shot KL |
|---|---:|
| T=.028, I=.729, recycling | 0.401152 |
| T=1, I=.729 | 0.103642 |
| T=1, I=1 | 0.222962 |
| T=.028, I=1, recycling | 0.362592 |
| T=.028, I=.84, recycling | 0.386301 |
| T=1, I=.84 | 0.146314 |

All displayed learning curves use finite-shot records after updates
10,20,...,1000. `Iteration Number = update/10`; the x limits are 0--100 while
the first plotted point is 1. The extra initial report is retained at update 0
in the numeric bundle, outside the plotted 100 points. Shaded bands in (a,c)
are SD, not SEM/confidence intervals, and their lower boundary is clipped at zero.

The Belenos family has 143 attempts including three original failures and three
explicitly separate sampling-stream retries. Do not relabel them as failure-free
original campaigns. All displayed training curves use recorded finite-shot
metrics.

| Original experiment | Complete / planned | Separate sampling retry |
|---|---:|---:|
| Gaussian, 3 conditions | 60/60 | 0 |
| Financial, 3 conditions | 60/60 | 0 |
| Main Belenos, 4 conditions | 79/80 | 1 complete |
| Belenos loss-only recycling I=1 | 19/20 | 1 complete |
| Belenos lossy recycling I=.84 | 19/20 | 1 complete |
| Lossless I=.84 | 20/20 | 0 |

There are **263 attempts**: 260 planned originals (257 complete, 3 failed) and
three complete retries. The failed originals reached committed update 683;
their last recorded losses are at update 680, not final update 1000. Each retry
inherits 69 history points and its gain, changes only the sampling stream by
1,000,000,000, and is linked to its failed parent. Retry-inclusive 20-seed
statistics are supplementary, not failure-free original results or 21 independent
initializations. For noisy recycling I=.729/I=1/I=.84, original-19 finite KL means
are .640690/.671449/.642129; supplemented-20 means are .648195/.674318/.650081.

## Interpretation of the comparison

| Paired complete originals | n | Finite-shot recycling wins |
|---|---:|---:|
| Gaussian | 20 | 20/20 |
| Financial ECB adaptation | 20 | 20/20 |
| Belenos I=.729 | 19 | 19/19 |

Recycling is a biased estimator. These finite-shot results support a conditional
training-objective comparison, not universal mitigation/overlap improvement or
advantage over a classical generative model. For I=.84 versus I=.729, original
lossy runs win 9/19; lossless I=.84 wins 2/20. The selected seed-6 curves do not
establish population-level monotonicity with indistinguishability.

## Selected files and regeneration

| File | Public purpose |
|---|---|
| Three PNG/SVG pairs | Retained final Figure 2 counterparts |
| `runs.csv` | 263 attempt identities, physical settings, status, endpoints, gain, parent linkage and source hashes; no phases/credentials/host paths |
| `curves.csv` | 26,467 recorded points, including finite-shot losses and initial/partial/retry history |
| `financial_histogram.csv` | 495-bin target counts, available native counts, selected displayed counts and edges; no raw rates/dates/returns |
| `figure2_manifest.json` | File hashes, sample/seed policies, paper deviations and figure provenance |
| `summary.json` | Recomputed original/supplementary statistics, best-five selections and paired comparisons |

The five compact numeric files total about **2.41 MB**. They are explicit
exceptions to the default figures-only result policy: all are needed for
standalone redraw or audit. Raw checkpoints/count ledgers, source archives,
server tools, intermediate/with-retry figures and exploratory mitigation plots
remain internal. Source hashes bind these exports to the frozen inputs; those
inputs are not automatically downloaded or included.

```bash
# From the flat workspace root; no training, network or new photonic sampling
python implementation.py --paper error-qcbm --config configs/plot_results.json
```

The command validates the CSV hashes, inventory, history grid, retry linkage,
sample sizes and selection, then writes three PNG/SVG pairs, `summary.json` and
`metrics.json` into a new timestamped run directory. It does not overwrite these
retained figures. The old financial histories use explicit TVD field names;
export maps those names to the common schema and verifies all final values
against saved probabilities without rewriting originals.

Committed call counts include inherited retry prefixes and exclude replay
overhead. Do not sum them as physical shot costs. See [TRAINING.md](../lib/TRAINING.md)
for 2,103-call normal runs, 666-call new retry suffixes and interrupted/replayed
costs. Data/characterization sources and unresolved publication limits remain in
[DATA_SOURCES.md](../DATA_SOURCES.md) and [README credits](../README.md#reference-and-attribution).
