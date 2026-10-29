# Finite-shot estimation, SPSA and continuation

The numerical reference is [Quandela/photonic-qcbm at
`d593d209128e403644b9aee595c474ce551391ae`](https://github.com/Quandela/photonic-qcbm/tree/d593d209128e403644b9aee595c474ce551391ae).
This implementation uses native MerLin sampling and classical SPSA. The
count-accumulation adapter and its environment gate are described in
[SAMPLING.md](SAMPLING.md). No gradient through sampled counts or recycling is
used for training.

## Finite-shot autodiff limitation and the implemented training path

**In the inspected MerLin 0.4.2, ordinary autodiff through actual finite-shot
sampling is unsupported in the training path used here.** When a layer is in
training mode, autograd is enabled and at least one parameter requires a
gradient, requesting `shots > 0` emits a warning, disables sampling and uses
effective shots zero. Backward then differentiates **exact probabilities**.
That is exact autodiff, even though the call included a finite `shots` argument.
Under `torch.no_grad()`, the shot request invokes actual native sampling.
This statement is scoped to the inspected release/path; it does not rule out
finite-shot gradient estimation through a separately implemented estimator.

Our experiments use **finite-shot SPSA**, following the pinned author's optimizer.
We do not train by exact autodiff or backpropagate through the sampled counts
and recycling estimator. The implemented update has this data flow:

```text
theta + ck*delta -> MerLin noisy forward under no_grad -> N initial shots
                -> aggregate integer counts across chunks
                -> post-selection or recycling -> L_plus
theta - ck*delta -> same pipeline, next sampling seed -> L_minus
g_hat[j] = (L_plus - L_minus) / (2*ck*delta[j])
theta <- theta - ak*g_hat
```

The sign-vector perturbation and update are implemented in `lib/spsa.py`
using NumPy. `lib/backend.py` calls `QuantumLayer(shots=N)` inside `no_grad`;
`lib/training.py::FiniteShotObjective` computes the probability estimate and
loss from its native sampled counts. `lib/recycling.py` converts counts to
Perceval `BSCount`, calls the official `perceval.photon_recycling()` function,
and aligns its output with the fixed target bins. MerLin supplies the optical/noise/detector
calculation and native sampling, rather than the classical optimizer or a
gradient through mitigation. Both perturbations consume N initial shots each.
Each objective call advances the configured sampling seed; the two losses
are not evaluations of one reused count ledger.

The optimized count adapter preserves native multinomial draws and changes
only aggregation. It does not introduce a gradient estimator. All sampled
conditions, including the lossless controls, follow the finite-shot objective.
Calibration and reporting also draw real shots, as counted below. Full runs
record a fresh finite-shot objective every ten updates.

The environment check and independent feedback reproducer verified the
sampling-disabled warning/equality with exact output under autograd and an
actual 37-shot native call under `no_grad`. Treat a future MerLin finite-shot
gradient API as a separately validated experiment, with its estimator and shot
budget stated explicitly.

### What is shared with the author implementation

The pinned author's [`src/models/qcbm.py`](https://github.com/Quandela/photonic-qcbm/blob/d593d209128e403644b9aee595c474ce551391ae/src/models/qcbm.py)
uses a Perceval SLOS processor and `Sampler.sample_count` to obtain finite-shot
counts. `pseudo_grad` evaluates the positive/negative phase perturbations,
applies recycling or post-selection, and differences the resulting losses.
`fit` passes that function to the NumPy
[`src/helpers/spsa.py`](https://github.com/Quandela/photonic-qcbm/blob/d593d209128e403644b9aee595c474ce551391ae/src/helpers/spsa.py)
optimizer. This is already a finite-shot SPSA protocol, without a backward
pass through sampled counts or recycling. Its `get_loss` reports a newly
sampled objective after each configured block of SPSA updates.

We retain that optimization protocol and replace the optical/sampling backend
with MerLin. Choosing SPSA therefore follows the author implementation as well
as avoiding any need for unsupported finite-shot autodiff in this release.
The author did not use MerLin's autograd path in this code. The sampler engine,
RNG implementation and documented reproduction choices differ; identical SPSA
rules do not imply identical sampled trajectories across backends. The extra
initial reporting call is accounted for below.

## Estimator and loss

Optical evolution includes collisions and loss before threshold detection.
Recycling receives the aggregated **integer threshold counts from all chunks**.
For four input photons it uses the four-, three- and two-click sectors. A
three-click event can result from a collision even when no photon was lost.
Zero- and one-click events remain in shot accounting but are unused by the
estimator. Target bins are the fixed 495 four-click patterns, not a restriction
on physical evolution.

Each lower-click pattern contributes to every compatible four-click completion.
The three expanded distributions are normalized separately. For missing-click
count `r`, expanded mass is the sector count multiplied by
`comb(modes - photons + r, r)`. Any empty required sector raises an error after
that call's sampling budget; no extra shots or replacement estimator are used.
Post-selection independently renormalizes the four-click counts on the same
fixed target support.

Recycling uses [Perceval's official `photon_recycling()` API](https://perceval.quandela.net/docs/v1.3/reference/runtime/error_mitigation/photon_recycling.html).
Perceval performs the completion expansion, sector normalization, global decay
fit, per-bin extrapolation, clipping and final normalization. The project only
validates the threshold count ledger, converts it to `BSCount`, and aligns the
returned `BSDistribution` with the fixed lexicographic target order.
No additional fit, smoothing or fallback estimator is applied.

The standalone function is also used for the two-photon smoke; Perceval's
automatic `Computer` mitigation layer requires at least three input photons.
SDK errors propagate and invalid probability outputs fail clearly. Fit warnings
and the Perceval version are recorded. The public SDK does not return the fitted
decay rate, pre-clipping probabilities or expanded sector distributions, so
those internal arrays are not exported by the adapter.

Resumable recycling runs include the SDK estimator identity and Perceval version
in their scientific signature. An incompatible checkpoint is rejected before
sampling; use a separate run directory for a different estimator/version.

Gaussian training uses `KL(model || target)` with base-two logarithms. Zero
model mass contributes zero. Positive model mass at zero target probability
produces an infinite KL and fails the objective without pseudocounts. The
authors' helper instead skips such bins; that edge difference does not affect
the strictly positive Gaussian target. The financial experiment uses
`TVD = 0.5 * sum(abs(model - target))`, including empty empirical target bins.
Training and reporting use the finite-shot loss.

## SPSA and matched controls

For update `k=1,...,1000`, with `A=100`:

```text
alpha = .602, gamma = .101, c = .1
g0 = [L(theta0 + c*delta0) - L(theta0 - c*delta0)] / (2*c*delta0)
a  = .001 * (A+1)^alpha / abs(mean(g0))
ak = a / (k+A)^alpha
ck = c / k^gamma
gk = [L(theta + ck*deltak) - L(theta - ck*deltak)] / (2*ck*deltak)
theta <- theta - ak*gk
```

Perturbations have independent `+1/-1` entries. The denominator for calibration
is `abs(mean(g0))`, **not** `mean(abs(g0))`. A nonfinite magnitude or magnitude
at most `1e-12` fails explicitly; there is no epsilon substitution or automatic
recalibration. The gain is calibrated once per condition/initialization and
restored unchanged on continuation. All conditions use the same calibration
rule; sampled losses can produce different numerical `a` values. This is a
fixed algorithm comparison with condition-specific calibration, not a comparison
at one shared numerical gain. Alpha itself remains `.602`.

Initial phases follow `RandomState(seed).normal(0, 2*pi, parameter_count)`.
Across conditions, initialization seed `s`, perturbation seed `100000+s` and
initial-shot budget match. Sampling call `j` uses Torch seed
`10000 + 1000000*s + j`. Equal sampling seeds preserve a matched stream policy;
different physical probabilities and parameter trajectories need not give the
same outcomes. Native chunk size is part of the continuation signature because
changing chunking can change draw order. See the base JSON files for explicit
settings; reduced smoke runs use their own update-defined `A`.

## Reporting and shot cost

There are 1,000 parameter updates and a separate finite-shot evaluation after
each tenth update. Our histories additionally contain an evaluation at the
initial parameters. They have 101 points at updates `0,10,...,1000`; the plotted
`Iteration Number` is `update/10`. The pinned author's `fit()` returns 100
post-update evaluations and has no initial reporting call. The additional
initial call is a reproduction choice applied to every condition.

| Call purpose | Count per uninterrupted full run |
|---|---:|
| Initial reporting | 1 |
| Gain calibration | 2 |
| Positive/negative SPSA objectives | 2,000 |
| Reporting after each tenth update | 100 |
| Total | 2,103 |

| Experiment | Initial shots per call | Initial shots per uninterrupted run |
|---|---:|---:|
| Gaussian | 200,000 | 420,600,000 |
| Financial | 60,000 | 126,180,000 |
| Belenos characteristics and controls | 50,000,000 | 105,150,000,000 |

The analogous author loop costs 2,102 calls without our initial reporting call.
Figures that draw additional samples have a separate inference budget.

`lib/artifacts.py` counts planned objectives and append-only sampling
attempts. Resumable status distinguishes committed calls/shots from physical
ledger costs. `total_initial_shots` retains the legacy committed-path meaning;
it includes any inherited retry prefix and excludes extra replay work. Use
`shot_accounting` and the separate original/retry ledgers for physical cost.
Completed calls repeated after interruption are counted again. A killed call
without a terminal record has an unknown actual cost, bounded by its requested
shots; a torn final JSON record prevents automatic continuation until inspected.
Legacy failed calls without a completion marker also remain uncertain unless
other diagnostics establish their full budget.

## Checkpoint contract

Resumable configurations use `lib/resumable.py` through the shared runner.
The ordinary Gaussian/financial runner saves progress artifacts but does not
support continuation from those artifacts. Only the resumable runner implements
the contract below.

- `checkpoint.npz` is authoritative at a committed update boundary. It is
  flushed and atomically replaced after every update, independently of the
  ten-update reporting interval. Initial state is committed after calibration.
- It contains phases, calibrated gain/magnitude, stability constant, update
  number, complete perturbation RNG state, objective-call count, next sampling
  seed, initial/last counts and probabilities, history and configuration signature.
- Resume restores these values without initial evaluation or gain recalibration.
  An uncommitted update replays from the previous boundary with the same
  perturbations and sampling seeds. Its already attempted shots remain in the
  append-only ledger.
- A checkpoint is committed before the human-readable progress files. A process
  killed between these writes resumes from the checkpoint even if progress is
  stale. Completion reuse requires a compatible committed checkpoint and matching
  final phases; it consumes no new shots.
- Scientific changes (target, circuit, noise, seed offsets, optimizer settings,
  total updates, shot count, chunk size, dtype/device) are rejected. Segment
  duration and output location are operational settings. A stock-to-vectorized
  transition requires the independent full-budget evidence gate.
- Numerical failures remain failed and are not automatically retried. I/O errors
  propagate and are marked interrupted when progress can be written; append-only
  execution events preserve them across a later explicit resume. No in-process
  numerical retry loop is introduced. File replacement alone retries recognized
  Windows access/sharing errors (5/32/33) at most eight times, with 2.25 seconds
  total delay and warning logs; persistent errors propagate with the temporary
  file retained. This never resamples or repeats an optimizer update.
  Invalid persisted metadata is rejected before
  sampling or overwriting status. Only one writer may own an instance directory.

The deterministic comparisons cover this environment, finite-shot post-selection
and recycling, both accumulation implementations, unrecorded boundaries,
calibration/gradient/report interruptions, and checkpoint/progress write ordering.
They do not guarantee bitwise continuation across Python/NumPy/Torch/platform
changes or recovery from disk corruption/power loss. Archived Linux initial
normal draws showed sub-`1e-14` differences when regenerated on Windows; the
stored paired arrays agree exactly. The strict initialization check is retained
for public continuation. Preserve archived artifacts when moving environments;
do not silently regenerate or replace their state.

## Explicit sampling-stream retries

Historical seed-eight failures in three noisy recycling campaigns are retained
as original failed attempts. Separate authorized retries branch from committed
update 683, inheriting 69 history points and the same phases, gain and optimizer
RNG. Only the sampling offset changes by `1,000,000,000`: the next seed changes
from `8011437` to `1008011437`. The parent hash and preparation checks are recorded
separately. This is a changed sampling stream, not exact continuation of the
failed call and not an independent new initialization. Report originals and
retry-inclusive summaries separately, and count each branch's new ledger only
once rather than charging its inherited prefix twice.
