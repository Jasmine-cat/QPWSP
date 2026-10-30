# QPWSP

Code for *Leak-Localization-Oriented Pressure Sensor Placement in Water Distribution Networks Using QUBO and Physics-Informed Graph Neural Optimization*.

The workflow prepares network features, generates hydraulic response assets, builds the budgeted H-response QUBO, optimizes sensor placement, and evaluates localization. Algorithm comparison, MVC objective comparison, component ablation and budget sensitivity use separate configurations.

## Structure

```text
configs/             Network parameters and experiment profiles
data/networks/       User-supplied INP files
data/features/       Prepared features and q0 calibration (generated locally)
run.py               Unified command-line entry point
src/
  data/              Preprocessing, loading, response statistics and ID alignment
  hydraulics/        Constant additional leakage demand and H/R/S generation
  qubo/              H-response, budget penalty and MVC comparison objective
  models/            Graph neural model
  optimization/      Training, local search, GA, greedy swap, DE and information selection
  evaluation/        Residual-dictionary matching and localization metrics
  experiments/       Configuration loading and experiment dispatch
```

## Environment

Use Python 3.10+ and install from the repository root:

```bash
pip install -e .
```

Alternatively, use `conda env create -f environment.yml`. Select a PyTorch build suitable for your CPU/CUDA environment.

Local checks used Python 3.12, NumPy 2.4.2, pandas 2.2.3, SciPy 1.15.2, NetworkX 3.2.1, PyYAML 6.0.2, PyTorch 2.5.1, WNTR 1.4.0, epynet 1.1 and pyarrow 19.0.1. These are local verification versions, not an archived environment lock.

## Inputs

Place each `<net>.inp` in `data/networks/`, with names matching the configuration, such as `Hanoi`, `Richmond_skeleton`, `CY_DMA` or `EX`. Obtain the study's specific network variants from their original sources; benchmark collections include [KIOS EPANET Benchmarks](https://github.com/KIOS-Research/EPANET-Benchmarks) and [Exeter Centre for Water Systems](https://www.exeter.ac.uk/research/centres/cws/resources/benchmarks/).

Preparation creates `<net>_node_features.csv`, `<net>_edge_features.csv`, `<net>_graph_meta.json` and `<net>q0.json` under `data/features/`. Existing matching features and calibration can be supplied instead. Keep them paired with the same INP version.

## Run

Run from the repository root:

```bash
python run.py prepare --nets Hanoi
python run.py assets --nets Hanoi
python run.py compare --nets Hanoi --seeds 0
```

Other recorded experiment profiles:

```bash
python run.py objective --nets Hanoi --seeds 0
python run.py ablation --nets Hanoi --seeds 0
python run.py budget --nets Hanoi --seeds 0
```

The fixed files in `configs/` contain the recorded profiles. `base.yaml` holds shared settings; the other four files retain experiment-specific settings. The optional information comparison uses 16 bins in `objective_comparison.yaml`.

The optional demand-bias term is retained with `optional_objective.alpha_demand: 0.0` (off). A nonzero coefficient enables the historical diagonal cost `alpha_demand / N * (1 - w_i)`, where junction base demands are clipped at the 95th percentile of positive values and normalized to `[0, 1]`. Set the coefficient in a copied configuration to run this optional variant. Enabling it changes the placement objective; the adopted profiles leave it off.

Use `--show-config` to inspect resolved settings and `--help` for arguments. Without overrides, algorithm comparison uses its ten configured seeds; budget sensitivity uses its network-specific six-seed schedule. Main comparison has the additional scalar physics gate off; component ablation retains its own variants and weights.

H/R/S assets, layouts and metric CSVs are generated under Git-ignored `outputs/`. No intermediate or historical result files are distributed. Existing outputs are not silently overwritten: use a fresh `--assets-dir` or `--output-dir`. Preparation requires a fresh feature directory when calibration already exists. Algorithm/objective comparison supports custom input directories; ablation and budget sensitivity use `data/networks/` and `data/features/`.

Fresh runs use the maintained simulation and evaluation implementation, including constant additional leakage demand, failed-simulation recovery, ID alignment and shared valid scenarios. Their outputs are new executions and do not replace the archived manuscript results.

## Information-theoretic objective comparison

The optional baseline implements the relevance / mean-redundancy ranking in Algorithm 1 of [Santos-Ruiz et al. (2022)](https://doi.org/10.3390/s22020443), without converting it to QUBO:

```bash
python run.py objective --nets Hanoi --objectives information
python run.py objective --nets Hanoi --objectives response mvc information
```

The default objective command still runs response and MVC only. The information route reads the same labeled `<net>_H.npz` as H-response and treats each available leak column as one sample, with the leak node as its class. All sensor rows are retained when failed H leak columns were explicitly excluded. Quantization uses 16 equal-width bins fitted on this design asset. R/S pressures are used only for the common evaluation, not selection or fitting bin boundaries.

This is a **nominal H-based adaptation** of the published ranking criterion. The source study built a pressure dataset with multiple leak magnitudes per location; this route instead compares criteria using the existing normalized nominal responses. With one sample per class, empirical relevance equals the entropy of the discretized sensor response. It does not estimate variation within a leak class or reproduce the paper's multi-magnitude experiment. No synthetic samples, noise or additional hydraulic simulations are introduced.

| Design resource | H-response | Information route | MVC |
| --- | --- | --- | --- |
| Existing H asset | Used | Same asset | Not required |
| Additional design simulations when H is supplied | 0 | 0 | 0 |
| Nominal leak cases available | Available H columns | Same H columns | Topology only |

Building a fresh H normally requires one no-leak baseline plus one leak simulation per candidate, with extra calls for failed-node recovery. This cost is incurred once per network and reused across optimization seeds. Calibration and R/S generation are common preparation/evaluation costs; they are not repeated for each placement or counted as private information-theory training data. MVC does not need hydraulic design data, so it is not assigned artificial simulation calls.

The response and MVC profiles use seeds **0, 11, 22, 33, 44, 55, 66, 77, 88, 99**. Given a fixed H, information selection is deterministic and runs once per network; `--seeds` affects only the stochastic optimizers. Its CSV has an empty seed and `repetition=fixed_design`. Do not count copies of this layout as independent seed repeats. Results include raw SR/MRR/TE/ETE, all1-relative metrics, the H fingerprint and zero additional design simulations, with layout and ranking files beside the CSV. All outputs remain local under the selected output directory.

For the optional multi-strength design comparison:

```bash
python run.py objective --nets Hanoi --objectives information information_sampled --simulation-workers 4
```

The `information_sampled` route samples approximately one third of the leak locations without replacement for each configured seed. It simulates three fixed leakage magnitudes at each sampled location and keeps every sensor candidate. The three magnitudes are the equal-width stratum midpoints of `[0.25, 2] × q0`: approximately `0.5417, 1.125, 1.7083 × q0`. Timing and the nominal flow scale come from the paired H metadata. Leaks use constant additional demand. Quantization is fitted separately on each seed's design pressures, using the same 16 bins. Selection never reads R/S pressures, and the evaluation retains the full leak dictionary.

Each layout can access approximately N design cases, specifically `3 × round(N/3)` with at least two leak classes. The seed controls sampled leak locations; it does not introduce randomness into ranking. Repeated node/magnitude cases are cached across seeds, so unique simulation calls can be lower than the sum of the per-layout design budgets. Both quantities are recorded in the design manifest. A group recovery, if needed, repeats the design at a lower common flow scale and is counted as additional simulation cost. Failed simulations remain missing; a leak class is used only when all three magnitudes are valid, and more than 30% incomplete sampled locations stops that comparison.

This arm changes both location coverage and intensity coverage relative to the full-H arm. Their difference therefore measures the combined design-data choice, not the isolated effect of adding multiple strengths. H-response can reuse its archived layouts; neither information route requires PIGNN training. The generated design pressures, seed layouts and metrics stay under the chosen output directory.

## Citation

See `CITATION.cff`. Third-party input data are not bundled. No open-source license has been selected for this repository.
