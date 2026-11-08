# QPWSP

Code for *Leak-Localization-Oriented Pressure Sensor Placement in Water Distribution Networks Using QUBO and Hydraulically Informed Graph Neural Optimization*. The proposed hydraulically informed graph neural optimizer is referred to as HIGNO.

The workflow prepares network features, generates hydraulic response assets, builds the budgeted H-response QUBO, optimizes sensor placement, and evaluates localization. Algorithm comparison, placement-criterion comparison, component ablation and budget sensitivity use separate configurations.

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

HIGNO training selects CUDA when available and otherwise uses CPU. To force CPU execution, uncomment the CPU device line in `src/optimization/trainer.py`. This device override alone does not set the number of CPU threads; configure thread limits separately for single-thread timing comparisons.

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

The fixed files in `configs/` contain the recorded profiles. `base.yaml` holds shared settings; the other four files retain experiment-specific settings. The information baseline uses 16 bins in `objective_comparison.yaml`.

The budget penalty is normalized by K squared: `alpha_bud * (sum(x) - K)^2 / K^2`.

The optional demand-bias term is retained with `optional_objective.alpha_demand: 0.0` (off). A nonzero coefficient enables the historical diagonal cost `alpha_demand / N * (1 - w_i)`, where junction base demands are clipped at the 95th percentile of positive values and normalized to `[0, 1]`. Set the coefficient in a copied configuration to run this optional variant. Enabling it changes the placement objective; the adopted profiles leave it off.

Use `--show-config` to inspect resolved settings and `--help` for arguments. Without overrides, algorithm comparison uses its ten configured seeds; budget sensitivity uses its network-specific six-seed schedule. Main comparison has the additional scalar physics gate off; component ablation retains its own variants and weights.

H/R/S assets, layouts and metric CSVs are generated under Git-ignored `outputs/`. No intermediate or historical result files are distributed. Existing outputs are not silently overwritten: use a fresh `--assets-dir` or `--output-dir`. Preparation requires a fresh feature directory when calibration already exists. Algorithm/objective comparison supports custom input directories; ablation and budget sensitivity use `data/networks/` and `data/features/`.

Fresh runs use the maintained simulation and evaluation implementation, including constant additional leakage demand, failed-simulation recovery, ID alignment and shared valid scenarios. Their outputs are new executions and do not replace the archived manuscript results.

## Information-theoretic objective comparison

The objective profile compares MVC, H-response and an information-theoretic (IT) placement baseline. IT follows the relevance / mean-redundancy forward ranking in Algorithm 1 of [Santos-Ruiz et al. (2022)](https://doi.org/10.3390/s22020443). It selects K sensor nodes directly, without a QUBO reformulation or HIGNO training.

```bash
python run.py objective --nets Hanoi
python run.py objective --nets Hanoi --objectives information --simulation-workers 4
```

The adopted `information` route samples approximately one third of the eligible leak locations without replacement for each seed, then simulates three leakage magnitudes at each sampled location. The three fixed magnitudes are the equal-width stratum midpoints of `[0.25, 2] × q0`, approximately `0.5417, 1.125, 1.7083 × q0`. Timing and the network flow scale q0 come from the paired H metadata. Leaks use constant additional demand. All pressure-sensor candidates remain available, including nodes not sampled as design leaks.

Each layout can read approximately N design cases, specifically `3 × round(N/3)` with at least two leak classes. This keeps design-simulation counts near the nominal one-leak-case-per-node H construction budget. Repeated node/magnitude cases are cached across seeds; each layout reads only its assigned subset. The manifest distinguishes per-layout information budgets from unique simulation calls across seeds and counts recovery calls separately. Failed scenarios remain missing. A leak class is used only when all three magnitudes are valid; more than 30% incomplete sampled locations stops that comparison after the documented common-flow recovery.

Pressures are quantized into 16 equal-width bins (4bit), fitted separately on each seed's design data. Leak-location labels define relevance; mutual information with previously selected pressures defines redundancy. R/S pressures are used only for the shared localization evaluation, which retains all eligible test leak locations. They never fit bin boundaries or select design locations. This is a simulation-budget-controlled adaptation of the source method's multi-magnitude pressure design; sampling locations and using 16 bins are explicit experiment settings.

The profile uses seeds **0, 11, 22, 33, 44, 55, 66, 77, 88, 99**. For IT, the seed controls sampled leak locations. Forward ranking is deterministic for a fixed design dataset. Outputs include raw SR/MRR/TE/ETE, all1-relative metrics, per-seed layouts and ranking traces, and the cached design pressures under the selected output directory. MVC and H-response use their recorded HIGNO settings and the same candidate sets, K and evaluation assets.

Building a fresh H normally requires one no-leak baseline plus one leak simulation per candidate, with extra calls for failed-node recovery. H is prepared once and shared across optimizer seeds. MVC uses topology and needs no hydraulic design simulations. Calibration and R/S generation are common preparation/evaluation costs, separate from each method's design-data budget. No artificial calls are assigned to MVC to equalize costs.

## Citation

See `CITATION.cff`. Third-party input data are not bundled. No open-source license has been selected for this repository.
