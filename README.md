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
  optimization/      Training, local search, GA, greedy swap and DE
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

The fixed files in `configs/` contain the adopted settings only. `base.yaml` holds shared settings; the other four files retain the differences required by each experiment.

The optional demand-bias term is retained with `optional_objective.alpha_demand: 0.0` (off). A nonzero coefficient enables the historical diagonal cost `alpha_demand / N * (1 - w_i)`, where junction base demands are clipped at the 95th percentile of positive values and normalized to `[0, 1]`. Set the coefficient in a copied configuration to run this optional variant. Enabling it changes the placement objective; the adopted profiles leave it off.

Use `--show-config` to inspect resolved settings and `--help` for arguments. Without overrides, algorithm comparison uses its ten configured seeds; budget sensitivity uses its network-specific six-seed schedule. Main comparison has the additional scalar physics gate off; component ablation retains its own variants and weights.

H/R/S assets, layouts and metric CSVs are generated under Git-ignored `outputs/`. No intermediate or historical result files are distributed. Existing outputs are not silently overwritten: use a fresh `--assets-dir` or `--output-dir`. Preparation requires a fresh feature directory when calibration already exists. Algorithm/objective comparison supports custom input directories; ablation and budget sensitivity use `data/networks/` and `data/features/`.

Fresh runs use the maintained simulation and evaluation implementation, including constant additional leakage demand, failed-simulation recovery, ID alignment and shared valid scenarios. Their outputs are new executions and do not replace the archived manuscript results.

## Citation

See `CITATION.cff`. Third-party input data are not bundled. No open-source license has been selected for this repository.
