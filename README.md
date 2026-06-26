# QUBO-PIGNN Sensor Placement for Water Distribution Networks

This repository provides a minimal public-release structure for the study:

**Leak-Localization-Oriented Sensor Placement in Water Distribution Networks via QUBO and Physics-Informed Graph Neural Optimization**

The repository is intended to contain the cleaned core code required for graph construction, response-informed QUBO formulation, physics-informed graph neural optimization, baseline optimization, and deployment evaluation.

## Repository structure

```text
configs/     Experiment configuration files.
data/        Data notes and optional redistributable WDN input files.
src/         Reusable source-code modules.
scripts/     Reproduction entry-point scripts.
```

## Planned workflow

```bash
conda env create -f environment.yml
conda activate qubo-pignn

python scripts/run_objective_comparison.py --config configs/objective_comparison.yaml
python scripts/run_optimizer_comparison.py --config configs/optimizer_comparison.yaml
python scripts/run_ablation.py --config configs/ablation.yaml
python scripts/run_sensitivity.py --config configs/sensitivity.yaml
python scripts/make_figures.py
```

## Data

This repository expects EPANET-compatible WDN models in `.inp` format. Network files are included only when their redistribution license permits public sharing. Otherwise, source information and preparation instructions should be provided in `data/README.md`.

## Repository status

This repository is currently a clean public-release scaffold. 

## Citation

A formal citation entry will be added after manuscript acceptance or preprint release.

## License

No open-source license has been selected yet. Until a license is added, all rights are reserved by default.
