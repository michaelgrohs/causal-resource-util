# Causal Analysis of the Impact of Resource Utilisation on Decisions in a Process

This repository contains the code and results for the manuscript "TBD" submitted to Information Systems. 
The core question is whether the resource utilisation at a choice point in a process model has a causal effect on which path is taken.

---

## Overview

The pipeline takes an enriched multi-run as input, and then runs causal machine learning (Double ML + Causal Forest) to test whether utilisation is a significant driver of routing choices.

Five process discovery algorithms are compared as miners:

| Miner                        | ID          |
|------------------------------|-------------|
| Directly-Follows Graph       | `dfg-miner` |
| Flower model                 | `flw-miner` |
| Inductive Miner              | `im-miner`  | 
| Inductive Miner infrequent   | `imf-miner` | 
| Split Miner                  | `spl-miner` | 

---

## Repository structure

```
.
├── causal_analysis.py          # Core causal inference functions (Double ML, Causal Forest, backdoor check)
├── reporting.py                # Generates Excel + PDF reports from pipeline results
├── pipeline_per_miner.ipynb    # Main pipeline: iterates over miners and datasets
├── visualizations.ipynb        # Resource utilisation distribution plots (per dataset + overall)
├── correlation_analysis.ipynb  # Correlation matrix: dataset size metrics vs. computation times
│
├── confounders.xlsx            # Number of confounders (base / resource / alt-util) per dataset × miner
├── correlations.csv            # Saved pairwise Pearson correlation matrix
├── generated.exs               # Minimal synthetic .exs file for development / unit testing
│
├── input/                      # ⚠ NOT in git — see "Input data" below
│   ├── 1-discoveredmodels/     # Discovered process models (.dfg, .ptree) + SVG visualisations
│   │   └── *.dfg / *.ptree
│   └── 3-executions-per-miner/ # enriched multi-runs as .exs files
│       ├── dfg-miner/          #   .exs execution replay files + .exs.time (computation time)
│       ├── flw-miner/
│       ├── im-miner/
│       ├── imf-miner/
│       └── spl-miner/
│
└── output/
    ├── feasibility_study.csv   # Per-dataset timing + descriptive statistics summary
    ├── compute_times.xlsx      # All .exs.time values across all input folders
    ├── correlation_matrix.pdf  # Heatmap: size metrics vs. computation times
    ├── util_dist_overall.pdf   # Resource utilisation distribution — feasible logs, all miners
    ├── util_dist_bpic17o_dfg.pdf  # Distribution for BPI 2017 Offer log, DFG miner
    ├── dfg-miner/              # Per-dataset causal analysis reports (Excel + PDF)
    ├── flw-miner/
    ├── im-miner/
    ├── imf-miner/
    └── spl-miner/
```


---

## Input data

> **⚠ The enriched multi-run and discovered model files (`input/`) are too large for this repository.**
>
> Download them from: **[PLACEHOLDER — external link to be added]**
>
> Extract the archive so that the folder structure matches:
> ```
> input/3-executions-per-miner/dfg-miner/*.exs
> input/3-executions-per-miner/flw-miner/*.exs
> input/3-executions-per-miner/im-miner/*.exs
> ```
> Each `.exs` file has a companion `.exs.time` file recording how long the replay took.



### Datasets

The following public event logs from the [4TU Research Data repository](https://data.4tu.nl/) were used:

| Log | Short name used in filenames |
|---|---|
| BPI Challenge 2017 — Offer log | `BPI Challenge 2017 - Offer log` |
| BPI Challenge 2013 — Closed problems | `BPI_Challenge_2013_closed_problems` |
| BPI Challenge 2013 — Incidents | `BPI_Challenge_2013_incidents` |
| BPI Challenge 2013 — Open problems | `BPI_Challenge_2013_open_problems` |
| BPIC 2012-a | `bpic12-a` |
| BPIC 2018 Parcel document | `bpic18 Parcel document` |
| CoSeLoG WABO 1–5 | `CoSeLoG WABO {1..5}` |
| Hospital log | `Hospital_log` |
| Road Traffic Fine Management | `Road_Traffic_Fine_Management_Process` |
| Sepsis | `Sepsis` |
| BPIC 2020 (Domestic / International / Prepaid / RequestForPayment) | `bpic2020-*` |

---

## Setup

**Requires Python 3.10.**

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install pm4py==2.7.22.2 pandas numpy scipy scikit-learn econml causalml \
            matplotlib seaborn openpyxl jupyterlab ebi-pm
```

Key package versions used during development:

| Package | Version |
|---|---|
| pm4py | 2.7.22.2 |
| pandas | 2.3.3 |
| numpy | 2.2.6 |
| econml | 0.16.0 |
| causalml | 0.15.5 |
| ebi-pm | 0.3.11 |
| scipy | 1.15.3 |
| seaborn | 0.13.2 |

---

## Running the pipeline

1. Ensure the input data is in place (see "Input data" above).
2. Launch JupyterLab:
   ```bash
   source .venv/bin/activate
   jupyter lab
   ```
3. Open and run **`pipeline_per_miner.ipynb`** — this iterates over all three miners and all datasets, writes per-dataset Excel/PDF reports to `output/{miner}/`, and appends a row to `output/feasibility_study.csv`.

### Analysis notebooks

| Notebook | Purpose |
|---|---|
| `visualizations.ipynb` | Resource utilisation histograms; saves PDFs for the overall distribution and BPI 2017 dfg |
| `correlation_analysis.ipynb` | Pearson correlation matrix of dataset size metrics vs. computation times; saves `output/correlation_matrix.pdf` |

### Key output files

| File | Description |
|---|---|
| `output/feasibility_study.csv` | One row per dataset × miner: counts, timing per pipeline step |
| `output/compute_times.xlsx` | All `.exs.time` values from all input folders |
| `output/confounders.xlsx` (= root `confounders.xlsx`) | Number of confounders per dataset × miner |
| `output/{miner}/{dataset}/` | Full causal analysis report (Excel + PDF) per dataset |

---

## .exs file format

`.exs` files are JSON produced by replaying a Petri net against an event log. Each entry in `"executions"` records one transition firing: which transition fired, what other transitions were simultaneously enabled, and the resource utilisation values at the moment of the choice.

Two formats exist (auto-detected by the parser in `pipeline_per_miner.ipynb`):

- **Old format** — single `resource_utilisation` scalar per firing
- **New format (alt-util)** — per-transition utilisation: `resource_utilisation_fired_transition` + `other_enabled_transitions_resource_utilisation`

The new format enables the full confounder set used in causal analysis.
