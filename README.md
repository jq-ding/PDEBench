
# PDEGNN-BENCH

**PDEGNN-BENCH: Benchmarking Graph Neural Networks through Governing Laws of Learning Dynamics**

A unified benchmark for understanding and comparing graph neural networks through the lens of their underlying **governing equations** — the partial and ordinary differential equations that shape how each model propagates information across a graph.

## Overview

Most GNN benchmarks evaluate architectural variants on accuracy. `PDEGNN-BENCH` takes a different angle: it groups GNN models by the **governing equation** that defines their learning dynamics, and asks how each family of dynamics responds to the two central practical challenges in deep graph learning:

1. **Oversmoothing** — the degradation of node representations as depth increases
2. **Homophily sensitivity** — the dependence of model performance on graph structural alignment

By linking each GNN to its corresponding PDE/ODE, the benchmark provides mechanistic interpretations of empirical behavior and offers guidance for choosing or designing GNNs for new graph data.

## Six Families of Governing Equations

The benchmark covers six representative classes of dynamics, each represented by one or more state-of-the-art models:

| Family | Equation Type | Models Evaluated |
|---|---|---|
| **Isotropic Diffusion** | Parabolic heat flow with constant conductance | GCN, GCNII |
| **Anisotropic Diffusion** | Feature-adaptive diffusion | GRAND, GAT |
| **Non-Local Diffusion** | Fractional Laplacian | fLode |
| **Reaction–Diffusion** | Diffusion + nonlinear kinetics | GREAD, ACMP |
| **Hamiltonian Systems** | Energy-conserving dynamics | HamGNN |
| **Wave Transport** | Hyperbolic wave propagation | GraphCON |
| **Oscillatory Synchronization** | Phase-coupled oscillators | KuramotoGNN, BRICK |

## Diagnostic Metrics

Two complementary indicators are computed layer-wise to diagnose oversmoothing:

- **Effective rank (`r_eff`)** — quantifies the intrinsic dimensionality of node representations. A drop toward 1 signals representation collapse.
- **Class-mix score (`S`)** — measures how well class structure is preserved. Approaching 0 indicates that within-class and between-class distances become indistinguishable.

Together, these two metrics separate two distinct failure modes that accuracy alone cannot disentangle: rank collapse vs. class mixing.

## Key Findings

The benchmark uncovers several patterns across PDE families:

- **Anisotropic diffusion** (GRAND), **reaction–diffusion** (ACMP), and **phase-coupled oscillation** (BRICK) are the most depth-stable, maintaining high `r_eff` and `S` even at 128 layers
- **Isotropic diffusion** (GCN) collapses fastest, with sharp accuracy drops beyond 8–16 layers on homophilous graphs
- **Wave transport** (GraphCON) and **Hamiltonian systems** (HamGNN) show distinct failure modes on different graph structures
- Most PDE-GNNs benefit from homophily, but the degree of dependence varies systematically by governing equation: `diffusion ≫ reaction ≈ wave/Hamiltonian > non-local > oscillation`

## Model-Agnostic Oversmoothing Alarm

A practical contribution of the benchmark is a **dataset-specific empirical envelope** in the `(r_eff, S)` plane. By aggregating validation-optimal points across many models, the envelope captures the typical operating regime where representations remain stable. During training of a new model, drift outside the envelope provides an early-warning signal for oversmoothing — without requiring labels beyond the validation set.

## Repository Structure

```
PDEGNN-BENCH/
├── GE-Bench/              # Main benchmark implementation
│   ├── models/            # PDE-governed GNN implementations
│   ├── data/              # Dataset loaders
│   ├── metrics/           # r_eff, class-mix score, alarm system
│   ├── configs/           # Per-model hyperparameter configurations
│   └── scripts/           # Training and evaluation scripts
├── LICENSE
└── README.md
```

## Datasets

Seven node classification datasets spanning a wide homophily range `h ∈ [0.11, 0.81]`:

| Dataset | Homophily `h` | # Nodes | # Edges | # Classes |
|---|---|---|---|---|
| Texas | 0.11 | 183 | 295 | 5 |
| Wisconsin | 0.21 | 251 | 466 | 5 |
| Squirrel | 0.22 | 5,201 | 198,493 | 5 |
| Chameleon | 0.23 | 2,277 | 31,421 | 5 |
| Cornell | 0.30 | 183 | 280 | 5 |
| CiteSeer | 0.74 | 3,327 | 4,676 | 7 |
| Cora | 0.81 | 2,708 | 5,278 | 6 |

All datasets are evaluated under the 10-fold cross-validation splits from Geom-GCN.

## Setup

```bash
conda create -n pdegnn python=3.10
conda activate pdegnn
pip install torch torch-geometric numpy scipy scikit-learn matplotlib
```

## Usage

### Run a single model on a single dataset

```bash
python GE-Bench/scripts/run.py \
    --model GRAND \
    --dataset Cora \
    --depths 2,4,8,16,32,64,128 \
    --gpu 0
```


## Hyperparameter Protocol

To ensure fair comparison:
- Per-model hyperparameters from the original papers are kept unchanged
- Only the number of layers (or integration time, for ODE-based models) is varied
- Learning rate is tuned over `{1e-2, 5e-3, 1e-3}`
- Weight decay is fixed at `1e-4`
- Hidden dimension is fixed at 64
- Models without explicit layer definitions use the GRAND-style protocol: RK4 integrator with step size 1, integration time from 2 to 128

## License

MIT License — see `LICENSE` for details.

## Contact

Jiaqi Ding — `jiaqid@cs.unc.edu`
[Personal website](https://jq-ding.github.io/) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=5h5qru8AAAAJ)
