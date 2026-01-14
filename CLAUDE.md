# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains ¹³C metabolic flux analysis (¹³C-MFA) simulations using the Elementary Metabolite Unit (EMU) algorithm. It models isotope labeling patterns in metabolic networks to estimate intracellular flux distributions.

## Commands

### Install dependencies
```bash
uv sync
```

### Run Gibbs energy analysis (most commonly used)
```bash
uv run python scripts/gibbs_landscape_2d.py          # 2D FBA1/FBA2 landscape
uv run python scripts/gibbs_comparison_1d.py         # Mechanistic vs non-mechanistic
uv run python scripts/gibbs_comparison_fba1_fixed.py # FBA1 fixed, FBA2 varies
uv run python scripts/gibbs_comparison_fba2_fixed.py # FBA2 fixed, FBA1 varies
```

### Run enzyme simulation
```bash
uv run python scripts/enzyme_simulation.py
```

### Run main script
```bash
uv run python main.py
```

## Architecture

### Dependencies
- **freeflux**: Core library for ¹³C-MFA flux fitting using the EMU algorithm
- **diffrax**: JAX-based ODE solver for enzyme kinetics simulation
- **blackjax**: MCMC sampling (NUTS) for parameter estimation
- **jax**: JAX for numerical computation (pinned to <=0.4.18)
- Python 3.10+

### Model Definition

Models are defined via:
1. **TSV reaction files** (like `reactions.tsv`) - uses freeflux format with atom mappings:
   ```
   #reaction_ID	reactant_IDs(atom)	product_IDs(atom)	reversibility
   v1	OAA(abcd)+AcCoA(ef)	Cit(dcbfea)	0
   ```
2. **Python API** - configure labeling, flux bounds, measurements, and fitting via freeflux `Model` and fitter context managers

### models/ Directory

Contains TOML-based model configurations for glycolysis pathway models:

- **glycolysis_gly3p/**: Non-mechanistic glycolysis model with single FBA reaction
- **glycolysis_mech_gly3p/**: Mechanistic glycolysis model with decomposed FBA reaction (FBA1 + FBA2)

Each model directory contains:
- `reactions.tsv`: Reaction network with atom mappings (freeflux format)
- `glycolysis_fluxes.toml`: Net flux values for each reaction
- `glycolysis_thermodynamics.toml`: Gibbs energy (deltaG) values for reversibility calculations
- `glycolysis_network.toml`: Network topology definition

#### Mechanistic FBA Decomposition

The mechanistic model decomposes the FBA (fructose-bisphosphate aldolase) reaction into two elementary steps:
```
FBA1: fdp(abcdef) → g3p(def) + FBA_dhap(cba)   # g3p released, dhap bound to enzyme
FBA2: FBA_dhap(abc) → dhap(abc)                 # dhap released from enzyme
```

This allows separate Gibbs energy values for each step (default: FBA1=-3.0, FBA2=-2.0 kJ/mol).

## Gibbs Energy Analysis Scripts

Scripts for analyzing how Gibbs energy parameters affect isotope labeling patterns using INST (isotopically non-stationary) simulations.

### Configuration (shared across scripts)
- **Timepoints**: 100 points logarithmically spaced from 1e-3s to 1e1s, plus final at 1e3s
- **Target EMUs**: g6p, f6p, dhap, g3p (full carbon fragments)
- **Noise level**: 5% standard deviation for chi-squared calculations
- **Labeling**: Glucose input with "110000" pattern (carbons 1-2 labeled)

### Scripts

| Script | Description | Results Directory |
|--------|-------------|-------------------|
| `gibbs_landscape_2d.py` | 2D scan of FBA1 vs FBA2 Gibbs energies (30x30 grid) | `results/gibbs_landscape_2d/` |
| `gibbs_timecourse_chisquared.py` | Chi-squared evolution over time for different FBA1/FBA2 combinations | `results/gibbs_timecourse/` |
| `gibbs_comparison_1d.py` | Compare mechanistic vs non-mechanistic models (batlow colors, separate final plot) | `results/gibbs_comparison_1d/` |
| `gibbs_comparison_fba1_fixed.py` | 1D scan: FBA1 fixed at -3.0, FBA2 varies -5 to -1 (with timecourse) | `results/gibbs_comparison_fba1_fixed/` |
| `gibbs_comparison_fba2_fixed.py` | 1D scan: FBA2 fixed at -2.0, FBA1 varies -5 to -1 (with timecourse) | `results/gibbs_comparison_fba2_fixed/` |

### Output Files

**gibbs_comparison_1d/** (mechanistic vs non-mechanistic):
- `gibbs_comparison_1d.csv` - SSR vs non-mechanistic FBA dG
- `gibbs_comparison_1d.png` - Timecourse SSR plot (batlow colormap)
- `gibbs_comparison_1d_final.png` - Final timepoint (1000s) SSR plot

**gibbs_comparison_fba1_fixed/** (FBA1=-3.0 fixed):
- `gibbs_comparison_fba1_fixed.csv` - SSR vs FBA2 parameter
- `gibbs_comparison_fba1_fixed.png` - 1D SSR plot
- `timecourse_chisquared.csv` - Chi-squared at each timepoint per condition
- `timecourse_aggregated.png` - Total chi-squared vs time for each FBA2 value
- `timecourse_per_metabolite.png` - Per-metabolite chi-squared subplots

**gibbs_comparison_fba2_fixed/** (FBA2=-2.0 fixed):
- `gibbs_comparison_fba2_fixed.csv` - SSR vs FBA1 parameter
- `gibbs_comparison_fba2_fixed.png` - 1D SSR plot
- `timecourse_chisquared.csv` - Chi-squared at each timepoint per condition
- `timecourse_aggregated.png` - Total chi-squared vs time for each FBA1 value
- `timecourse_per_metabolite.png` - Per-metabolite chi-squared subplots

### Running Scripts
```bash
uv run python scripts/gibbs_landscape_2d.py        # ~15-20 min (900 simulations)
uv run python scripts/gibbs_timecourse_chisquared.py
uv run python scripts/gibbs_comparison_1d.py
uv run python scripts/gibbs_comparison_fba1_fixed.py
uv run python scripts/gibbs_comparison_fba2_fixed.py
```

### Key Functions (shared pattern)

| Function | Description |
|----------|-------------|
| `net_to_forward_backward()` | Convert net flux to forward/backward fluxes using Gibbs energy |
| `run_simulation()` | Run freeflux INST simulation with thermodynamic overrides |
| `compute_normalized_ssr()` | Calculate chi-squared/SSR between reference and test MDVs |

### Gibbs Energy to Flux Conversion

Forward/backward fluxes are computed from net flux and Gibbs energy:
```python
exp_term = exp(-deltaG / (R * T))
v_forward = v_net * exp_term / (exp_term - 1)
v_backward = v_net / (exp_term - 1)
```

Where R = 8.314e-3 kJ/(mol*K) and T = 298.15 K (default).

## Enzyme Simulation Module

### Files
- **models/simple_network/reactions.tsv**: Reaction network with atom mappings for enzyme mechanism
- **scripts/enzyme_simulation.py**: Main simulation script

### Enzyme Kinetics Model

ODE system modeling enzyme-catalyzed reaction:
```
dA/dt = v1 - v2
dQ/dt = v4 - vq_out + vq_in + v_pq
dP/dt = v5 - vp_out + vp_in - v_pq
dEA/dt = v2 - v3
dEPQ/dt = v3 - v4
dEP/dt = v4 - v5
```

With enzyme conservation: `E + EA + EPQ + EP = E_T`

Atom mapping:
- `A(ab) <-> EPQ(ab)`
- `EPQ(ab) <-> EP(a) + Q(b)`
- `P(a) <-> Q(a)` (interconversion)

Labeling: A from v1 is 100% labeled on atom 'a' only (pattern "10")

### Key Functions in enzyme_simulation.py

| Function | Description |
|----------|-------------|
| `simulate_ode()` | Solve ODE system using diffrax (Tsit5 solver) |
| `compute_rates()` | Compute reaction rates from state and parameters |
| `compute_forward_backward_fluxes()` | Compute forward/backward fluxes for freeflux |
| `run_freeflux_inst_simulation()` | Run INST isotope labeling simulation |
| `run_freeflux_ss_simulation()` | Run steady-state isotope labeling simulation |
| `compute_log_likelihood()` | Log-likelihood for fluxes, concentrations, MIDs |
| `compute_log_posterior_jax()` | JAX-compatible log-posterior for MCMC |
| `run_mcmc()` | MCMC sampling using blackjax NUTS |

### MCMC Parameter Estimation

```python
from scripts.enzyme_simulation import run_mcmc, DEFAULT_PARAMS, DEFAULT_INITIAL

results = run_mcmc(
    ref_params=DEFAULT_PARAMS,
    initial=DEFAULT_INITIAL,
    n_warmup=500,
    n_samples=1000,
)
# results["samples"] is (n_samples, 10) array of rate constants
```

**Sampled parameters** (10 rate constants):
`k2r, k3f, k3r, k4f, k4r, k5f, k5r, kq_out, kp_out, k_pq_f`

**Computed from Wegscheider constraints**: `k2f, k_pq_r`

**Fixed parameters**: `v1, vq_in, vp_in, E_T`

### Log-Likelihood Distributions
| Data Type | Distribution | Std Dev |
|-----------|-------------|---------|
| Fluxes | Normal | 0.1 |
| Concentrations | Lognormal | 0.1 |
| Steady-state MIDs | Lognormal | 0.01 |
| Timeseries MIDs | Lognormal | 0.01 |
