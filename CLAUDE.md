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
uv run python scripts/enzyme_simulation.py             # Steady-state with isotope labeling
uv run python scripts/enzyme_simulation_instationary.py # Instationary isotope labeling
uv run python scripts/enzyme_simulation_no_labeling.py  # No isotope labeling
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

Scripts for simulating enzyme-catalyzed reactions with isotope labeling using the freeflux EMU algorithm with a pure JAX backend.

### Files

| File | Description |
|------|-------------|
| `models/simple_network/reactions.tsv` | Reaction network with atom mappings for enzyme mechanism |
| `scripts/enzyme_utils.py` | Shared constants, ODE, and utilities for ordered mechanism |
| `scripts/enzyme_utils_random.py` | Shared constants, ODE, and utilities for random mechanism |
| `scripts/emu_jax.py` | JAX-compatible EMU simulation (builds network at runtime from freeflux) |
| `scripts/mcmc/enzyme_simulation.py` | Main simulation with steady-state EMU labeling + MCMC |
| `scripts/mcmc/enzyme_simulation_instationary.py` | Instationary (time-course) EMU labeling |
| `scripts/enzyme_simulation_no_labeling.py` | Enzyme kinetics without isotope labeling |
| `scripts/mcmc/enzyme_simulation_random_instationary.py` | Random mechanism with instationary labeling |
| `scripts/mcmc/psis_loo_comparison.py` | PSIS-LOO model comparison (ordered vs random) |
| `tests/test_validate_exchange_model.py` | Validation: external pool exchange kinetics |

### Enzyme Kinetics Model

ODE system modeling enzyme-catalyzed reaction A → P + Q with external pool exchange:
```
dA/dt = v1 - v1_out - v2
dQ/dt = v4 - vq_out + vq_in + v_pq
dP/dt = v5 - vp_out + vp_in - v_pq
dEA/dt = v2 - v3
dEPQ/dt = v3 - v4
dEP/dt = v4 - v5
```

With enzyme conservation: `E + EA + EPQ + EP = E_T`

#### Exchange Reactions

Exchange reactions are modeled as **separate in/out fluxes** for proper isotope tracking:
- `v1`: A_ext → A (input of labeled substrate)
- `v1_out`: A → A_sink (output, removes labeled material proportionally)
- `vq_in`: Q_ext_in → Q (input of unlabeled Q)
- `vq_out`: Q → Q_ext (output proportional to labeling)
- `vp_in/vp_out`: Same pattern for P

This is critical for correct isotope dilution - even with net influx, the efflux removes labeled material.

#### Atom Mapping
- `A(ab) ↔ EA(ab) ↔ EPQ(ab)` (2 atoms, preserved)
- `EPQ(ab) ↔ EP(a) + Q(b)` (split: atom 'a' to EP, atom 'b' to Q)
- `P(a) ↔ Q(a)` (interconversion, 1 atom preserved)

#### Labeling
- A_ext: first atom labeled (pattern "10")
- Q_ext_in, P_ext_in: 100% unlabeled
- Initial states: pure unlabeled (`[1, 0, ...]`)

### Isotope Simulation

The **EMU Algorithm** (`emu_jax.py`) uses freeflux's Elementary Metabolite Unit decomposition
to compute Mass Isotopomer Distributions (MIDs). The EMU network is built at runtime via
`build_emu_network()` from the reactions file.

### Wegscheider Thermodynamic Constraints

Rate constants must satisfy thermodynamic consistency. Two dependent rate constants are computed:

```python
# Overall reaction: A → P + Q
# ΔG° = μ_P + μ_Q - μ_A
K_enzyme = exp(-ΔG° / RT)
k2f = K_enzyme * (k2r * k3r * k4r * k5r) / (k3f * k4f * k5f)

# P ↔ Q interconversion
# ΔG°_pq = μ_Q - μ_P
K_pq = exp(-ΔG°_pq / RT)
k_pq_r = k_pq_f / K_pq
```

Default formation energies: A=0.0, P=0.5, Q=-2.5 kJ/mol

### Key Functions

| Function | Description |
|----------|-------------|
| `simulate_enzyme_to_steady_state_jax()` | Solve enzyme ODE to metabolic steady state |
| `build_emu_network()` | Build EMU network at runtime from reactions file |
| `compute_fluxes_for_emu()` | Compute forward/backward fluxes for EMU simulation |
| `compute_wegscheider_constraints()` | Compute k2f and k_pq_r from thermodynamics |
| `compute_log_posterior_jax_emu()` | JAX-compatible log-posterior for MCMC |
| `run_mcmc()` | MCMC sampling using blackjax NUTS |

### MCMC Parameter Estimation

```python
from scripts.enzyme_simulation import run_mcmc, DEFAULT_PARAMS, DEFAULT_INITIAL

results = run_mcmc(
    ref_params=DEFAULT_PARAMS,
    initial=DEFAULT_INITIAL,
    n_warmup=50,
    n_samples=100,
)
# results["samples"] is (n_samples, 11) array of rate constants
```

**Sampled parameters** (11 rate constants):
`k2r, k3f, k3r, k4f, k4r, k5f, k5r, k_A_ex, k_Q_ex, k_P_ex, k_pq_f`

**Computed from Wegscheider constraints**: `k2f, k_pq_r`

**Fixed per condition**: `A_e, E_T` (experimental conditions)

### Log-Likelihood Distributions

| Data Type | Distribution | Std Dev |
|-----------|-------------|---------|
| Fluxes | Normal | 0.1 |
| Concentrations | Lognormal | 0.1 |
| Steady-state MIDs | Lognormal | 0.01 |
| Timeseries MIDs | Lognormal | 0.01 |

### Validation

Run the validation tests:
```bash
uv run pytest tests/test_validate_exchange_model.py      # Exchange kinetics validation
uv run pytest tests/test_emu_jax_mcmc.py                 # EMU simulation and MCMC
uv run pytest tests/test_emu_ss_jax.py                   # Steady-state EMU
```
