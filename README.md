# Elementary Fluxomics Simulations

13C metabolic flux analysis (13C-MFA) simulations using the Elementary Metabolite Unit (EMU) algorithm. Models isotope labeling patterns in metabolic networks to estimate intracellular flux distributions, with Bayesian parameter estimation via MCMC.

## Installation

Requires Python 3.10.

```bash
uv sync
```

## Scripts

Scripts are split into two folders, and their outputs follow the same split under `results/`:

- `paper_scripts/` holds the scripts behind the paper, one subfolder per analysis, with
  results in `results/paper_scripts/<set>/`. See [`paper_scripts/README.md`](paper_scripts/README.md).
- `other_scripts/` holds exploratory, superseded and supplementary scripts, with results in
  `results/other_scripts/`.

Run scripts from the repository root, e.g. `uv run python paper_scripts/gibbs_landscape/gibbs_landscape_2d.py --mode ne`.
The top-level `figures/` holds the hand-assembled (Inkscape) paper figures; figures
written by scripts go to a `figures/` subfolder of their results folder.

### Paper scripts (`paper_scripts/`)

| Folder | Script | Description |
|--------|--------|-------------|
| `common/` | `emu_jax.py` | JAX-compatible EMU MDV simulation with matrix exponential INST integration |
| `common/` | `enzyme_utils.py` | Shared constants and functions for ordered mechanism including Wegscheider constraints |
| `common/` | `style.py` | Shared plotting style |
| `forward_simulations/` | `gibbs_fba_split_comparison.py`, `gibbs_fba_split_comparison_ne.py` | Mechanistic vs aggregated forward simulations (SS + INST) far from / near equilibrium |
| `forward_simulations/` | `plot_ffe_ne_comparison.py` | 2x2 FFE/NE x SS/INST comparison figure |
| `confidence_contours/` | `gibbs_confidence_contours.py` | Near-equilibrium likelihood-ratio confidence contours, SS and INST overlaid |
| `mcmc_7_params/` | `enzyme_simulation{,_instationary,_no_labeling}.py` | MCMC fits of the 7 identifiable rate parameters |
| `mcmc_7_params/` | `plot_parameter_comparison_identifiable.py` | Credible intervals of the 7-parameter fits |
| `gibbs_landscape/` | `gibbs_landscape_2d.py` | 2D grid scan of FBA1/FBA2 Gibbs energies for the log-likelihood landscape |
| `gibbs_landscape/` | `plot_landscape_combined.py` | 2x2 NE/FFE x SS/INST landscape figure |
| `mcmc_11_params/` | `enzyme_simulation.py` | Steady-state enzyme kinetics with isotopomer labeling and MCMC parameter estimation |
| `mcmc_11_params/` | `enzyme_simulation_instationary.py` | Time-course enzyme kinetics with MCMC from instationary MID data |
| `mcmc_11_params/` | `enzyme_simulation_no_labeling.py` | Enzyme kinetics without isotope labeling for validating kinetic parameters |
| `mcmc_11_params/` | `plot_parameter_comparison.py` | Credible intervals of the 11-parameter fits |
| `parameter_identifiability/` | `fisher_information.py` | Fisher information matrix identifiability analysis |
| `standard_ne_fit/` | `gibbs_high_oxppp_fit_ne.py` | Aggregated vs mechanistic freeflux fits on the standard near-equilibrium flux map |
| `standard_ne_fit/` | `gibbs_high_oxppp_comparison_ne.py` | Simulation setup imported by the fit; run alone, compares MIDs on the high-oxPPP map |

### Other scripts (`other_scripts/`)

| Script | Description |
|--------|-------------|
| `enzyme_simulation_random_instationary.py` | Random uni-bi enzyme mechanism with instationary isotope labeling |
| `enzyme_utils_random.py` | Shared constants and functions for random mechanism |
| `psis_loo_comparison.py` | PSIS-LOO cross-validation comparing ordered vs random enzyme mechanisms |
| `gibbs_comparison_1d.py` | Compare mechanistic vs non-mechanistic model as FBA dG varies |
| `gibbs_comparison_fba1_fixed.py` | 1D scan of FBA2 with FBA1 fixed at -3.0 kJ/mol, with timecourse |
| `gibbs_comparison_fba2_fixed.py` | 1D scan of FBA1 with FBA2 fixed at -2.0 kJ/mol, with timecourse |
| `gibbs_timecourse_chisquared.py` | Timecourse chi-squared analysis where FBA1 + FBA2 = -5 kJ/mol |
| `fit_mechanistic_vs_aggregated.py` | Earlier SS/INST fits of aggregated and mechanistic models across FBA partitions |
| `compare_gibbs_energies.py` | Gibbs energies recovered by the fits of `fit_mechanistic_vs_aggregated.py` |
| `plot_flux_error_heatmaps.py` | Forward/reverse flux error heatmaps for those fits |
| `plot_pairplot.py` | Posterior pairplot of the 11-parameter MCMC fits |
| `plot_transition_state_diagram.py` | Free-energy profiles from the 7- and 11-parameter MCMC fits |

## Tests

Run the full test suite:

```bash
uv run pytest tests/ -v
```

### Test files

#### `test_emu_jax.py` — JAX EMU implementation vs freeflux

| Test | Description |
|------|-------------|
| `test_basic_convolution` | MDV convolution matches expected result |
| `test_convolution_matches_freeflux` | JAX convolution matches freeflux `gen_conv` |
| `test_convolution_with_pure_labeled` | Convolution with fully labeled MDV |
| `test_convolution_with_natural_abundance` | Convolution with natural abundance MDVs |
| `test_convolution_different_sizes` | Convolution with differently sized MDVs |
| `test_step_preserves_mass` | INST step preserves total mass in MDVs |
| `test_step_identity_system` | INST step with no flux produces no change |
| `test_jax_vs_freeflux_small_timepoints` | JAX matches freeflux for short timecourse |
| `test_jax_vs_freeflux_full_simulation` | JAX matches freeflux for full timecourse |
| `test_mdv_normalization` | Output MDVs sum to 1 |
| `test_steady_state_labeling` | Labeling reaches expected steady state |
| `test_default_simulation_produces_valid_mdvs` | Default simulation produces finite, non-negative MDVs |
| `test_mdv_sums_close_to_one` | MDVs sum to approximately 1 |
| `test_gibbs_overrides_affect_simulation` | Gibbs energy overrides change simulation output |

#### `test_emu_ss_jax.py` — Steady-state JAX EMU vs freeflux

| Test | Description |
|------|-------------|
| `test_ss_pure_jax_matches_freeflux` | Pure JAX steady-state matches freeflux native |
| `test_ss_with_different_flux_values` | Steady-state works with different flux configurations |
| `test_ss_gradient_computation` | Gradients computed through steady-state simulation |
| `test_ss_gradient_values_reasonable` | Gradients are not all zeros |
| `test_ss_jit_compilation` | Steady-state simulation can be JIT compiled |
| `test_ss_mdv_normalization` | Steady-state MDVs sum to 1 |
| `test_ss_mdv_non_negative` | Steady-state MDVs are non-negative |
| `test_ss_vs_inst_convergence` | Instationary at long time converges to steady-state |
| `test_ss_faster_than_inst` | Steady-state is faster than instationary |
| `test_ss_produces_valid_mdvs` | Steady-state produces finite MDVs |
| `test_ss_with_small_fluxes` | Steady-state works with very small flux values |

#### `test_emu_jax_mcmc.py` — Pure JAX EMU with MCMC integration

| Test | Description |
|------|-------------|
| `test_pure_jax_matches_freeflux` | Pure JAX EMU matches freeflux JAX backend |
| `test_gradient_computation` | Gradients computed through simulation |
| `test_repeated_calls_consistent` | Repeated calls produce identical results |
| `test_condition_likelihood_emu_computes` | `compute_condition_likelihood_emu` returns finite values |
| `test_condition_likelihood_emu_gradient` | Gradients computed through likelihood function |
| `test_flux_array_ordering` | Flux array has correct ordering and non-negative values |

#### `test_validate_exchange_model.py` — External pool exchange kinetics

| Test | Description |
|------|-------------|
| `test_steady_state_reached` | ODE solver reaches steady state (derivatives near zero) |
| `test_mass_balance` | Mass conservation at steady state |
| `test_exchange_direction` | Exchange responds correctly to concentration gradients |
| `test_isotopomer_consistency` | Isotopomer fractions sum to total concentrations |
| `test_labeling_patterns` | Labeling propagates correctly through network |
| `test_concentration_equilibration` | Internal concentrations approach external when k_ex is large |
| `test_derivatives_match_no_labeling` | Isotopomer ODE gives same total derivatives as no-labeling ODE |

#### `test_validate_isotopomer_vs_emu.py` — Isotopomer ODE vs freeflux EMU (validation script)

Standalone validation script (run directly, not via pytest) comparing steady-state MIDs from the custom isotopomer ODE against the freeflux EMU algorithm. Expected maximum absolute difference < 1%.

```bash
uv run python tests/test_validate_isotopomer_vs_emu.py
```
