# Elementary Fluxomics Simulations

13C metabolic flux analysis (13C-MFA) simulations using the Elementary Metabolite Unit (EMU) algorithm. Models isotope labeling patterns in metabolic networks to estimate intracellular flux distributions, with Bayesian parameter estimation via MCMC.

## Installation

Requires Python 3.10.

```bash
uv sync
```

## Scripts

### Enzyme simulation & MCMC

| Script | Description |
|--------|-------------|
| `enzyme_simulation.py` | Steady-state enzyme kinetics with isotopomer labeling and MCMC parameter estimation |
| `enzyme_simulation_instationary.py` | Time-course enzyme kinetics with MCMC from instationary MID data |
| `enzyme_simulation_no_labeling.py` | Enzyme kinetics without isotope labeling for validating kinetic parameters |
| `enzyme_simulation_random_instationary.py` | Random uni-bi enzyme mechanism with instationary isotope labeling |

### Core simulation

| Script | Description |
|--------|-------------|
| `isotopomer_simulation.py` | Isotopomer ODE tracking all labeled/unlabeled species (18-state system) |
| `isotopomer_simulation_random.py` | Isotopomer ODE for the random uni-bi mechanism (20-state system) |
| `isotopomer_network.py` | Isotopomer network definition for the ordered enzyme mechanism |
| `isotopomer_network_random.py` | Isotopomer network definition for the random uni-bi mechanism |
| `emu_jax.py` | JAX-compatible EMU MDV simulation with matrix exponential INST integration |

### Shared utilities

| Script | Description |
|--------|-------------|
| `enzyme_utils.py` | Shared constants and functions including Wegscheider thermodynamic constraints |

### Gibbs energy analysis

| Script | Description |
|--------|-------------|
| `gibbs_landscape_2d.py` | 2D grid scan of FBA1/FBA2 Gibbs energies for negative log-likelihood landscape |
| `gibbs_comparison_1d.py` | Compare mechanistic vs non-mechanistic model as FBA dG varies |
| `gibbs_comparison_fba1_fixed.py` | 1D scan of FBA2 with FBA1 fixed at -3.0 kJ/mol, with timecourse |
| `gibbs_comparison_fba2_fixed.py` | 1D scan of FBA1 with FBA2 fixed at -2.0 kJ/mol, with timecourse |
| `gibbs_timecourse_chisquared.py` | Timecourse chi-squared analysis where FBA1 + FBA2 = -5 kJ/mol |

### Serialization

| Script | Description |
|--------|-------------|
| `emu_network_serializer.py` | Serialize EMU network structure from freeflux for JAX-traceable simulation |
| `serialize_emu_network_random.py` | Serialize EMU network for the random uni-bi mechanism |

### Model comparison

| Script | Description |
|--------|-------------|
| `psis_loo_comparison.py` | PSIS-LOO cross-validation comparing ordered vs random enzyme mechanisms |

## Tests

Run the full test suite:

```bash
uv run pytest tests/ -v
```

### Test files

#### `test_flux_validation.py` — Isotopomer vs enzyme ODE flux consistency

| Test | Description |
|------|-------------|
| `test_steady_state_concentrations_match` | Summed isotopomer concentrations match enzyme ODE steady state |
| `test_net_fluxes_match` | Net fluxes from both simulations agree |
| `test_forward_backward_fluxes_match` | Forward/backward fluxes match between simulations |
| `test_enzyme_conservation` | Enzyme conservation E + EA + EPQ + EP = E_T holds |
| `test_mass_balance_at_steady_state` | Derivatives are zero at steady state |
| `test_flux_balance_at_steady_state` | Flux balances satisfied (A, Q, P, enzyme intermediates) |
| `test_all_isotopomers_non_negative` | All isotopomer concentrations are non-negative |
| `test_isotopomer_fractions_sum_to_one` | Isotopomer fractions sum to 1 for each metabolite |

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
