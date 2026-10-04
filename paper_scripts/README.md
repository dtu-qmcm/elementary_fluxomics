# Paper scripts

The scripts behind the paper's results, one folder per analysis. Each set writes to
`results/paper_scripts/<set>/`, and figures go to the `figures/` subfolder there. Run
everything from the repository root. Within a set, run the simulation or fitting
scripts before the plotting scripts.

`common/` holds the shared modules: `style.py` (plotting style), `emu_jax.py` (JAX EMU
simulation) and `enzyme_utils.py` (ordered-mechanism ODE and Wegscheider constraints).

| Set | Scripts | Outputs (`results/paper_scripts/...`) |
|-----|---------|----------------------------------------|
| `forward_simulations/`: mechanistic vs aggregated forward simulations of glycolysis near and far from equilibrium, SS + INST | `gibbs_fba_split_comparison.py` (far from equilibrium), `gibbs_fba_split_comparison_ne.py` (near equilibrium), then `plot_ffe_ne_comparison.py` | `forward_simulations/gibbs_fba_split/`, `gibbs_fba_split_ne/`, `figures/ffe_ne_comparison.*` (`ffe_ne_comparison/` is an older output of the same comparison) |
| `confidence_contours/`: near-equilibrium likelihood-ratio confidence contours, SS and INST combined | `gibbs_confidence_contours.py` (uses `gibbs_landscape/gibbs_landscape_2d.py`) | `confidence_contours/gibbs_confidence_contours/` |
| `mcmc_7_params/`: MCMC fits of the 7 identifiable rate parameters | `enzyme_simulation.py`, `enzyme_simulation_instationary.py`, `enzyme_simulation_no_labeling.py`, then `plot_parameter_comparison_identifiable.py` | `mcmc_7_params/mcmc_identifiable_*/`, `figures/parameter_comparison_identifiable.*` |
| `gibbs_landscape/`: FBA1/FBA2 Gibbs energy grid and log-likelihoods, near and far from equilibrium, SS + INST | `gibbs_landscape_2d.py --mode ffe` and `--mode ne`, then `plot_landscape_combined.py` | `gibbs_landscape/gibbs_landscape_2d_{ffe,ne}/`, `figures/gibbs_landscape_2d_combined.*` (`gibbs_landscape_2d/` is an older output) |
| `mcmc_11_params/`: MCMC fits of all 11 rate parameters | `enzyme_simulation.py`, `enzyme_simulation_instationary.py`, `enzyme_simulation_no_labeling.py`, then `plot_parameter_comparison.py` | `mcmc_11_params/enzyme_simulation*/`, `figures/parameter_comparison.*` |
| `parameter_identifiability/`: Fisher information identifiability analysis | `fisher_information.py` | `parameter_identifiability/fisher_information/`, `figures/fim_*.pdf` |
| `standard_ne_fit/`: aggregated vs mechanistic fits for the standard near-equilibrium condition | `gibbs_high_oxppp_fit_ne.py` (imports its setup from `gibbs_high_oxppp_comparison_ne.py`) | `standard_ne_fit/gibbs_standard_fit_ne/` (`gibbs_high_oxppp_ne/` is the comparison script's own output) |

```bash
uv run python paper_scripts/gibbs_landscape/gibbs_landscape_2d.py --mode ne
uv run python paper_scripts/gibbs_landscape/plot_landscape_combined.py
uv run python paper_scripts/confidence_contours/gibbs_confidence_contours.py --plot-only
uv run python paper_scripts/standard_ne_fit/gibbs_high_oxppp_fit_ne.py --plot-only
```

Scripts in different sets import each other by adding the other folder to `sys.path`,
so keep the folder names as they are. The MCMC scripts can also be imported as packages
from the repository root (`paper_scripts.mcmc_11_params.enzyme_simulation`), which is
how `tests/` uses them.
