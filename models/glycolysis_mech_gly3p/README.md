# Glycolysis Model

This directory contains a glycolysis and pentose phosphate pathway (PPP) model configured for EMU simulation.

## Model Overview

The model includes:
- **25 metabolites**: Core glycolysis intermediates, pentose phosphate pathway metabolites, enzyme-bound carriers, and external source/sink pools
- **24 reactions**: Complete glycolysis pathway plus oxidative and non-oxidative PPP (the latter as elementary half reactions)
- **Labeling tracer**: 1,2-¹³C D-Glucose (53.74% enrichment)

## Key Metabolites

### Glycolysis
- glc__D (Glucose)
- g6p (Glucose 6-phosphate)
- f6p (Fructose 6-phosphate)
- fdp (Fructose 1,6-bisphosphate)
- dhap (Dihydroxyacetone phosphate)
- FBA_dhap (FBA-bound dihydroxyacetone phosphate)
- g3p (Glyceraldehyde 3-phosphate)
- 3pg (3-Phosphoglycerate)
- pep (Phosphoenolpyruvate)
- pyr (Pyruvate)
- lac (Lactate)

### Pentose Phosphate Pathway
- 6pgc (6-Phosphogluconate)
- ru5p__D (Ribulose 5-phosphate)
- r5p (Ribose 5-phosphate)
- xu5p__D (Xylulose 5-phosphate)
- s7p (Sedoheptulose 7-phosphate)
- e4p (Erythrose 4-phosphate)
- EC2 (Transketolase-bound 2-carbon carrier, dihydroxyethyl-TPP)
- EC3 (Transaldolase-bound 3-carbon carrier, dihydroxyacetone)

## Key Reactions

### Glycolysis
- GLUT2: Glucose transport
- HEX: Hexokinase
- GPI: Glucose-6-phosphate isomerase
- PFK: Phosphofructokinase
- FBA1: Fructose-bisphosphate aldolase, first elementary step
  (fdp -> g3p + FBA_dhap; g3p released, dhap still enzyme-bound)
- FBA2: Fructose-bisphosphate aldolase, second elementary step
  (FBA_dhap -> dhap; dhap released from the enzyme)
- TPI: Triose-phosphate isomerase
- GAPD_PGK: Combined GAPD + PGK
- PGM: Phosphoglycerate mutase
- PYK: Pyruvate kinase
- LDH: Lactate dehydrogenase

### Pentose Phosphate Pathway
- G6PDH2r_PGL: Glucose-6-phosphate dehydrogenase
- GND: 6-Phosphogluconate dehydrogenase (produces CO₂)
- RPI: Ribose-5-phosphate isomerase
- RPE: Ribulose-5-phosphate epimerase

The non-oxidative branch is written as elementary half reactions rather than the
lumped TKTL1/TKTL2/TALA overall reactions. Carbon passes through two enzyme-bound
carriers, EC2 (2 carbons, on transketolase) and EC3 (3 carbons, on transaldolase):

- X5P_EC2: xu5p__D -> EC2 + g3p (transketolase, xylulose-5-P cleavage)
- F6P_EC2: f6p -> EC2 + e4p (transketolase, fructose-6-P cleavage)
- S7P_EC2: s7p -> EC2 + r5p (transketolase, sedoheptulose-7-P cleavage)
- F6P_EC3: f6p -> EC3 + g3p (transaldolase, fructose-6-P cleavage)
- S7P_EC3: s7p -> EC3 + e4p (transaldolase, sedoheptulose-7-P cleavage)

Each overall reaction is one half reaction forward plus another in reverse:

| Overall reaction | Decomposition |
|---|---|
| TKT1: xu5p + r5p <-> s7p + g3p | X5P_EC2 forward + S7P_EC2 reverse |
| TKT2: xu5p + e4p <-> f6p + g3p | X5P_EC2 forward + F6P_EC2 reverse |
| TALA: s7p + g3p <-> e4p + f6p | S7P_EC3 forward + F6P_EC3 reverse |

Balancing the EC2 and EC3 pools gives the net fluxes. `F6P_EC2`, `S7P_EC2` and
`F6P_EC3` run backwards relative to how they are written, so they carry a
negative net flux and a positive deltaG.

## Files

- `experiment.toml`: Main experiment configuration
- `glycolysis_network.toml`: Metabolite and reaction definitions with atom mappings
- `glycolysis_fluxes.toml`: Flux distribution (100 mmol/gDCW/h glucose uptake)
- `glycolysis_labeling.toml`: 1,2-¹³C glucose tracer configuration
- `glycolysis_measurements.toml`: Measurement specifications
- `glycolysis_simulation.toml`: ODE solver settings
- `base_glycolysis_timeseries.mat`: Original MATLAB model (source data)

## Running the Simulation

From the repository root:

```bash
python scripts/run_glycolysis.py
```

Or for a simple test:

```bash
python scripts/test_glycolysis_simple.py
```

## Expected Results

With 1,2-¹³C glucose labeling (53.74% enrichment at positions C1 and C2):

- **Pyruvate**: Primarily M+0 (~61%), M+1 (~24%), M+2 (~14%)
  - M+0: unlabeled pyruvate
  - M+1: one ¹³C label
  - M+2: two ¹³C labels (from C1-C2 of glucose)

- **G6P/6PGC**: Primarily M+0 (~21%), M+1 (~50%), M+2 (~29%)
  - Reflects direct labeling from glucose

- **R5P**: M+0 (~33%), M+1 (~38%)
  - C1 of glucose lost as CO₂ in oxidative PPP

## Notes

- The model uses the explicit substrates/products format for reactions with multiple products (FBA, GND, the transketolase/transaldolase half reactions)
- Flux values are approximate and represent typical mammalian cell metabolism
- The model is at steady state (constant pool sizes and fluxes)
- Simulation runs from 0.1 to 100 seconds to reach isotopic steady state

## Model Source

Derived from `base_glycolysis_timeseries.mat` - a MATLAB-based glycolysis model with EMU framework data.
