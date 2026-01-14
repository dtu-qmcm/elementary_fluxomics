# Glycolysis Model

This directory contains a glycolysis and pentose phosphate pathway (PPP) model configured for EMU simulation.

## Model Overview

The model includes:
- **20 metabolites**: Core glycolysis intermediates plus pentose phosphate pathway metabolites
- **19 reactions**: Complete glycolysis pathway plus oxidative and non-oxidative PPP
- **Labeling tracer**: 1,2-¹³C D-Glucose (53.74% enrichment)

## Key Metabolites

### Glycolysis
- glc__D (Glucose)
- g6p (Glucose 6-phosphate)
- f6p (Fructose 6-phosphate)
- fdp (Fructose 1,6-bisphosphate)
- dhap (Dihydroxyacetone phosphate)
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

## Key Reactions

### Glycolysis
- GLUT2: Glucose transport
- HEX: Hexokinase
- GPI: Glucose-6-phosphate isomerase
- PFK: Phosphofructokinase
- FBA: Fructose-bisphosphate aldolase (splits 6C -> 3C + 3C)
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
- TKTL1, TKTL2: Transketolase reactions
- TALA: Transaldolase

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

- The model uses the explicit substrates/products format for reactions with multiple products (FBA, GND, transketolase, transaldolase)
- Flux values are approximate and represent typical mammalian cell metabolism
- The model is at steady state (constant pool sizes and fluxes)
- Simulation runs from 0.1 to 100 seconds to reach isotopic steady state

## Model Source

Derived from `base_glycolysis_timeseries.mat` - a MATLAB-based glycolysis model with EMU framework data.
