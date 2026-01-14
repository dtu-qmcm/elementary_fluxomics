# INST-EMU Algorithm: Isotopically Non-Stationary Elementary Metabolite Unit Framework

## Overview

This document describes the complete implementation of the **INST-EMU (Isotopically Non-Stationary Elementary Metabolite Unit)** algorithm as implemented in the ELMO (EMU-based Likelihood for Metabolic Observations) codebase. The INST-EMU framework enables efficient simulation of isotopic labeling dynamics in metabolic networks.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Core Concepts](#2-core-concepts)
3. [Algorithm Components](#3-algorithm-components)
   - [3.1 EMU Data Structures](#31-emu-data-structures)
   - [3.2 EMU Decomposition](#32-emu-decomposition)
   - [3.3 Forward Simulation](#33-forward-simulation)
   - [3.4 Convolution for Bimolecular Reactions](#34-convolution-for-bimolecular-reactions)
4. [Mathematical Formulation](#4-mathematical-formulation)
5. [Implementation Details](#5-implementation-details)
6. [Flux Decomposition](#6-flux-decomposition)
7. [Configuration Format](#7-configuration-format)
8. [References](#8-references)

---

## 1. Introduction

The Elementary Metabolite Unit (EMU) framework provides a computationally efficient approach to simulate isotopic labeling dynamics in metabolic networks. Rather than tracking all 2^N isotopomers for an N-carbon metabolite, EMU-based methods track only the subset of atoms relevant to experimental measurements.

**Key advantages:**
- Exponential reduction in computational complexity
- Natural handling of mass spectrometry data (MDV measurements)
- Block-decoupling of EMUs by size for efficient solving
- Support for both steady-state and non-stationary (INST) simulations

---

## 2. Core Concepts

### 2.1 Elementary Metabolite Unit (EMU)

An EMU represents a subset of atoms in a metabolite that are tracked together through the metabolic network.

```
EMU Definition:
- metabolite_id: Parent metabolite (e.g., "Glucose")
- atom_indices: Set of 0-indexed atom positions being tracked
- size: Number of atoms (len(atom_indices))

Example:
  EMU(metabolite="Pyruvate", atoms={0, 1, 2})  # Full 3-carbon pyruvate
  EMU(metabolite="Glucose", atoms={0, 1})      # First two carbons of glucose
```

### 2.2 EMU State Vector

For an n-atom EMU, the state vector has (n+1) elements representing mass isotopomer distributions:

```
x = [x₀, x₁, x₂, ..., xₙ]

Where:
  x₀ = fraction with 0 labeled atoms (M+0)
  x₁ = fraction with 1 labeled atom (M+1)
  ...
  xₙ = fraction with n labeled atoms (M+n)

Constraint: Σxᵢ = 1.0 (mass conservation)
```

### 2.3 EMU Reaction

An EMU reaction describes how atoms flow from substrate EMUs to product EMUs:

```
Types:
1. Unimolecular: A[0,1] → B[0,1]
2. Bimolecular:  B[0,1] + D[0] → C[0,1,2]  (requires convolution)
3. Source:       ∅ → A[0,1,2]  (external input)
```

---

## 3. Algorithm Components

### 3.1 EMU Data Structures

**Location:** `src/elmo/core/emu.py`

```python
@dataclass(frozen=True)
class EMU:
    """Elementary Metabolite Unit.

    Attributes:
        metabolite_id: Identifier for the parent metabolite
        atom_indices: Frozenset of 0-indexed atom positions
        size: Number of atoms in this EMU
    """
    metabolite_id: str
    atom_indices: FrozenSet[int]
    size: int

    @classmethod
    def from_atoms(cls, metabolite_id: str, atom_indices: FrozenSet[int]) -> "EMU":
        return cls(
            metabolite_id=metabolite_id,
            atom_indices=frozenset(atom_indices),
            size=len(atom_indices),
        )

@dataclass(frozen=True)
class EMUReaction:
    """EMU transformation in a metabolic reaction.

    Attributes:
        substrate_emus: Tuple of input EMUs
        product_emu: Output EMU
        flux_id: Identifier of the metabolic flux
        coefficient: Stoichiometric coefficient
        direction: "forward" or "reverse"
    """
    substrate_emus: Tuple[EMU, ...]
    product_emu: EMU
    flux_id: str
    coefficient: float
    direction: str = "forward"

    def is_bimolecular(self) -> bool:
        return len(self.substrate_emus) == 2

    def is_source(self) -> bool:
        return len(self.substrate_emus) == 0

@dataclass
class EMUState:
    """State vector for an EMU representing mass distribution.

    Attributes:
        emu: The EMU this state belongs to
        state_vector: JAX array of shape (n+1,) with mass fractions
    """
    emu: EMU
    state_vector: Array

    def validate(self, tolerance: float = 1e-10) -> None:
        total = float(jnp.sum(self.state_vector))
        if abs(total - 1.0) > tolerance:
            raise ValueError(f"EMU state doesn't sum to 1.0: {total}")
```

### 3.2 EMU Decomposition

**Location:** `src/elmo/core/decomposition.py`

The decomposition algorithm uses **backward tracing** from measured metabolites to identify the minimal set of EMUs needed for simulation.

```python
class EMUDecomposer:
    """Decompose metabolic network into EMUs via backward tracing."""

    def decompose(self, measurements: Dict[str, Set[int]]) -> EMUNetwork:
        """Decompose network starting from measured metabolites.

        Algorithm:
        1. Initialize queue with measured EMUs
        2. Use breadth-first search (BFS)
        3. For each EMU, find all reactions that produce it
        4. Trace atoms backward through reaction atom mappings
        5. Create substrate EMUs and add to queue
        6. Stop when all substrates trace to external inputs (source EMUs)
        """
        initial_emus = []
        for met_id, atom_indices in measurements.items():
            emu = EMU.from_atoms(met_id, frozenset(atom_indices))
            initial_emus.append(emu)
            self.emu_network.add_emu(emu)

        self._backward_trace(initial_emus)
        return self.emu_network

    def _backward_trace(self, initial_emus: List[EMU]) -> None:
        """Recursively trace back from required EMUs using BFS."""
        queue = deque(initial_emus)
        visited = set()

        while queue:
            product_emu = queue.popleft()
            if product_emu in visited:
                continue
            visited.add(product_emu)

            # Find all reactions that produce this EMU
            producing_reactions = self._find_producing_reactions(product_emu)

            if not producing_reactions:
                # This is a source EMU (external input)
                source_rxn = EMUReaction(
                    substrate_emus=tuple(),
                    product_emu=product_emu,
                    flux_id=f"source_{product_emu.metabolite_id}",
                    coefficient=1.0,
                )
                self.emu_network.add_reaction(source_rxn)
                continue

            # Add all producing reactions (forward and reverse)
            for rxn, substrate_emus, direction in producing_reactions:
                emu_rxn = EMUReaction(
                    substrate_emus=substrate_emus,
                    product_emu=product_emu,
                    flux_id=rxn.id,
                    coefficient=1.0,
                    direction=direction,
                )
                self.emu_network.add_reaction(emu_rxn)

                # Add substrate EMUs to queue for further tracing
                for sub_emu in substrate_emus:
                    if sub_emu not in visited:
                        queue.append(sub_emu)
```

**Atom Backward Tracing:**

```python
def _trace_atoms_backward_for_instance(
    self, product_emu: EMU, rxn: Reaction, instance_mapping: Dict[int, int]
) -> Tuple[EMU, ...] | None:
    """Trace product EMU atoms back to substrate EMUs.

    Example for reaction B + D → C:
      substrates = {'B': [{0: 0, 1: 1}], 'D': [{0: 2}]}
      products = {'C': [{0: 0, 1: 1, 2: 2}]}

      - C[0,1,2] traces back to (B[0,1], D[0])
      - C[0,1] traces back to (B[0,1],)
      - C[2] traces back to (D[0],)
    """
    # Check if all product atoms are in this instance's mapping
    product_atoms_covered = all(
        atom in instance_mapping for atom in product_emu.atom_indices
    )
    if not product_atoms_covered:
        return None

    # Get global atom indices needed from substrates
    global_atoms_needed = {
        instance_mapping[atom] for atom in product_emu.atom_indices
    }

    # Map back from global atoms to substrate atoms
    substrate_atoms: Dict[str, Set[int]] = defaultdict(set)
    for substrate_id, substrate_instances in rxn.get_substrate_instance_mappings().items():
        for substrate_instance in substrate_instances:
            for sub_atom, global_atom in substrate_instance.items():
                if global_atom in global_atoms_needed:
                    substrate_atoms[substrate_id].add(sub_atom)

    # Create substrate EMUs
    substrate_emus = []
    for sub_id in sorted(substrate_atoms.keys()):
        sub_emu = EMU.from_atoms(sub_id, frozenset(substrate_atoms[sub_id]))
        substrate_emus.append(sub_emu)

    return tuple(substrate_emus) if substrate_emus else None
```

### 3.3 Forward Simulation

**Location:** `src/elmo/core/simulator.py`

The forward simulation integrates the EMU ODE system over time:

```python
class EMUSimulator:
    """Forward simulator for EMU dynamics.

    ODE System:
        dx/dt = A @ x + convolution_terms + source_terms - dilution_terms

    Where:
        x: Vector of all EMU states
        A: Linear transition matrix (unimolecular reactions)
        convolution_terms: Contributions from bimolecular reactions
        source_terms: External substrate inputs
        dilution_terms: Pool turnover/dilution
    """

    def simulate(
        self,
        t_span: tuple[float, float],
        dt: float = 0.1,
        rtol: float = 1e-10,
        atol: float = 1e-12,
    ) -> SimulationResult:
        """Run forward simulation using diffrax ODE solver."""
        y0 = self._build_state_vector(self.initial_states)

        def ode_func(t: float, y: Array, args: None) -> Array:
            return self._compute_derivatives(t, y)

        solver = diffrax.Kvaerno5()
        term = diffrax.ODETerm(ode_func)

        solution = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=dt,
            y0=y0,
            stepsize_controller=diffrax.PIDController(rtol=rtol, atol=atol),
            max_steps=int(1e6),
        )

        return SimulationResult(
            emu_states=self._extract_emu_states(solution.ys[-1]),
            trajectory=solution.ys,
            times=solution.ts,
        )

    def _compute_derivatives(self, t: float, y: Array) -> Array:
        """Compute dy/dt for the EMU ODE system."""
        dydt = jnp.zeros_like(y)

        # Extract current EMU states
        current_states = {}
        for i, emu in enumerate(self.emu_list):
            offset = self.state_offsets[i]
            size = self.state_sizes[i]
            current_states[emu] = y[offset : offset + size]

        # Process each EMU reaction
        for emu_reaction in self.emu_network.reactions:
            product_emu = emu_reaction.product_emu
            if product_emu not in self.emu_to_idx:
                continue  # Skip external sinks

            product_idx = self.emu_to_idx[product_emu]
            product_offset = self.state_offsets[product_idx]
            product_size = self.state_sizes[product_idx]

            flux = self._get_flux_for_reaction(emu_reaction)

            if emu_reaction.is_bimolecular():
                # Bimolecular: requires convolution
                sub1, sub2 = emu_reaction.substrate_emus
                state1 = current_states.get(sub1, self.initial_states[sub1])
                state2 = current_states.get(sub2, self.initial_states[sub2])
                product_state = convolve_emu_states(state1, state2)
            else:
                # Unimolecular: direct transfer
                substrate_emu = emu_reaction.substrate_emus[0]
                product_state = current_states.get(
                    substrate_emu, self.initial_states[substrate_emu]
                )

            # Add flux-weighted contribution (normalized by pool size)
            pool_size = self.network.metabolites[product_emu.metabolite_id].pool_size
            dydt = dydt.at[product_offset : product_offset + product_size].add(
                flux * product_state / pool_size
            )

        # Apply dilution term: dx/dt -= (total_outflow / pool_size) * x
        for i, emu in enumerate(self.emu_list):
            if emu not in self.emu_network.source_emus:
                offset = self.state_offsets[i]
                size = self.state_sizes[i]
                dilution_rate = self.emu_outflow[emu]
                dydt = dydt.at[offset : offset + size].add(
                    -dilution_rate * current_states[emu]
                )

        return dydt
```

### 3.4 Convolution for Bimolecular Reactions

**Location:** `src/elmo/utils/convolution.py`

When two metabolites combine (A + B → C), the product's isotope distribution is the **convolution** of the substrate distributions:

```python
def convolve_emu_states(state_a: Array, state_b: Array) -> Array:
    """Convolve two EMU state vectors for bimolecular reactions.

    For reaction A + B → C:
        (state_a ⊗ state_b)[k] = Σ(i+j=k) state_a[i] * state_b[j]

    This represents the probability that the product has k labeled atoms
    given the substrate labeling patterns.

    Args:
        state_a: EMU state vector of shape (n_a + 1,)
        state_b: EMU state vector of shape (n_b + 1,)

    Returns:
        Convolved EMU state vector of shape (n_a + n_b + 1,)

    Example:
        B (2 atoms): [0.01, 0.00, 0.99]  # Nearly fully labeled
        D (1 atom):  [0.45, 0.55]         # Mixed labeling

        C = B ⊗ D:
        C[0] = B[0]*D[0] = 0.01*0.45 = 0.0045
        C[1] = B[0]*D[1] + B[1]*D[0] = 0.01*0.55 + 0.00*0.45 = 0.0055
        C[2] = B[1]*D[1] + B[2]*D[0] = 0.00*0.55 + 0.99*0.45 = 0.4455
        C[3] = B[2]*D[1] = 0.99*0.55 = 0.5445

        Result: [0.0045, 0.0055, 0.4455, 0.5445]
    """
    return jnp.convolve(state_a, state_b, mode='full')
```

---

## 4. Mathematical Formulation

### 4.1 EMU Balance Equations

For each internal EMU, the balance equation is:

```
d[EMU]/dt = Σ(production terms) - Σ(consumption terms)

Production:
  - Unimolecular: v_rxn × [substrate_EMU]
  - Bimolecular: v_rxn × ([EMU_A] ⊗ [EMU_B])

Consumption:
  - Dilution: (Σv_out / pool_size) × [EMU]
```

### 4.2 Matrix Form

The system can be written in matrix form:

```
dx/dt = A·x + B(x) + u

Where:
  x: State vector containing all EMU distributions
  A: Linear matrix for unimolecular reactions
  B(x): Nonlinear convolution terms for bimolecular reactions
  u: Source terms (labeled substrate inputs)
```

### 4.3 Isotopomer to MDV Relationship

For an n-atom EMU, the MDV (Mass Distribution Vector) is computed as:

```
MDV[k] = fraction of molecules with exactly k labeled atoms

For position-specific labeling with enrichments e = [e₀, e₁, ..., eₙ₋₁]:
  MDV = convolve([1-e₀, e₀], [1-e₁, e₁], ..., [1-eₙ₋₁, eₙ₋₁])
```

---

## 5. Implementation Details

### 5.1 Numerical Integration

The ODE system is integrated using JAX/diffrax:

```python
# Solver configuration
solver = diffrax.Kvaerno5()  # 5th-order implicit Runge-Kutta
stepsize_controller = diffrax.PIDController(
    rtol=1e-10,  # Relative tolerance
    atol=1e-12,  # Absolute tolerance
)
max_steps = 1_000_000
```

### 5.2 Block Decoupling by EMU Size

EMUs of different sizes can be solved independently, enabling computational efficiency:

```python
# EMUs are grouped by size
emus_by_size: Dict[int, Set[EMU]]

# Size-1 EMUs depend only on other size-1 EMUs
# Size-2 EMUs depend on size-1 and size-2 EMUs
# Size-n EMUs depend on size-1 through size-n EMUs
```

### 5.3 External Sink Handling

External metabolites (sinks) are computed algebraically rather than integrated:

```python
def _compute_external_sink_states(self, internal_states):
    """Compute external sink EMU states algebraically.

    External sinks don't have steady-state pools. Their labeling
    is the flux-weighted average of their producing reactions.
    """
    for sink_emu in self.external_sink_emus:
        producing_reactions = self.emu_network.emu_reactions.get(sink_emu, [])

        total_flux = 0.0
        weighted_state = jnp.zeros(sink_emu.size + 1)

        for emu_reaction in producing_reactions:
            flux = self._get_flux_for_reaction(emu_reaction)
            # Compute product state from substrates
            # Add flux-weighted contribution
            weighted_state += flux * product_state
            total_flux += flux

        final_state = weighted_state / total_flux if total_flux > 0 else unlabeled
```

### 5.4 Stoichiometric Reactions

Reactions with stoichiometry > 1 create multiple EMU reactions:

```python
# For reaction C → 2*D + E:
# Creates separate EMU reactions for each product instance:
#   C[0] → D[0]  (first D molecule)
#   C[1] → D[0]  (second D molecule)
#   C[2] → E[0]
```

---

## 6. Flux Decomposition

**Location:** `src/elmo/thermo/flux_decomposition.py`

For reversible reactions, net fluxes are decomposed into forward and reverse components using thermodynamic detailed balance:

```python
class FluxDecomposer:
    """Decompose net fluxes into forward/reverse components.

    Detailed balance relationship:
        ΔG/RT = ln(vr/vf)

    Where:
        ΔG: Gibbs free energy of reaction (kJ/mol)
        R: Gas constant = 8.314 J/(mol·K)
        T: Temperature (K)
        RT ≈ 2.479 kJ/mol at 25°C

    Solving:
        v_net = vf - vr
        vr/vf = exp(ΔG/RT)

        vf = v_net / (1 - exp(ΔG/RT))
        vr = vf × exp(ΔG/RT)
    """

    def _decompose_flux(self, v_net: float, delta_G: float) -> Tuple[float, float]:
        # Edge case: Equilibrium (ΔG ≈ 0)
        if abs(delta_G) < self.equilibrium_threshold:
            return abs(v_net) / 2, abs(v_net) / 2

        # Edge case: Reverse direction (v_net < 0)
        if v_net < 0:
            vf_fwd, vr_fwd = self._decompose_flux(abs(v_net), -delta_G)
            return vr_fwd, vf_fwd  # Swap roles

        # Standard decomposition
        exp_term = jnp.exp(jnp.clip(delta_G / self.RT, -50, 50))
        vf = v_net / (1 - exp_term)
        vr = vf * exp_term

        return vf, vr
```

---

## 7. Configuration Format

Networks are defined using TOML configuration files with explicit atom mappings:

```toml
[network]
name = "Example Network"

[metabolites.A]
name = "Metabolite A"
n_atoms = 3
pool_size = 10.0
is_external = true

[metabolites.B]
name = "Metabolite B"
n_atoms = 3
pool_size = 1.0
is_external = false

# Reaction with atom mapping
[reactions.v1]
name = "A to B"
reversible = false

[reactions.v1.substrate_carbons]
A = [[0, 1, 2]]  # List of instances, each with atom indices

[reactions.v1.product_carbons]
B = [[0, 1, 2]]

[reactions.v1.transition]
"0" = 0  # Global atom mapping: substrate atom → product atom
"1" = 1
"2" = 2

# Bimolecular reaction
[reactions.v2]
name = "B plus D to C"
notes = "Requires convolution"

[reactions.v2.substrate_carbons]
B = [[0, 1]]
D = [[2]]

[reactions.v2.product_carbons]
C = [[0, 1, 2]]

[reactions.v2.transition]
"0" = 0
"1" = 1
"2" = 2

# Stoichiometric reaction: C → 2*D + E
[reactions.v3]
name = "C to D and E"
notes = "Stoichiometric: creates 2 D molecules"

[reactions.v3.substrate_carbons]
C = [[0, 1, 2, 3, 4]]

[reactions.v3.product_carbons]
D = [[1, 2, 3]]      # First D instance
E = [[0], [4]]       # Two E instances (stoichiometry = 2)

[reactions.v3.transition]
"0" = 3
"1" = 0
"2" = 1
"3" = 2
"4" = 4
```

---

## 8. References

### Primary Literature

1. **Antoniewicz, M. R., Kelleher, J. K., & Stephanopoulos, G. (2007)**
   - *"Elementary metabolite units (EMU): a novel framework for modeling isotopic distributions"*
   - *Metabolic Engineering*, 9(1), 68-86
   - DOI: [10.1016/j.ymben.2006.09.001](https://doi.org/10.1016/j.ymben.2006.09.001)
   - **Foundational EMU paper defining the mathematical framework**

2. **Young, J. D., Walther, J. L., Antoniewicz, M. R., Yoo, H., & Stephanopoulos, G. (2008)**
   - *"An elementary metabolite unit (EMU) based method of isotopically nonstationary flux analysis"*
   - *Biotechnology and Bioengineering*, 99(3), 686-699
   - DOI: [10.1002/bit.21632](https://doi.org/10.1002/bit.21632)
   - [PDF](https://www.vanderbilt.edu/younglab/pdf/young08a.pdf)
   - **INST-MFA methodology extending EMU to time-course data**

3. **Jazmin, L. J., & Young, J. D. (2013)**
   - *"Isotopically nonstationary 13C flux analysis"*
   - *Current Opinion in Biotechnology*, 24(6), 1124-1133
   - DOI: [10.1016/j.copbio.2013.05.005](https://doi.org/10.1016/j.copbio.2013.05.005)
   - **Comprehensive review of INST-MFA methods**

### Implementation Resources

4. **JAX Documentation**
   - [JAX GitHub](https://github.com/google/jax)
   - Pure functional numerical computing with automatic differentiation

5. **diffrax Documentation**
   - [diffrax GitHub](https://github.com/patrick-kidger/diffrax)
   - JAX-based differential equation solvers

### Related Tools

| Tool | Description | URL |
|------|-------------|-----|
| FreeFlux | Python package for 13C metabolic flux analysis | [GitHub](https://github.com/Chaowu88/freeflux/) |
| INCA | MATLAB-based platform for MFA | [Website](http://mfa.vueinnovations.com/) |
| OpenFLUX | Open-source steady-state MFA | [GitHub](https://github.com/bzrmed/openflux) |

---

## Appendix A: Complete Architecture

```
elmo/
├── core/                   # Core EMU algorithms
│   ├── emu.py                 # EMU, EMUReaction, EMUState classes
│   ├── network.py             # Metabolite, Reaction, Network classes
│   ├── decomposition.py       # EMU decomposition algorithm (backward tracing)
│   └── simulator.py           # Forward ODE solver
├── io/                     # Input/output handling
│   └── toml_parser.py         # TOML configuration loading
├── utils/                  # Helper functions
│   └── convolution.py         # Bimolecular reaction convolution
├── likelihood/             # Likelihood calculation
│   └── mdv.py                 # MDV extraction and comparison
├── thermo/                 # Thermodynamic analysis
│   └── flux_decomposition.py  # vf/vr decomposition from ΔG
├── validation/             # Flux validation tools
│   ├── flux_balance.py        # Stoichiometric validation (S·v = 0)
│   └── gibbs.py               # Thermodynamic feasibility (flux × ΔG < 0)
└── analysis/               # Analysis utilities
    └── flux_independence.py   # Nullspace/DOF analysis
```

---

## Appendix B: Quick Reference

### EMU State Vector Sizes

| EMU Size (atoms) | State Vector Length | Example |
|------------------|---------------------|---------|
| 1 | 2 | [M+0, M+1] |
| 2 | 3 | [M+0, M+1, M+2] |
| 3 | 4 | [M+0, M+1, M+2, M+3] |
| n | n+1 | [M+0, M+1, ..., M+n] |

### Convolution Examples

| Substrate A | Substrate B | Product C |
|-------------|-------------|-----------|
| [0.5, 0.5] | [0.5, 0.5] | [0.25, 0.50, 0.25] |
| [1.0, 0.0] | [0.0, 1.0] | [0.0, 1.0, 0.0] |
| [0.0, 1.0] | [0.0, 1.0] | [0.0, 0.0, 1.0] |
| [0.01, 0.00, 0.99] | [0.45, 0.55] | [0.0045, 0.0055, 0.4455, 0.5445] |

### Key Equations

1. **Mass Conservation**: Σxᵢ = 1.0 for all EMU state vectors
2. **Convolution**: (a ⊗ b)[k] = Σ(i+j=k) a[i] × b[j]
3. **ODE System**: dx/dt = A·x + B(x) + u
4. **Detailed Balance**: vr/vf = exp(ΔG/RT)
5. **Pool Dilution**: d[EMU]/dt = (production) - (v_out/pool_size)×[EMU]

---

*Document generated from ELMO codebase, December 2025*
