Summary of revised review comments
Technical issues to address

State the forward-operation assumption explicitly. Add a standing assumption before Theorem 1: the network Ω operates at steady state with net flux in the forward direction, so ∆rG^Φ ≤ 0 for every pathway Φ ⊆ Ω. This assumption is doing real work in both proofs and should be visible to the reader.
Simplify Theorem 1's proof via a lemma. Under the forward-operation assumption, ∆rG^Ω = ∆rG^Φ + ∆rG^(Ω∖Φ) with both terms ≤ 0, so |∆rG^Φ| ≤ |∆rG^Ω|. Theorem 1 then follows in two lines from Eq. 8.
Clean up Theorem 2's proof. Remove placeholder text ("¡expr.¿"), fix the malformed flux ratio ("v^{{1,r}"), and make the logical connection between the nested-subnetwork monotonicity argument and the main inequality explicit.
Derive Equation 10. Show the derivative computation, even in a supplement. The sign argument (ζ_f > ζ_r under forward operation) is clean but currently requires the reader to take the functional form on faith.
Derive Equation 9. The flux-weighted averaging formula for branched subnetworks needs a three-line derivation showing v^{Φ,f} = v^Φ/(1−Γ^Φ), v^{Φ,r} = v^Φ·Γ^Φ/(1−Γ^Φ), and Γ^{A↔B} = Σ v^{Φ,r} / Σ v^{Φ,f}. Also resolve the Γ^Φ vs. R^Φ notational inconsistency.
Promote subnetwork multiplicativity to a proposition. The claim that species-level exchange Γ^(A↔P) is the product of exchanges over constituent subnetworks currently appears only in Fig. 1's caption. This is load-bearing and needs a proof in the main text.
Link exchange inequalities to MID bias explicitly. The inference from Γ^(Si↔Pj) > Γ^Ω to biased MIDs is currently asserted rather than demonstrated. A paragraph — or supplementary derivation — connecting exchange ratios to isotopomer dynamics would close this gap.
Delimit the scope of the theorems. Note explicitly that the results assume globally forward-operating networks. This matters for reactions (e.g., TPI, aldolase in gluconeogenic contexts) that can flip direction depending on cell state. The simulated cases are safely in scope; the framing is not.
Equation 3 under-specification. Clarify the role of ν (stoichiometry vs. summation bound) and how Eq. 3 handles metabolites appearing in multiple elementary steps with different stoichiometries.
Figure 3 and Figure 4 carry identical captions. This is a labeling conflict independent of cross-referencing — one figure is mislabeled.

Writing and presentation

Write the abstract and significance statement. Both are currently PNAS template placeholders. State the bias claim, the theoretical result, the simulation conclusion, and the inference result.
Restructure the front matter. Lead with biological motivation and a concrete example (FBA in aggregate IMFA), then introduce the elementary-reaction framework alongside Fig. 1, then present the theorems. Currently the dense notation precedes the reader's understanding of why it's needed.
Reduce notational overload. ω (elementary reaction) vs. Ω (network), η (node) vs. ν (stoichiometry), and Φ used both as pathway and superscript decorator all create friction. Consider a symbol table or renaming.
Tighten sentence-level prose. Several key sentences stack three claims into one; break them up. Strengthen passive framings of the contribution (e.g., "We argue that existing IMFA experiments overlook…" → direct statement of the bias and its consequence).
Define jargon on first use. Cleland mechanisms, ping-pong, Wegscheider condition, microscopic reversibility, detailed balance, and "instationary" all need brief glosses for a non-specialist audience.
Eliminate repetition. The species-vs-aggregate exchange inequality is restated in three places in Results. Say it once and use it as a premise.
Mechanical cleanup. Missing words, ellipses in Methods ("priors on the elementary rates were . . ."), "stop-flow" → "stopped-flow", inconsistent v^{Φ,−}/v^{Φ,+} vs. v^−/v^+ notation across Eqs. 4 and 5, and labelling/labeling consistency.

Recommendation
Major revision, primarily presentational. The mathematics is defensible under the clarified operating regime; the manuscript needs to surface the assumptions that make it so, derive the results that are currently asserted, and lead the reader into the formalism rather than starting inside it.

Proposed figure captions
Figure 2
Metabolic network of glycolysis and the pentose phosphate pathway used for simulated IMFA experiments, highlighting the aggregated versus mechanistic treatment of fructose-bisphosphate aldolase (FBA). Glucose (glc) enters via SLC transporters and is processed through the Embden-Meyerhof-Parnas pathway (HEX, PGI, PFK, FBA, TPI, GAPDH, PGK, PGM, ENO, PYK, LDH) to lactate (lac), with branching into the oxidative (G6PDH, PGL, GND) and non-oxidative (RPI, RPE, TKT1, TKT2, TALA) pentose phosphate pathway and into lipid synthesis via dhap. Boxed insets compare the two treatments of FBA considered throughout this work. In the aggregated model (upper box), FBA is represented as a single bulk step converting fdp directly into dhap + g3p, with one lumped exchange Γ^{fdp↔g3p+dhap}. In the mechanistic model (lower box), FBA is resolved as an ordered uni-bi sequence of elementary reactions (E + fdp ↔ E-fdp ↔ E-dhap-g3p ↔ E-dhap + g3p, followed by E-dhap ↔ E + dhap), which exposes two distinct exchange quantities — the pathway exchange Γ^{fdp↔g3p} and the terminal-step exchange Γ^{E-dhap↔dhap} — that the aggregated model cannot distinguish. TPI interconverts dhap and g3p. This decomposition is the basis for the bias quantified in Figs. 3–4 and for the parameter inference in Fig. 5.
Figure 3
Maximum absolute MID difference between the mechanistic and aggregated FBA models across central carbon metabolites, as a function of the fraction of ∆rG dissipated by the first product release step. Simulations use 100% [1,2-¹³C]-glucose tracer in the glycolytic network of Fig. 2. Top row: steady-state labelling. Bottom row: instationary labelling. Left column: near-equilibrium glycolysis (C. cellulolyticum-like, PPi-dependent PFK). Right column: far-from-equilibrium glycolysis (iBMK cell Gibbs energies from Park et al., 2019). Curves show max_t |MID_mech − MID_agg| for selected metabolites (g6p, f6p, dhap, g3p, r5p). The near-equilibrium condition produces deviations of up to ~5 percentage points that propagate beyond the immediate substrates of FBA, whereas the far-from-equilibrium condition produces deviations below 1 percentage point and only in the triose phosphates. In both regimes the bias is largest when ∆rG is dissipated in the intermediate catalytic step rather than the terminal release step, and instationary labelling is more sensitive to the partition than steady-state labelling.
Figure 4
Profile log-likelihood surfaces over the two-dimensional Gibbs energy partition of FBA, quantifying identifiability of the elementary energy allocation from isotope labelling. Axes are the Gibbs energy of the first substrate release step (y-axis) and the total FBA ∆rG (x-axis), both in kJ/mol. Top row: steady-state labelling. Bottom row: instationary labelling. Left column: near-equilibrium glycolysis. Right column: far-from-equilibrium glycolysis. Colour encodes the log-likelihood (LL) scaled by the number of measurements; brighter regions indicate partitions more consistent with the ground-truth system. The near-equilibrium condition shows a clearly convex likelihood landscape centred on the true partition in both steady-state and instationary experiments, confirming that the elementary energy split is identifiable. The far-from-equilibrium condition shows a substantially flatter landscape with reduced LL dynamic range, indicating that the partition is only weakly identifiable even though the maximum still coincides with the true value. Instationary labelling offers a modest but consistent improvement in identifiability over steady-state in both regimes.
Figure 5
Identifiability and posterior inference of elementary rate parameters for an ordered uni-bi enzyme under three experimental designs: no labelling, steady-state labelling, and instationary labelling. (Top) Schematic of the simulated system: an ordered uni-bi enzyme (E ↔ EA ↔ EPQ ↔ EQ ↔ E, releasing P then Q) embedded in a network with external pools A*, P*, Q* and interconverting metabolites. (Middle left) Smallest normalised eigenvalue of the Fisher information matrix as parameters are sequentially fixed by the identifiability algorithm of Quaiser & Mönnigmann (2009); the dashed line indicates the identifiability threshold (ε = 10⁻⁴). The instationary design requires the fewest parameters fixed before the remaining set becomes identifiable. (Middle right) Order in which parameters are removed from the identified set across the three designs, with identifiable parameters in pink and unidentifiable in dark blue. The instationary experiment identifies two more elementary parameters than steady-state labelling and four more than the non-labelling case; the non-labelling design resolves only a single elementary rate. (Bottom) Posterior distributions from MCMC inference of the identifiable parameters (with the forward and reverse rates of steps 2 and 3 fixed), shown on a log scale. Horizontal lines mark ground-truth values. Instationary posteriors are tight and centred on truth across all identifiable parameters, whereas non-labelling posteriors show substantial uncertainty, consistent with the Fisher-information analysis.


# Poorly constructed sentences
Poorly constructed sentences, ranked and catalogued
I went through the manuscript line by line. Below is a ranked list of the sentences with the most serious problems, followed by a consolidated table of line numbers with specific grammatical mistakes.
Ranked list (worst first)
Tier 1 — Sentences that are ungrammatical or fail to parse
1. Lines 256–258 (Theorem 2 proof opening)

"Suppose ∆rG for a multi-substrate ordered is fixed and operates under isothermal, isobaric, and isochoric conditions."

Missing noun after "ordered" (presumably "reaction"). Also copy-pasted verbatim from Theorem 1's proof, where the same mistake appears. The sentence does not parse.
2. Lines 262–266 (Theorem 2 proof body)

"We can extend Ω1 to include an additional adjacent node that is part of the existing pathways such that ¡expr. for pathway¿ Ω2 = Ω1 ∪ {ωi}|(ωi ∉ Ω1 ∧ (ηj ∈ ωi|ηj ∈ Ω1))."

Contains literal placeholder text ("¡expr. for pathway¿") that was never filled in. The set-builder notation that follows is also malformed — the scope of the "|" separator is ambiguous and ηj is introduced without quantification.
3. Lines 260–261 (Theorem 2 proof)

"Given a subnetwork Ω1 ⊂ Ω that connects all nodes between two species and only contains pathways ¡expr.¿."

Another unfilled placeholder. Sentence terminates mid-definition.
4. Lines 277–278 (Theorem 2 proof)

"v^{Ω1,r}/(vnet + v^{{1,r}) > v^{Ω2,r}/(vnet + v^{Ω2,r})"

Malformed superscript ("v^{{1,r}" — doubled brace, missing Ω). This is a typesetting error but renders the inequality unreadable.
5. Lines 211–213 (Theorem 1 proof opening)

"Suppose ∆rG for a multi-substrate ordered is fixed and operates under isothermal, isobaric, and isochoric conditions."

Same missing noun as #1. Identical defect, different location.
Tier 2 — Sentences that parse but are seriously overloaded or obscure
6. Lines 358–362 (Results, near-equilibrium simulation)

"Near-equilibrium glycolysis in both steady-state and instationary simulations demonstrated a difference of over five percentage points in the mass isotopomer distribution components (MID) from the mechanistic reaction with 10% of the ∆Gr dissipated by the first substrate release."

Three claims stacked into one sentence (experimental regime, magnitude of deviation, parameter value). The phrase "from the mechanistic reaction" is ambiguously attached — is the difference from the mechanistic reaction, or in the mechanistic reaction? Dangling modifier problem.
7. Lines 605–609 (Elementary parameter inference)

"This is in fact not the first time that labelling experiments were 1) proposed to identify elementary rate parameters, or 2) suggested to operate independent of the overall reaction [Boyer 1959]. However, this is the first time that isotope labelling experiments under steady-state conditions were proposed as a method to elucidate elementary rate parameters."

The "however" sentence is the actual contribution claim and is buried behind a clunky enumeration of prior work. The numbered list inside the first sentence breaks flow; "operate independent" should be "operate independently."
8. Lines 368–371 (Results, far-from-equilibrium)

"We only simulated a bias of approximately 0.5% in the stationary far-from equilibrium MIDs, observed in g3p, suggesting that the steady-state isotope labelling experiments are insensitive to the partition."

"Only simulated" is ambiguous — did the authors simulate only this one scenario, or did the simulation produce only a small bias? Also "far-from equilibrium" should be "far-from-equilibrium" (hyphenation inconsistent with line 362 and elsewhere).
9. Lines 16–20 (Introduction)

"In general, it is unsafe to ignore these elementary steps that treat the reaction as a bulk transfer, which convert all substrates and products in a single step."

The relative clause attaches incorrectly. As written, "these elementary steps" is the thing that "treats the reaction as a bulk transfer" — but it is the aggregate model that does this, not the elementary steps. The "which convert" clause then dangles off the misattached referent.
10. Lines 20–23 (Introduction)

"However, with the exception of transketolases, previous implementations of IMFA have made exactly this assumption, leading to avoidably biased analysis."

"Avoidably biased analysis" is awkward. "Analysis" should be "analyses" (plural, since implementations is plural) or rephrased as "avoidable bias."
Tier 3 — Sentences that are grammatical but stylistically poor
11. Lines 23–27 (Introduction)

"Our theoretical result is that the exchange between any two species, at least one of which is not a terminal binding or release step, will exceed the exchange between the terminal species."

Category error: species are not steps. The clause "at least one of which is not a terminal binding or release step" conflates species with the reaction steps they participate in. Should be something like "at least one of which is not involved in a terminal binding or release step."
12. Lines 335–340 (Results, setup)

"This enzyme is especially interesting because of its complex mechanism that has multiple traced carbon reactants - dihydroxyacetone phosphate (dhap) and glyceraldehyde-3-phosphate (g3p)."

Dash usage is inconsistent (en-dash vs. em-dash vs. hyphen appears mixed throughout). "Multiple traced carbon reactants" is awkward — "multiple ¹³C-traced products" would be clearer, and note that dhap and g3p are products of FBA, not reactants.
13. Lines 341–343 (Results, transition)

"Two conditions were used where we simulated the isotope differences between the aggregated and mechanistic models."

Unnecessary passive-to-active shift ("were used where we simulated"). Rephrase as "We simulated isotope differences… under two conditions."
14. Lines 293–298 (Results, opening)

"Hence, in random networks the exchange between a single substrate and product (e.g. S1 and P1) the reversibility will always be higher than that of all substrates (e.g. S1 + S2 ↔ P1 + P2)."

Missing preposition/comma. Should read "…between a single substrate and product (e.g. S1 and P1), the reversibility will always be higher…". As written, the subject is doubled ("the exchange… the reversibility").
15. Lines 6–10 (Introduction)

"Introduced in ..., IMFA has become widely adopted in fields ranging from metabolic engineering to health sciences and pharmacology [CITATIONS]."

Unfilled placeholder ("Introduced in ..."). Also "[CITATIONS]" is a placeholder that I was asked to ignore, but the "Introduced in ..." is a grammatical gap, not a citation gap.
16. Lines 531–533 (Instationary experiments paragraph)

"Figure 3 shows the profile likelihood of the steady-state and the profile likelihood weighted by the number of measurements of the instationary experiments."

"Profile likelihood of the steady-state" is missing a noun ("steady-state experiments" or "steady-state labelling"). The sentence also describes Figure 3 but the content described is what Figure 4 shows (see caption conflict flagged earlier).
17. Lines 596–599 (Conclusion bridge)

"Furthermore, we have shown that instationary experiments are better suited to identifying these parameters, and conversely, performing an instationary experiment without considering the mechanism is more sensitive to flux inference."

"Conversely" is the wrong logical connector — the second clause is not the converse of the first. Also "more sensitive to flux inference" is ambiguous: does it mean the experiment is more sensitive, or that flux inference is more affected? The intended meaning is probably "introduces more bias into flux inference."
18. Lines 190–192 (Species exchange definition)

"We've defined reversibility between nodes in a pathway, a sufficient metric to define the exchange between two species Γ^{A↔B} in an ordered reaction network."

Contraction ("We've") is inconsistent with the formal register used elsewhere. The appositive "a sufficient metric to define the exchange…" floats awkwardly — unclear whether it modifies "reversibility" or "pathway."
19. Lines 651–655 (Conclusion)

"This work formalised that the exchange of any subnetwork in a reaction will always be greater than the aggregate."

"Formalised that" is ungrammatical; "formalise" takes a noun phrase, not a that-clause. Should be "formalises the result that…" or "establishes that…".
20. Lines 663–669 (Conclusion)

"Standard approaches require stop-flow apparatuses – these are not only complicated to configure but are limited to single enzyme characterisations."

"Stop-flow" is incorrect; the technique is "stopped-flow." Dash separates two independent clauses that should either be joined with a semicolon or split into two sentences. "Single enzyme characterisations" needs hyphenation: "single-enzyme characterisations."

Consolidated table of grammatical defects
LinesDefect typeSpecific issue6–10Unfilled placeholder"Introduced in ..." — missing historical reference16–20Misattached relative clause"these elementary steps that treat the reaction as a bulk transfer" — the aggregate model, not elementary steps, does this20–23Number disagreement"avoidably biased analysis" — should be plural or rephrased23–27Category errorSpecies described as "steps"152–153Variable clashν used as stoichiometry and summation bound190–192Register inconsistency; dangling appositive"We've" + floating modifier211–213Missing noun"multi-substrate ordered is fixed" — noun elided256–258Missing nounSame defect as 211–213, copy-pasted260–261Unfilled placeholder"only contains pathways ¡expr.¿"262–266Unfilled placeholder + malformed set notation"¡expr. for pathway¿"; ambiguous "|" scope277–278Typesetting errorMalformed superscript "v^{{1,r}"293–298Missing punctuation; doubled subjectMissing comma after "(e.g. S1 and P1)"; "the exchange… the reversibility"335–340Reactant/product confusion; inconsistent dashesdhap/g3p are products of FBA, not reactants341–343Awkward passive-to-active shift"Two conditions were used where we simulated…"358–362Overloaded sentence; dangling modifierThree claims stacked; "from the mechanistic reaction" ambiguous368–371Ambiguous "only"; hyphenation inconsistency"only simulated" scope unclear; "far-from equilibrium" should be hyphenated531–533Elided noun; figure/caption mismatch"profile likelihood of the steady-state" — missing head noun596–599Wrong logical connector; ambiguous reference"conversely" is not the converse; "sensitive to flux inference" ambiguous605–609Buried main clause; adverb formContribution claim hidden behind prior-work enumeration; "operate independent" → "independently"651–655Wrong verb complement"formalised that" — "formalise" takes a noun phrase663–669Technical term error; dash misuse; missing hyphen"stop-flow" → "stopped-flow"; clause-joining dash; "single enzyme" → "single-enzyme"ThroughoutInconsistent hyphenation"far-from-equilibrium" vs. "far-from equilibrium"ThroughoutInconsistent symbol notationΓ^Φ vs. R^Φ for the same quantity; v^{Φ,−}/v^{Φ,+} (Eq. 5) vs. v^−/v^+ (Eq. 4)Methods (various)Ellipses as placeholders"priors on the elementary rates were . . ." — unfilled
The proofs (lines 211–290) and the results paragraph at 358–371 are the two densest clusters of defects and should be the first targets for revision. The Introduction (lines 6–27) is the third priority — it sets the tone and currently reads as placeholder-heavy draft text rather than finished prose.
