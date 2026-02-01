#!/usr/bin/env python3
"""Isotopomer network definition for the random uni-bi enzyme mechanism.

Defines the labelled and unlabelled species that need to be tracked for the
random mechanism where either product (P or Q) can release first from EPQ.

Ordered (original): E + A -> EA -> EPQ -> EP + Q -> E + P (Q must release first)

Random (new):
                    +-> EP + Q -> E + P + Q  (path 1: Q released first)
E + A <-> EA <-> EPQ -|
                    +-> EQ + P -> E + Q + P  (path 2: P released first)

State vector: 20 species total
- A:   4 isotopomers (2 atoms: a, b)
- EA:  4 isotopomers (2 atoms: a, b)
- EPQ: 4 isotopomers (2 atoms: a, b)
- EP:  2 isotopomers (1 atom: a)
- EQ:  2 isotopomers (1 atom: b)  NEW
- Q:   2 isotopomers (1 atom: b)
- P:   2 isotopomers (1 atom: a)
"""

# State vector: 20 species total
# A:   A_00, A_01, A_10, A_11
# EA:  EA_00, EA_01, EA_10, EA_11
# EPQ: EPQ_00, EPQ_01, EPQ_10, EPQ_11
# EP:  EP_0, EP_1
# EQ:  EQ_0, EQ_1  (NEW)
# Q:   Q_0, Q_1
# P:   P_0, P_1

STATE_NAMES_RANDOM = [
    "A_00", "A_01", "A_10", "A_11",        # indices 0-3 (2 atoms)
    "EA_00", "EA_01", "EA_10", "EA_11",    # indices 4-7 (2 atoms)
    "EPQ_00", "EPQ_01", "EPQ_10", "EPQ_11",  # indices 8-11 (2 atoms)
    "EP_0", "EP_1",                         # indices 12-13 (1 atom: a)
    "EQ_0", "EQ_1",                         # indices 14-15 (1 atom: b) NEW
    "Q_0", "Q_1",                           # indices 16-17 (1 atom)
    "P_0", "P_1",                           # indices 18-19 (1 atom)
]

STATE_INDEX_RANDOM = {name: i for i, name in enumerate(STATE_NAMES_RANDOM)}

# Number of isotopomers
N_ISOTOPOMERS_RANDOM = len(STATE_NAMES_RANDOM)

# Total concentration state variable names and indices
TOTAL_CONC_NAMES = ["A", "Q", "P", "EA", "EPQ", "EP", "EQ"]
TOTAL_CONC_INDEX = {name: i for i, name in enumerate(TOTAL_CONC_NAMES)}
N_TOTAL_CONC = len(TOTAL_CONC_NAMES)
