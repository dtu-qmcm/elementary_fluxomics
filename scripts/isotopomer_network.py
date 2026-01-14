#!/usr/bin/env python3
"""Isotopomer network definition for the enzyme mechanism.

Defines the labelled and unlabelled species that need to be tracked.
"""

# Species with 2 atoms (a, b): track 4 isotopomers each
# Species with 1 atom: track 2 isotopomers each (labelled/unlabelled)

# State vector: 18 species total
# A:   A_00, A_01, A_10, A_11
# EA:  EA_00, EA_01, EA_10, EA_11
# EPQ: EPQ_00, EPQ_01, EPQ_10, EPQ_11
# EP:  EP_0, EP_1
# Q:   Q_0, Q_1
# P:   P_0, P_1

STATE_NAMES = [
    "A_00", "A_01", "A_10", "A_11",
    "EA_00", "EA_01", "EA_10", "EA_11",
    "EPQ_00", "EPQ_01", "EPQ_10", "EPQ_11",
    "EP_0", "EP_1",
    "Q_0", "Q_1",
    "P_0", "P_1",
]

STATE_INDEX = {name: i for i, name in enumerate(STATE_NAMES)}
