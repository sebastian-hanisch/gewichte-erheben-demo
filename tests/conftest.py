"""Gemeinsame Fixtures: die Mini-Instanz, an der jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def w():
    """Gewichte des Entscheiders der Mini-Instanz: m = 3, w = (0,6; 0,3; 0,1).
    Rangfolge 1, 2, 3. ROC (0,6111; 0,2778; 0,1111), Rangsumme (0,5; 0,3333; 0,1667), Rangreziprok (0,5455; 0,2727; 0,1818).
    TV zu w: ROC 0,0222; Rangsumme 0,1; Rangreziprok 0,0818. Best-Worst mit a_B = (1, 2, 6), a_W = (6, 3, 1): exakt w."""
    return np.array([0.6, 0.3, 0.1])


@pytest.fixture
def swing_range():
    """Spannweiten des Wertes in der Stichprobe: Schwenk w * Delta = (0,48; 0,27; 0,05); Punkte 100 / 56 / 10; Gewichte 0,6024 / 0,3373 / 0,0602."""
    return np.array([0.8, 0.9, 0.5])
