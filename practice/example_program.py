"""Example program for the practice exam: the organisers' baseline (starter/baseline.py).

Whoever is closest to the ball has it; where the ball is not seen, nobody does (D). Then a
majority vote over +/- 2 s smooths out the flicker. On the full competition match this is the
published floor, spellF1 0.2433.

To start your own program, copy this file and change predict(). practice_exam.py needs nothing
else from it.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MTAH_REPO") or os.path.dirname(HERE)      # this folder sits inside the repository
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "starter"))
import baseline  # noqa: E402

PITCH = (105.0, 68.0)
WINDOW = 50   # frames on each side of the majority vote: 50 frames = 2 s


def predict(atd_meta, atd_raw, n_frames, halves):
    """One label per tracking line: H, A, D (or X where you do not want to answer)."""
    codes = baseline.nearest_player_labels(atd_raw, baseline.team_of_each_track(atd_meta),
                                           n_frames, PITCH)
    return baseline.smooth(codes, WINDOW)
