"""Connector: the team's possession program (../possession) as a predict() for the practice exam.

It runs the program's own code unchanged -- signals.compute_signals and decode.label_frames, the
same two calls possession/make_submission.py makes -- and only replaces the loading step, so it
reads whichever tracking pair it is given instead of the competition match. Nothing here opens the
competition data.

    ../.venv/bin/python practice_exam.py our_program.py --name "what this version is"
    ../.venv/bin/python tracking_noise.py our_program.py
"""

import os
import sys
import xml.etree.ElementTree as ET

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TEAM_REPO = os.environ.get("TEAM_REPO") or os.path.dirname(HERE)   # this folder sits inside the repository
sys.path.insert(0, os.path.join(TEAM_REPO, "possession"))
from decode import LABELS, label_frames  # noqa: E402
from signals import compute_signals  # noqa: E402
from tracking import Tracking, _slot_teams  # noqa: E402


def pitch_of(atd_meta):
    """(length, width) in metres from the metadata's FieldSize."""
    size = ET.parse(atd_meta).getroot().find(".//FieldSize")
    return float(size.findtext("Width")), float(size.findtext("Height"))


def read_tracking(atd_meta, atd_raw, n_frames):
    """The same reading as tracking.load_competition, for any tracking pair."""
    team = _slot_teams(atd_meta)
    xy = np.full((n_frames, len(team), 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    with open(atd_raw) as handle:
        for line in handle:
            parts = line.strip().split(":")
            if len(parts) != 3:
                continue
            frame_id, slots, ball_part = parts
            row = int(frame_id) - 1
            numbers = np.array(slots.replace(";", ",").split(","), dtype=np.float32)
            xy[row] = numbers.reshape(len(team), -1)[:, :2]
            ball[row] = np.array(ball_part.split(",")[:2], dtype=np.float32)
    pitch = pitch_of(atd_meta)
    scale = np.array(pitch, dtype=np.float32)
    return Tracking(xy=xy * scale, team=team, ball=ball * scale, pitch=pitch)


def label_tracking(tracking, halves):
    """Labels for a match already in memory (tracking_noise.py uses this to test noisy copies)."""
    return LABELS[label_frames(compute_signals(tracking), halves)]


def predict(atd_meta, atd_raw, n_frames, halves):
    return label_tracking(read_tracking(atd_meta, atd_raw, n_frames), halves)
