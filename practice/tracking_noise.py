"""Score a program on the practice matches with realistic tracking errors added.

    ../.venv/bin/python tracking_noise.py                 # our_program.py, the team's program
    ../.venv/bin/python tracking_noise.py my_model.py

The program file needs one function:

    def label_tracking(tracking, halves):
        return labels      # one of "H", "A", "D", "X" per frame

where tracking has .xy (frames, slots, 2), .team (slots), .ball (frames, 2) and .pitch, in
metres, like Tracking in possession/tracking.py. our_program.py shows how.

WHY. The practice ball was drawn between events: it moves in straight lines at constant speed
and sits on the player who has it. Anything that reads fine ball movement -- a touch detector,
a learned model -- therefore looks better on the clean practice matches than it will on the
competition match, whose tracking is automatic and uncorrected. A change is worth keeping when
it helps on "noisy" as well as on "clean", and does no harm on "harsh".

LEVELS (standard deviations in metres)
    clean   nothing added
    noisy   ball: drift 0.5 + shake 0.06    players: drift 0.25 + shake 0.03
    harsh   ball: drift 1.0 + shake 0.12    players: drift 0.5  + shake 0.05
drift is an error that changes slowly, over about a second; shake is independent on every
frame. Seeds are fixed, so every run gives the same numbers. The levels are our choice: nobody
outside the team has measured the competition match's errors.

Use add_noise() on training matches too, so a model does not learn the tidiness.
"""

import dataclasses
import importlib.util
import os
import sys

import numpy as np

import practice_exam as pe
from our_program import read_tracking

FPS = 25
LEVELS = {  # (ball drift, ball shake, player drift, player shake)
    "clean": (0.0, 0.0, 0.0, 0.0),
    "noisy": (0.5, 0.06, 0.25, 0.03),
    "harsh": (1.0, 0.12, 0.5, 0.05),
}
SEED = 900                 # + the match number
DRIFT_SECONDS = 1.0        # how slowly the drift changes

_t = np.arange(-3 * DRIFT_SECONDS * FPS, 3 * DRIFT_SECONDS * FPS + 1)
_KERNEL = np.exp(-0.5 * (_t / (DRIFT_SECONDS * FPS)) ** 2)
_KERNEL /= _KERNEL.sum()


def drift(rng, shape, sigma):
    """A slowly changing error with standard deviation sigma, one series per column."""
    white = rng.normal(0, 1, shape)
    smooth = np.apply_along_axis(lambda v: np.convolve(v, _KERNEL, mode="same"), 0, white)
    return (smooth / smooth.std() * sigma).astype(np.float32)


def add_noise(tracking, level, seed):
    """A copy of the tracking with the errors of `level` added. Missing positions stay missing."""
    ball_drift, ball_shake, player_drift, player_shake = LEVELS[level]
    if level == "clean":
        return tracking
    rng = np.random.RandomState(seed)
    ball = (tracking.ball + drift(rng, tracking.ball.shape, ball_drift)
            + rng.normal(0, ball_shake, tracking.ball.shape).astype(np.float32))
    xy = (tracking.xy + drift(rng, tracking.xy.shape, player_drift)
          + rng.normal(0, player_shake, tracking.xy.shape).astype(np.float32))
    return dataclasses.replace(tracking, ball=ball, xy=xy)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(pe.HERE, "our_program.py")
    spec = importlib.util.spec_from_file_location("program_under_test", path)
    program = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(program)
    if not hasattr(program, "label_tracking"):
        raise SystemExit("%s has no label_tracking(tracking, halves) function" % path)

    matches = {}
    for game in pe.GAMES:
        match = pe.exam_match(game)
        matches[game] = (read_tracking(match["atd_meta"], match["atd_raw"], match["n_frames"]),
                         pe.pm.label_codes(pe.read_answers(game)), match["halves"])

    print("%s, average spellF1 on the 3 practice matches\n" % os.path.basename(path))
    print("%-8s %8s   %s" % ("", "average", "per match"))
    for level in LEVELS:
        scores = []
        for game, (tracking, answers, halves) in matches.items():
            labels = program.label_tracking(add_noise(tracking, level, SEED + game), halves)
            labels = pe.clean_labels(labels, len(answers))
            scores.append(pe.pm.evaluate(answers, pe.pm.label_codes(labels))["spell_f1_mean"])
        print("%-8s %8.4f   %s" % (level, np.mean(scores), "  ".join("%.3f" % s for s in scores)),
              flush=True)


if __name__ == "__main__":
    main()
