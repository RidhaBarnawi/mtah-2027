"""A learned clue for who has the ball: a small logistic regression on binned signals.

For every frame where the ball is seen, it gives the log-odds that the team of the player nearest
the ball is the team in possession. It looks at how close that player is and whether the ball
moves with him, how close the other team is, the ball's speed, kinks in the ball's path, how
crowded the ball is, and which team touched the ball last and touches it next (the owner is the
last team to play the ball, so a ball in flight stays with the team that played it).

The model is a sigmoid over a sum of weights, one weight per bin of each feature -- the logistic
regression of an ML course, with binned features so that it can follow curves. The weights are in
team_model.json, as plain numbers, so labelling a match needs nothing but numpy.

Training (possession/train_team_model.py) used only Metrica's three public sample matches, in the
competition's tracking format, each as a clean copy and two copies with tracking errors added.
The label is whether the nearest player's team is the owner, from Metrica's own event data.
Nothing from the competition match went into it.
"""

import json
import os

import numpy as np

from signals import FPS

HERE = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_PATH = os.path.join(HERE, "team_model.json")

# Bin edges. Every feature also gets one more bin for "unknown" where it can be unknown.
D_EDGES = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 10.0])     # m, nearest player to the ball
R_EDGES = np.array([1.0, 2.0, 4.0, 8.0])                          # m/s, ball relative to a player
GAP_EDGES = np.array([0.5, 1.0, 2.0, 4.0, 8.0])                   # m, other team's nearest minus ours
PACE_EDGES = np.array([1.0, 3.0, 6.0, 10.0, 15.0])                # m/s, the ball's own speed
KINK_EDGES = np.array([1.0, 2.0, 4.0, 8.0])                       # m/s, kink now
MAXKINK_EDGES = np.array([2.0, 4.0, 8.0, 12.0])                   # m/s, strongest kink within a second
TOUCH_TIME_EDGES = np.array([0.5, 1.0, 2.0, 4.0])                 # s, since the last / until the next touch
TOUCH_HORIZON = 6.0                                               # s; an older or later touch counts as none

NAMES = ["distance x relative speed", "gap to other team", "relative speed, other team", "ball speed",
         "kink", "max kink last second", "max kink next second", "players within 3 m",
         "last touch", "next touch"]
SIZES = [(len(D_EDGES) + 1) * (len(R_EDGES) + 2), len(GAP_EDGES) + 2, len(R_EDGES) + 2,
         len(PACE_EDGES) + 2, len(KINK_EDGES) + 1, len(MAXKINK_EDGES) + 1, len(MAXKINK_EDGES) + 1,
         16, 21, 21]
OFFSETS = np.concatenate(([0], np.cumsum(SIZES)[:-1]))
K = int(np.sum(SIZES))   # number of weights, plus one intercept at the end


def load_weights(path=WEIGHTS_PATH):
    with open(path) as handle:
        saved = json.load(handle)
    if saved["sizes"] != SIZES:
        raise ValueError("{} was trained with other bins than this code uses".format(path))
    return np.array(saved["weights"], dtype=np.float64)


def binned(x, edges):
    """Bin index 0..len(edges); one more bin, the last, for unknown (NaN or infinite)."""
    out = np.digitize(x, edges)
    out[~np.isfinite(x)] = len(edges) + 1
    return out


def touch_category(touches, near, seen, forward):
    """Per frame: 0 = no touch within the horizon; else 1 + same*10 + gap*5 + time bin, for the
    last touch (forward=False) or the next one (forward=True).

    same: the touch was by the team that is nearest now. gap: the ball went unseen in between, so
    something may have happened that the tracking did not show.
    """
    n = len(near)
    index = np.arange(n)
    if not forward:
        mark = np.full(n, -1, dtype=np.int64)
        mark[touches] = touches
        at = np.maximum.accumulate(mark)
        unseen = np.maximum.accumulate(np.where(~seen, index, -1))
        has = at >= 0
        gap = unseen > at
        seconds = np.where(has, (index - at) / float(FPS), np.inf)
    else:
        mark = np.full(n, n, dtype=np.int64)
        mark[touches] = touches
        at = np.minimum.accumulate(mark[::-1])[::-1]
        at = np.concatenate((at[1:], [n]))                      # strictly after this frame
        unseen = np.minimum.accumulate(np.where(~seen, index, n)[::-1])[::-1]
        has = at < n
        gap = unseen < at
        seconds = np.where(has, (at - index) / float(FPS), np.inf)
    same = near[np.clip(at, 0, n - 1)] == near
    time_bin = np.digitize(seconds, TOUCH_TIME_EDGES)
    category = 1 + same.astype(np.int64) * 10 + gap.astype(np.int64) * 5 + time_bin
    return np.where(has & (seconds <= TOUCH_HORIZON), category, 0)


def feature_bins(signals, touches):
    """(n_frames, 10) bin numbers: one column per feature in NAMES."""
    near = signals["near_team"].to_numpy()
    seen = signals["ball_seen"].to_numpy()
    d_near = signals["d_near"].to_numpy()
    d_other = signals["d_other"].to_numpy()
    gap = np.where(np.isfinite(d_other), d_other, np.inf) - np.where(np.isfinite(d_near), d_near, 0.0)
    kink = np.nan_to_num(signals["kink"].to_numpy(), nan=0.0)
    return np.stack([
        np.digitize(d_near, D_EDGES) * (len(R_EDGES) + 2) + binned(signals["rel_speed"].to_numpy(), R_EDGES),
        binned(gap, GAP_EDGES),
        binned(signals["rel_other"].to_numpy(), R_EDGES),
        binned(signals["ball_pace"].to_numpy(), PACE_EDGES),
        np.digitize(kink, KINK_EDGES),
        np.digitize(signals["kink_before"].to_numpy(), MAXKINK_EDGES),
        np.digitize(signals["kink_after"].to_numpy(), MAXKINK_EDGES),
        signals["c_near"].to_numpy() * 4 + signals["c_other"].to_numpy(),
        touch_category(touches, near, seen, forward=False),
        touch_category(touches, near, seen, forward=True),
    ], axis=1).astype(np.int64)


def usable(signals):
    """Frames the model speaks on: the ball is seen on the pitch and a player is tracked."""
    return (signals["ball_seen"].to_numpy() & ~signals["ball_off"].to_numpy()
            & np.isfinite(signals["d_near"].to_numpy()))


def team_logit(signals, touches, weights=None):
    """Per frame: the log-odds that the nearest player's team has the ball. 0 where it cannot say."""
    w = load_weights() if weights is None else weights
    z = w[K] + w[feature_bins(signals, touches) + OFFSETS].sum(axis=1)
    return np.where(usable(signals), z, 0.0)
