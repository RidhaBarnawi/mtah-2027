"""Score the decoder on Metrica's public sample matches -- the only place its settings are chosen.

    python possession/evaluate_samples.py           # score decode.PARAMS on the three matches
    python possession/evaluate_samples.py --grid    # also try a grid of switch costs

Each match is scored with the official metric code (evaluation/possession_metrics.py) against the
labels from make_testset.py, next to the organisers' baseline -- nearest player to the ball, then
a majority vote over +/-50 frames -- run on the same data, so the two can be compared directly.

The sample matches are cleaner than the competition match: their ball is almost only missing while
play is dead, while the competition's is missing on 41% of graded frames, much of it in play.
Settings chosen on the clean data would trust "ball not seen = dead ball" far too much. So by
default the ball is also hidden in random stretches covering about a quarter of the frames, one
second long on average, with a fixed seed so every run gives the same numbers (--hide-ball 0 turns
this off).
"""

import argparse
import dataclasses
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "evaluation"))
import possession_metrics as pm  # noqa: E402
from decode import PARAMS, label_frames  # noqa: E402
from signals import compute_signals  # noqa: E402
from tracking import SAMPLE_DIR, load_sample  # noqa: E402

# Params
MATCHES = (1, 2, 3)
HIDE_RATE = 0.25       # share of frames on which the ball is hidden
HIDE_LENGTH = 25       # frames; mean length of one hidden stretch
SEED = 0
BASELINE_WINDOW = 50   # the organisers' published baseline is baseline_smooth50
GRID = {"team_switch": (5, 10, 20, 40, 80), "dead_switch": (5, 10, 20, 40, 80)}


def hide_ball(tracking, rate, length, seed):
    """Copy of the tracking with the ball hidden in random stretches covering about `rate` of frames.

    A two-state chain: a hidden stretch ends with probability 1/length on each frame, and one starts
    with the probability that makes the long-run hidden share equal to `rate`.
    """
    if rate <= 0:
        return tracking
    end = 1.0 / length
    start = rate / (1.0 - rate) * end
    draws = np.random.default_rng(seed).random(len(tracking.ball))
    hidden = np.zeros(len(draws), dtype=bool)
    state = False
    for i, draw in enumerate(draws):
        state = draw >= end if state else draw < start
        hidden[i] = state
    ball = tracking.ball.copy()
    ball[hidden] = np.nan
    return dataclasses.replace(tracking, ball=ball)


def baseline_codes(signals, blocks, window=BASELINE_WINDOW):
    """The organisers' baseline on these signals: the nearest player's team where the ball is seen,
    then a majority vote over +/-window frames, D where nobody has a vote."""
    n = len(signals)
    seen = signals["ball_seen"].to_numpy()
    d_home = np.nan_to_num(signals["d_home"].to_numpy(), nan=np.inf)
    d_away = np.nan_to_num(signals["d_away"].to_numpy(), nan=np.inf)
    nobody = ~seen | (np.isinf(d_home) & np.isinf(d_away))
    raw = np.where(nobody, -1, np.where(d_home <= d_away, pm.H, pm.A))

    home = np.concatenate(([0], np.cumsum(raw == pm.H)))
    away = np.concatenate(([0], np.cumsum(raw == pm.A)))
    index = np.arange(n)
    low, high = np.maximum(0, index - window), np.minimum(n, index + window + 1)
    home_votes, away_votes = home[high] - home[low], away[high] - away[low]
    voted = np.where(home_votes + away_votes == 0, pm.D,
                     np.where(home_votes >= away_votes, pm.H, pm.A))

    codes = np.full(n, pm.X, dtype=np.int8)
    for first, last in blocks:
        codes[first:last + 1] = voted[first:last + 1]
    return codes


def load_match(match, hide_rate):
    """(signals, ground-truth codes, graded blocks) for one sample match."""
    tracking = hide_ball(load_sample(match), hide_rate, HIDE_LENGTH, SEED + match)
    path = os.path.join(SAMPLE_DIR, "labels", "Sample_Game_{}_labels.csv".format(match))
    gt = pm.label_codes(pd.read_csv(path)["label"].to_numpy())
    blocks = [(start, stop - 1) for start, stop in pm.valid_runs(gt)]
    return compute_signals(tracking), gt, blocks


def score(gt, codes):
    result = pm.evaluate(gt, codes)
    return {"spellF1": result["spell_f1_mean"], "macroF1": result["macro_f1"],
            "poss": result["n_pred_spells"], "true_poss": result["n_gt_spells"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--grid", action="store_true", help="also try the grid of switch costs")
    parser.add_argument("--hide-ball", type=float, default=HIDE_RATE,
                        help="share of frames to hide the ball on (0 = off)")
    args = parser.parse_args()

    matches = {match: load_match(match, args.hide_ball) for match in MATCHES}

    rows = []
    for match, (signals, gt, blocks) in matches.items():
        rows.append({"match": match, "method": "baseline_smooth50",
                     **score(gt, baseline_codes(signals, blocks))})
        rows.append({"match": match, "method": "decoder", **score(gt, label_frames(signals, blocks))})
    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    print()
    print(table.groupby("method")[["spellF1", "macroF1"]].mean().round(4).to_string())

    if args.grid:
        results = []
        for values in itertools.product(*GRID.values()):
            params = dict(PARAMS, **dict(zip(GRID, values)))
            scores = {match: score(gt, label_frames(signals, blocks, params))
                      for match, (signals, gt, blocks) in matches.items()}
            results.append({**dict(zip(GRID, values)),
                            **{"spell_{}".format(m): s["spellF1"] for m, s in scores.items()},
                            "spell_mean": np.mean([s["spellF1"] for s in scores.values()]),
                            "macro_mean": np.mean([s["macroF1"] for s in scores.values()])})
        grid = pd.DataFrame(results).sort_values("spell_mean", ascending=False)
        print()
        print(grid.head(10).round(4).to_string(index=False))

        # Choose on two matches, score on the third: what the choice is worth on an unseen match.
        print()
        for match in matches:
            others = ["spell_{}".format(m) for m in matches if m != match]
            best = grid.loc[grid[others].mean(axis=1).idxmax()]
            print("chosen without match {}: team_switch {:g}, dead_switch {:g} -> match {} "
                  "spellF1 {:.4f}".format(match, best["team_switch"], best["dead_switch"], match,
                                          best["spell_{}".format(match)]))
