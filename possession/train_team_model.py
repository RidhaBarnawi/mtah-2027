"""Train the learned "who has the ball" clue, or check it on matches it has not seen.

    python possession/train_team_model.py            # fit on the 3 practice matches, write team_model.json
    python possession/train_team_model.py --check    # leave one match out at a time and score it

Needs the practice exam first: Metrica's three public sample matches in the competition's tracking
format, with answers built from Metrica's event data. practice/README.md says how to build it.
Nothing here reads the competition match.

The model (possession/team_model.py) predicts, for a frame where the ball is seen, whether the
team of the player nearest the ball is the team in possession. Each practice match is used three
times: as it is, and with two levels of tracking error added (practice/tracking_noise.py), so the
model does not learn the tidiness of the practice ball. The fit is a logistic regression by
Newton's method with a small ridge penalty; it is deterministic, and the weights are rounded to 4
decimals before they are saved.

--check trains on two matches and labels the third with the full program, with the model
(model_scale and team_switch as in decode.PARAMS) and without it (model_scale 0, team_switch 20),
on clean, noisy and harsh tracking. That is the comparison the model was accepted on.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "practice"))
import our_program as connector      # noqa: E402  reads a match in the competition's format
import practice_exam as exam         # noqa: E402
import tracking_noise as noise       # noqa: E402
from decode import LABELS, PARAMS, label_frames, touch_frames   # noqa: E402
from signals import compute_signals  # noqa: E402
from team_model import K, NAMES, OFFSETS, SIZES, WEIGHTS_PATH, feature_bins, usable   # noqa: E402

pm = exam.pm

# Params
TRAIN_COPIES = (("clean", 0), ("noisy", 100), ("harsh", 200))   # (error level, seed base) per match
CHECK_LEVELS = ("clean", "noisy", "harsh")
L2 = 1.0
WITHOUT_MODEL = dict(PARAMS, model_scale=0.0, team_switch=20.0)  # the program before the model


def read_match(game):
    """(tracking, answer codes, graded halves) of one practice match."""
    match = exam.exam_match(game)
    tracking = connector.read_tracking(match["atd_meta"], match["atd_raw"], match["n_frames"])
    return tracking, pm.label_codes(exam.read_answers(game)), match["halves"]


def training_rows(tracking, answers, halves):
    """(bins, y) for the frames the model learns from: in play, ball seen, a player tracked."""
    signals = compute_signals(tracking)
    bins = feature_bins(signals, touch_frames(signals, PARAMS))
    graded = np.zeros(len(answers), dtype=bool)
    for first, last in halves:
        graded[first:last + 1] = True
    rows = graded & ((answers == pm.H) | (answers == pm.A)) & usable(signals)
    nearest_is_owner = answers == signals["near_team"].to_numpy()    # codes: H = 0 = home, A = 1 = away
    return bins[rows], nearest_is_owner[rows]


def fit(bins, y, l2=L2, iterations=40):
    """Logistic regression on one-hot bins by Newton's method. Returns K weights + an intercept."""
    cols = bins + OFFSETS
    n, groups = cols.shape
    y = y.astype(np.float64)
    w = np.zeros(K + 1)

    def loss_of(weights):
        z = weights[K] + weights[cols].sum(axis=1)
        return float(np.mean(np.logaddexp(0, z) - y * z) + 0.5 * l2 * np.sum(weights[:K] ** 2) / n)

    loss = loss_of(w)
    for _ in range(iterations):
        z = w[K] + w[cols].sum(axis=1)
        p = 1.0 / (1.0 + np.exp(-z))
        residual, curve = p - y, p * (1.0 - p)
        gradient = np.zeros(K + 1)
        hessian = np.zeros((K + 1, K + 1))
        for a in range(groups):
            gradient[:K] += np.bincount(cols[:, a], weights=residual, minlength=K)
            hessian[:K, K] += np.bincount(cols[:, a], weights=curve, minlength=K)
            for b in range(a, groups):
                block = np.bincount(cols[:, a] * K + cols[:, b], weights=curve,
                                    minlength=K * K).reshape(K, K)
                hessian[:K, :K] += block if a == b else block + block.T
        gradient[K] = residual.sum()
        gradient[:K] += l2 * w[:K]
        hessian[K, :K] = hessian[:K, K]
        hessian[K, K] = curve.sum()
        hessian[np.arange(K), np.arange(K)] += l2
        step = np.linalg.solve(hessian, gradient)
        size = 1.0
        while size > 1e-4 and loss_of(w - size * step) > loss:
            size /= 2.0
        w = w - size * step
        new_loss = loss_of(w)
        if loss - new_loss < 1e-9:
            loss = new_loss
            break
        loss = new_loss
    return w, loss


def train(matches, games):
    bins, y = [], []
    for game in games:
        tracking, answers, halves = matches[game]
        for level, seed in TRAIN_COPIES:
            b, t = training_rows(noise.add_noise(tracking, level, seed + game), answers, halves)
            bins.append(b)
            y.append(t)
    bins, y = np.concatenate(bins), np.concatenate(y)
    w, loss = fit(bins, y)
    print("fitted on matches {}: {} frames, the nearest team is the owner on {:.1f}%, loss {:.4f}".format(
        list(games), len(y), 100 * y.mean(), loss), flush=True)
    return np.round(w, 4)


def save(w, path):
    with open(path, "w") as handle:
        json.dump({"weights": [round(float(v), 4) for v in w], "sizes": SIZES, "names": NAMES}, handle)


def check(matches):
    """Leave one match out: train on the other two, label it with and without the model."""
    import team_model
    kept = team_model.load_weights
    scores = {level: {"without": [], "with": []} for level in CHECK_LEVELS}
    try:
        for game in exam.GAMES:
            w = train(matches, [g for g in exam.GAMES if g != game])
            team_model.load_weights = lambda path=None, w=w: w       # use these weights, not the saved ones
            tracking, answers, halves = matches[game]
            for level in CHECK_LEVELS:
                signals = compute_signals(noise.add_noise(tracking, level, noise.SEED + game))
                for name, params in (("without", WITHOUT_MODEL), ("with", PARAMS)):
                    codes = label_frames(signals, halves, params)
                    scores[level][name].append(pm.evaluate(answers, codes)["spell_f1_mean"])
                print("match {} {:<5}  without the model {:.4f}   with it {:.4f}".format(
                    game, level, scores[level]["without"][-1], scores[level]["with"][-1]), flush=True)
    finally:
        team_model.load_weights = kept
    print("\naverage spellF1 on the match left out")
    for level in CHECK_LEVELS:
        without, with_model = np.mean(scores[level]["without"]), np.mean(scores[level]["with"])
        print("  {:<5}  without the model {:.4f}   with it {:.4f}   ({:+.4f})".format(
            level, without, with_model, with_model - without))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true",
                        help="leave one match out at a time and score it; writes nothing")
    parser.add_argument("--out", default=WEIGHTS_PATH, help="where to write the weights")
    args = parser.parse_args()

    matches = {game: read_match(game) for game in exam.GAMES}
    if args.check:
        check(matches)
    else:
        save(train(matches, exam.GAMES), args.out)
        print("wrote", args.out)
