"""Turn per-frame signals into H/A/D labels: a dead-ball detector and a possession decoder.

Every frame gets two pieces of evidence from the signals (possession/signals.py):

    dead   how much the frame looks like a stoppage: the ball off the pitch or not seen, players
           walking, the ball lying still
    team   how much the frame looks like home rather than away possession: which team's nearest
           player is closer to the ball, counted only when someone is close enough to have it

Both are logits (log-odds), which give the probability of each of the three states:

    P(D) = sigmoid(dead)
    P(H) = (1 - P(D)) * sigmoid(team)
    P(A) = (1 - P(D)) * sigmoid(-team)

Labelling each frame by its most likely state would flicker, and the ranking metric punishes
flicker because it scores whole possessions. So the labels are the single best path through each
half (the Viterbi algorithm), where every change of state pays a fixed cost: TEAM_SWITCH between H
and A, DEAD_SWITCH into or out of D. A label only changes when the evidence for the change, added
up over the frames that follow, outweighs that cost. Each half is decoded as a whole -- the task is
post-match, so looking ahead is allowed (organisers' answer to repo issue #15).

Every number in PARAMS was set by reasoning about football or chosen on Metrica's public sample
matches with possession/evaluate_samples.py -- never on the competition match.
"""

import numpy as np

# The scorer's codes, from evaluation/possession_metrics.py
H, A, D, X = 0, 1, 2, 3
LABELS = np.array(["H", "A", "D", "X"])

# Params
PARAMS = {
    # Dead-ball evidence, as a logit
    "dead_bias": -1.5,     # no evidence either way: probably live
    "ball_unseen": 1.0,    # added when the ball is not seen
    "ball_off": 3.0,       # added when the ball is outside the pitch
    "slow_players": 1.5,   # per m/s that the players' mean speed is below walk_speed (or above: minus)
    "walk_speed": 1.5,     # m/s
    "ball_still": 1.0,     # added when the ball is seen on the pitch and barely moving
    "still_speed": 0.5,    # m/s
    # Team evidence, as a logit
    "team_weight": 3.0,    # one team clearly closer and someone close enough to have the ball
    "team_margin": 3.0,    # m; the gap between the two nearest players that counts as clearly closer
    "control_dist": 2.0,   # m; nearest player this close or closer: full weight
    "loose_dist": 6.0,     # m; nearest player this far or further: no weight (loose ball, in flight)
    # Costs of changing state
    "team_switch": 20.0,
    "dead_switch": 20.0,
}
SPEED_CLIP = 1.5  # m/s; caps the players-speed evidence either way, so tracking noise can't dominate


def evidence(signals, params=PARAMS):
    """(dead, team): one logit each per frame. See the module docstring."""
    p = params
    seen = signals["ball_seen"].to_numpy()
    off = signals["ball_off"].to_numpy()
    ball_speed = signals["ball_speed"].to_numpy()
    players_speed = signals["players_speed"].fillna(p["walk_speed"]).to_numpy()
    still = seen & ~off & (ball_speed < p["still_speed"])  # NaN speed compares False

    dead = (p["dead_bias"]
            + p["ball_unseen"] * ~seen
            + p["ball_off"] * off
            + p["slow_players"] * np.clip(p["walk_speed"] - players_speed, -SPEED_CLIP, SPEED_CLIP)
            + p["ball_still"] * still)

    d_home = signals["d_home"].to_numpy()
    d_away = signals["d_away"].to_numpy()
    nearest = np.fmin(d_home, d_away)  # fmin ignores a NaN on one side
    closeness = np.clip((p["loose_dist"] - nearest) / (p["loose_dist"] - p["control_dist"]), 0, 1)
    gap = np.clip((d_away - d_home) / p["team_margin"], -1, 1)  # positive when home is closer
    team = p["team_weight"] * gap * closeness
    team[~seen | off | np.isnan(team)] = 0.0  # no ball, or one team not seen: no team evidence
    return dead, team


def log_probs(dead, team):
    """Per-frame log-probabilities of H, A and D, shape (n_frames, 3)."""
    log_dead = -np.logaddexp(0, -dead)  # log sigmoid(dead)
    log_live = -np.logaddexp(0, dead)   # log sigmoid(-dead)
    out = np.empty((len(dead), 3))
    out[:, H] = log_live - np.logaddexp(0, -team)
    out[:, A] = log_live - np.logaddexp(0, team)
    out[:, D] = log_dead
    return out


def best_path(log_p, team_switch, dead_switch):
    """The sequence of states with the highest total log-probability minus switch costs (Viterbi).

    Ties go to the lower state code, so the result is the same on every machine.
    """
    cost = np.array([[0.0, team_switch, dead_switch],
                     [team_switch, 0.0, dead_switch],
                     [dead_switch, dead_switch, 0.0]])
    n = len(log_p)
    back = np.empty((n, 3), dtype=np.int8)
    score = log_p[0].copy()
    states = np.arange(3)
    for i in range(1, n):
        options = score[:, None] - cost  # options[from, to]
        back[i] = options.argmax(axis=0)
        score = options[back[i], states] + log_p[i]
    path = np.empty(n, dtype=np.int8)
    path[-1] = score.argmax()
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def label_frames(signals, blocks, params=PARAMS):
    """Scorer codes for every frame: H/A/D inside `blocks`, X everywhere else.

    blocks: [(first_row, last_row)], inclusive, one per half. Each half is decoded on its own, so
    no possession can run across halftime.
    """
    dead, team = evidence(signals, params)
    log_p = log_probs(dead, team)
    codes = np.full(len(signals), X, dtype=np.int8)
    for first, last in blocks:
        codes[first:last + 1] = best_path(log_p[first:last + 1],
                                          params["team_switch"], params["dead_switch"])
    return codes
