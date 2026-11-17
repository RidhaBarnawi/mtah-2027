"""Turn per-frame signals into H/A/D labels: a dead-ball detector and a possession decoder.

Every frame gets two pieces of evidence from the signals (possession/signals.py):

    dead   how much the frame looks like a stoppage: the ball off the pitch or not seen, players
           walking, the ball lying still
    team   how much the frame looks like home rather than away possession. Three clues:
           1. which team's nearest player is closer to the ball, counted only when he is close
              enough to have it (full weight within 1.5 m, none beyond 3 m)
           2. whether the ball moves with that player: the clue counts in full when the ball's
              speed relative to him is under 3 m/s and not at all above 8 m/s (a ball flying past)
           3. touches: a sharp kink in the ball's path while a player is within 2 m adds a burst
              of evidence for his team. This is what catches one-touch passes and interceptions.
              Touches are only used when the ball track is steady enough to show them -- see
              touch_max_shake below.
           4. a learned clue (possession/team_model.py): a small logistic regression that gives,
              for every frame where the ball is seen, the log-odds that the nearest player's team
              is the owner. It adds what the hand-made clues cannot see, above all which team
              touched the ball last. It is added to the three clues above.

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
matches -- never on the competition match. The matches were rewritten in the competition's
tracking format and scored clean and with tracking errors added (the practice exam, see
practice/README.md); a setting was kept only when it helped with the errors added too.
"""

import numpy as np

from team_model import team_logit

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
    "team_weight": 6.0,    # one team clearly closer, its player on the ball and moving with it
    "team_margin": 3.0,    # m; the gap between the two nearest players that counts as clearly closer
    "control_dist": 1.5,   # m; nearest player this close or closer: full weight
    "loose_dist": 3.0,     # m; nearest player this far or further: no weight (loose ball, in flight)
    "with_player_speed": 3.0,   # m/s; ball this slow relative to the nearest player: full weight
    "flying_past_speed": 8.0,   # m/s; this fast or faster: no weight
    "unknown_weight": 0.5,      # weight when that speed is unknown (edge of a ball gap or a track)
    # Touches
    "touch_kink": 4.0,     # m/s; a smaller change in the ball's velocity is not a touch
    "touch_reach": 2.0,    # m; the nearest player must be this close
    "touch_window": 4,     # frames; one touch per kink, at its strongest frame
    "touch_burst": 30.0,   # evidence added for the toucher's team
    "touch_max_shake": 2.4,     # m/s; touches are used only on a ball track steadier than this
    # The learned clue
    "model_scale": 2.0,    # the model's log-odds count this many times (0 turns the model off)
    # Costs of changing state
    "team_switch": 60.0,
    "dead_switch": 20.0,
}
SPEED_CLIP = 1.5  # m/s; caps the players-speed evidence either way, so tracking noise can't dominate


def evidence(signals, params=PARAMS):
    """(dead, team): one logit each per frame, from the ball and the nearest players' distances."""
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


def with_player(signals, params=PARAMS):
    """Per frame, 0 to 1: how much the ball is moving with the nearest player rather than past him."""
    p = params
    relative = signals["rel_speed"].to_numpy()
    weight = np.clip((p["flying_past_speed"] - relative)
                     / (p["flying_past_speed"] - p["with_player_speed"]), 0, 1)
    weight[np.isnan(weight)] = p["unknown_weight"]
    return weight


def touch_frames(signals, params=PARAMS):
    """Frames where a player touches the ball: the strongest frame of each kink near a player."""
    p = params
    kink = np.nan_to_num(signals["kink"].to_numpy(), nan=0.0)
    strongest = kink.copy()
    for shift in range(1, p["touch_window"] + 1):
        strongest[shift:] = np.maximum(strongest[shift:], kink[:-shift])
        strongest[:-shift] = np.maximum(strongest[:-shift], kink[shift:])
    return np.flatnonzero((kink >= p["touch_kink"]) & (kink >= strongest)
                          & (signals["d_near"].to_numpy() <= p["touch_reach"]))


def ball_shake(signals, blocks):
    """How jittery the ball track is: the median kink, in m/s, over the graded frames.

    About 0.1 on Metrica's clean sample tracking. Every 6 cm of frame-to-frame shake in the ball's
    position adds about 1.2. No labels are involved, so it can be measured on any match.
    """
    kink = signals["kink"].to_numpy()
    graded = np.zeros(len(kink), dtype=bool)
    for first, last in blocks:
        graded[first:last + 1] = True
    known = kink[graded & ~np.isnan(kink)]
    return float(np.median(known)) if len(known) else np.inf


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
    p = params
    dead, team = evidence(signals, p)
    team = (team * with_player(signals, p)).astype(np.float64)
    touches = touch_frames(signals, p)
    near = signals["near_team"].to_numpy()
    if p["touch_burst"] and ball_shake(signals, blocks) < p["touch_max_shake"]:
        team[touches] += np.where(near[touches] == 0, p["touch_burst"], -p["touch_burst"])
    if p["model_scale"]:
        # The model says "the nearest team has it"; turn that into home (+) or away (-).
        team = p["model_scale"] * team_logit(signals, touches) * np.where(near == 0, 1.0, -1.0) + team

    log_p = log_probs(dead, team)
    codes = np.full(len(signals), X, dtype=np.int8)
    for first, last in blocks:
        codes[first:last + 1] = best_path(log_p[first:last + 1], p["team_switch"], p["dead_switch"])
    return codes
