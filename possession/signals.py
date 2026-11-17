"""Per-frame signals: one row per frame with the simple facts the possession model uses."""

import warnings
import numpy as np
import pandas as pd

FPS = 25                # frames per second
MAX_BALL_SPEED = 40.0   # m/s; anything faster is the tracker jumping, not a real kick
MAX_PLAYER_SPEED = 12.0  # m/s; faster "running" is the tracker swapping two players
SMOOTH = 5               # number of frames to smooth the ball's speed over
SPEED_SPAN = 12          # frames on each side: the players' speed is measured over about 1 second
VELOCITY_SPAN = 3        # frames on each side for the ball's and the nearest player's velocity
KINK_SPAN = 4            # frames: the ball's velocity over the 4 before against the 4 after


def velocity(xy, span=VELOCITY_SPAN):
    """Velocity in m/s from the positions `span` frames before and after. NaN where either is missing."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[span:-span] = (xy[2 * span:] - xy[:-2 * span]) / (2 * span / float(FPS))
    return v


def compute_signals(t):
    length, width = t.pitch
    x, y = t.ball[:, 0], t.ball[:, 1]
    n = len(t.ball)
    rows = np.arange(n)

    # 1. Is the ball seen? (False where its position is NaN)
    ball_seen = ~np.isnan(t.ball).any(axis=1)

    # 2. Is the ball outside the pitch? (only counts when it's seen)
    ball_off = ball_seen & ((x < 0) | (x > length) | (y < 0) | (y > width))

    # 3. Ball speed: distance moved since the previous frame, times 25 frames per second
    moved = np.linalg.norm(np.diff(t.ball, axis=0), axis=1)   # metres between frames
    ball_speed = np.concatenate(([np.nan], moved * FPS))       # the first frame has no previous one
    ball_speed[ball_speed > MAX_BALL_SPEED] = np.nan           # throw away impossible jumps

    # 4. How many players are seen on each frame
    n_players = (~np.isnan(t.xy[:, :, 0])).sum(axis=1)

    # 5. Distance from the ball to every player, then the nearest home and nearest away player
    distance = np.linalg.norm(t.xy - t.ball[:, None, :], axis=2)    # (frames, players), metres
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)             # frames where nobody is seen just give NaN
        d_home = np.nanmin(np.where(t.team == 0, distance, np.nan), axis=1)
        d_away = np.nanmin(np.where(t.team == 1, distance, np.nan), axis=1)

        # 6. Players' average speed, from where each player was about half a second before and
        #    after. Speed between two consecutive frames turns a few centimetres of tracking shake
        #    into metres per second, and then no stoppage looks like players walking.
        speed = np.full(t.xy.shape[:2], np.nan, dtype=np.float32)
        travelled = np.linalg.norm(t.xy[2 * SPEED_SPAN:] - t.xy[:n - 2 * SPEED_SPAN], axis=2)
        speed[SPEED_SPAN:n - SPEED_SPAN] = travelled / (2 * SPEED_SPAN / float(FPS))
        speed[speed > MAX_PLAYER_SPEED] = np.nan                    # the tracker swapping players, not running
        players_speed = np.nanmean(speed, axis=1)

    # 7. The one player nearest the ball: his team (0 home, 1 away), how far he is, and how fast
    #    the ball moves relative to him. A ball that moves with a player is his; one flying past
    #    him is not.
    reach = np.where(np.isnan(distance), np.inf, distance)          # not seen = infinitely far
    i_home = np.argmin(np.where(t.team == 0, reach, np.inf), axis=1)
    i_away = np.argmin(np.where(t.team == 1, reach, np.inf), axis=1)
    near_team = np.where(reach[rows, i_home] <= reach[rows, i_away], 0, 1)
    d_near = np.minimum(reach[rows, i_home], reach[rows, i_away])
    i_near = np.where(near_team == 0, i_home, i_away)
    ball_velocity = velocity(t.ball)
    slot_velocity = velocity(t.xy.reshape(n, -1)).reshape(t.xy.shape)
    rel_speed = np.linalg.norm(ball_velocity - slot_velocity[rows, i_near], axis=1)

    # 8. Kink: how much the ball's velocity changes, from the 4 frames before to the 4 frames
    #    after. A touch shows as a sharp kink in the ball's path.
    k = KINK_SPAN
    kink = np.full(n, np.nan, dtype=np.float32)
    after, before = t.ball[2 * k:] - t.ball[k:-k], t.ball[k:-k] - t.ball[:-2 * k]
    kink[k:-k] = np.linalg.norm(after - before, axis=1) * FPS / k

    # 9. More about the moment, for the learned "who has the ball" clue (possession/team_model.py):
    #    the other team's nearest player, the ball's own speed, the strongest kink in the second
    #    before and the second after, and how many players of each team are within 3 m.
    d_other = np.maximum(reach[rows, i_home], reach[rows, i_away])
    i_other = np.where(near_team == 0, i_away, i_home)
    rel_other = np.linalg.norm(ball_velocity - slot_velocity[rows, i_other], axis=1)
    ball_pace = np.linalg.norm(ball_velocity, axis=1)
    kink_filled = pd.Series(np.nan_to_num(kink, nan=0.0))
    kink_before = kink_filled.rolling(FPS, min_periods=1).max().to_numpy()
    kink_after = kink_filled[::-1].rolling(FPS, min_periods=1).max().to_numpy()[::-1]
    within = reach <= 3.0
    c_home, c_away = (within & (t.team == 0)).sum(axis=1), (within & (t.team == 1)).sum(axis=1)
    c_near = np.minimum(np.where(near_team == 0, c_home, c_away), 3)
    c_other = np.minimum(np.where(near_team == 0, c_away, c_home), 3)

    s = pd.DataFrame({
        "ball_seen": ball_seen,
        "ball_off": ball_off,
        "ball_speed": ball_speed,
        "d_home": d_home,
        "d_away": d_away,
        "n_players": n_players,
        "players_speed": players_speed,
        "near_team": near_team,
        "d_near": d_near,
        "rel_speed": rel_speed,
        "kink": kink,
        "d_other": d_other,
        "rel_other": rel_other,
        "ball_pace": ball_pace,
        "kink_before": kink_before,
        "kink_after": kink_after,
        "c_near": c_near,
        "c_other": c_other,
    })

    # 10. Smooth the ball's speed: each frame takes the middle value of the 5 frames around it
    s["ball_speed"] = s["ball_speed"].rolling(SMOOTH, center=True, min_periods=1).median()
    s.loc[~ball_seen, "ball_speed"] = np.nan   # smoothing must not invent a speed for an unseen ball
    return s


if __name__ == "__main__":
    import os
    from tracking import load_sample
    s = compute_signals(load_sample(3))
    print(s.iloc[[400, 401]].round(2))

    labels = pd.read_csv(os.path.expanduser("~/metrica-sample-data/labels/Sample_Game_3_labels.csv"))["label"].to_numpy()
    alive = np.isin(labels, ["H", "A"])          # frames where the ball is in play
    seen = s["ball_seen"].to_numpy()
    nearest = np.where(s["d_home"] < s["d_away"], "H", "A")
    print("nearest team = label:", round(100 * (nearest[alive & seen] == labels[alive & seen]).mean(), 1), "%")
    print("players' speed, dead ball:", round(s["players_speed"][labels == "D"].median(), 2), "m/s")
    print("players' speed, live ball:", round(s["players_speed"][alive].median(), 2), "m/s")
