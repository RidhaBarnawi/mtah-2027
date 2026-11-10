"""Per-frame signals: one row per frame with the simple facts the possession model uses."""

import warnings
import numpy as np
import pandas as pd

FPS = 25                # frames per second
MAX_BALL_SPEED = 40.0   # m/s; anything faster is the tracker jumping, not a real kick
MAX_PLAYER_SPEED = 12.0  # m/s; faster "running" is the tracker swapping two players
SMOOTH = 5               # number of frames to smooth speeds over

def compute_signals(t):
    length, width = t.pitch
    x, y = t.ball[:, 0], t.ball[:, 1]

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

        # 6. Players' average speed: how far each player moved since the previous frame, times 25
        steps = np.linalg.norm(np.diff(t.xy, axis=0), axis=2) * FPS  # (frames - 1, players)
        steps[steps > MAX_PLAYER_SPEED] = np.nan                    # the tracker swapping players, not running
        players_speed = np.concatenate(([np.nan], np.nanmean(steps, axis=1)))

    s = pd.DataFrame({
        "ball_seen": ball_seen,
        "ball_off": ball_off,
        "ball_speed": ball_speed,
        "d_home": d_home,
        "d_away": d_away,
        "n_players": n_players,
        "players_speed": players_speed,
    })

    # 7. Smooth the two speeds: each frame takes the middle value of the 5 frames around it
    for column in ("ball_speed", "players_speed"):
        s[column] = s[column].rolling(SMOOTH, center=True, min_periods=1).median()
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