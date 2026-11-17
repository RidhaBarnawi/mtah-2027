import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # the repo's top folder
sys.path.insert(0, ROOT)   # so Python can find games.py there
import games  # noqa: E402    # the organisers' file: data paths and match facts

SAMPLE_DIR = os.path.expanduser("~/metrica-sample-data")
SAMPLE_PITCH = (105.0, 68.0)  # metres: length, width


@dataclass
class Tracking:
    xy: np.ndarray    # (frames, players, 2) positions in metres, NaN = not seen
    team: np.ndarray  # (players,) 0 = home, 1 = away
    ball: np.ndarray  # (frames, 2) ball position in metres, NaN = not seen
    pitch: tuple      # (length, width) in metres


def load_sample(game, cache=True):
    """Load one of Metrica's public matches (1, 2 or 3) with positions in metres."""
    from kloppy import metrica   # only the practice matches need kloppy

    cache_path = os.path.join(SAMPLE_DIR, "cache", f"Sample_Game_{game}_tracking.npz")
    if cache and os.path.exists(cache_path):   # saved before? then just read it
        return _load(cache_path)
    stem = os.path.join(SAMPLE_DIR, "data", f"Sample_Game_{game}", f"Sample_Game_{game}")
    if game in (1, 2):   # matches 1 and 2 come as two CSV files, one per team
        dataset = metrica.load_tracking_csv(home_data=stem + "_RawTrackingData_Home_Team.csv",
                                            away_data=stem + "_RawTrackingData_Away_Team.csv",
                                            coordinates="metrica")
    else:                # match 3 is in the same format as the competition
        dataset = metrica.load_tracking_epts(meta_data=stem + "_metadata.xml",
                                             raw_data=stem + "_tracking.txt",
                                             coordinates="metrica")
    df = dataset.to_df()
    # Which row of our arrays each table row goes to: frame 1 -> row 0, frame 2 -> row 1, ...
    rows = df["frame_id"].to_numpy() - 1
    n_frames = rows.max() + 1

    # All players, home team first, plus a 0/1 team label for each
    home, away = dataset.metadata.teams
    players = list(home.players) + list(away.players)
    team = np.array([0] * len(home.players) + [1] * len(away.players))

    # Start with arrays full of NaN ("not seen"), then fill in what the table has
    xy = np.full((n_frames, len(players), 2), np.nan, dtype=np.float32)
    for j, player in enumerate(players):
        x_col = f"{player.player_id}_x"
        y_col = f"{player.player_id}_y"
        if x_col in df:                       # a substitute who never played has no column
            xy[rows, j, 0] = df[x_col]
            xy[rows, j, 1] = df[y_col]

    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)
    ball[rows] = df[["ball_x", "ball_y"]].to_numpy()

    # From fractions of the pitch (0 to 1) to metres
    scale = np.array(SAMPLE_PITCH, dtype=np.float32)
    t = Tracking(xy=xy * scale, team=team, ball=ball * scale, pitch=SAMPLE_PITCH)
    if cache:
        _save(cache_path, t)                  # keep a copy for next time
    return t

def _slot_teams(meta_path):
    """0 for each home slot, 1 for each away slot, in the order the slots appear in each line."""
    root = ET.parse(meta_path).getroot()
    team_ids = [team.get("id") for team in root.iter("Team")]   # the first team is home
    return np.array([0 if player.get("teamId") == team_ids[0] else 1
                     for player in root.iter("Player")])


def load_competition(cache=True):
    """The real match, read straight from its raw tracking file. Positions in metres."""
    meta_path, raw_path = games.require(["atd_meta", "atd_raw"])          # the two file paths
    cache_path = os.path.join(games.DATA_DIR, "cache", "tactical_tracking.npz")  # in ~/mtah-data
    if cache and os.path.exists(cache_path):
        return _load(cache_path)

    team = _slot_teams(meta_path)
    n_frames = games.GAMES[games.GAME]["n_video_frames"]                 # 199,319
    xy = np.full((n_frames, len(team), 2), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 2), np.nan, dtype=np.float32)

    with open(raw_path) as handle:
        for line in handle:                                              # one line = one frame
            parts = line.strip().split(":")
            if len(parts) != 3:                                          # skip anything that isn't a frame
                continue
            frame_id, slots, ball_part = parts
            row = int(frame_id) - 1
            numbers = np.array(slots.replace(";", ",").split(","), dtype=np.float32)
            xy[row] = numbers.reshape(len(team), -1)[:, :2]               # keep x, y; drop speed
            ball[row] = np.array(ball_part.split(",")[:2], dtype=np.float32)

    pitch = games.GAMES[games.GAME]["pitch"]                             # (105.3, 68.0) metres
    scale = np.array(pitch, dtype=np.float32)
    t = Tracking(xy=xy * scale, team=team, ball=ball * scale, pitch=pitch)
    if cache:
        _save(cache_path, t)
    return t

def _save(path, t):
    """Save a Tracking to one compressed file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)   # create the cache folder if it doesn't exist
    np.savez_compressed(path, xy=t.xy, team=t.team, ball=t.ball, pitch=np.array(t.pitch))


def _load(path):
    """Read a Tracking back from the file _save wrote."""
    saved = np.load(path)
    return Tracking(xy=saved["xy"], team=saved["team"], ball=saved["ball"],
                    pitch=tuple(float(v) for v in saved["pitch"]))

# This part runs only when you run this file directly, not when another file uses it
if __name__ == "__main__":
    t = load_competition()
    graded = np.r_[11185:90652, 110808:190198]            # the frames that get scored
    seen = ~np.isnan(t.ball).any(axis=1)                   # True where the ball is seen
    players = (~np.isnan(t.xy[:, :, 0])).sum(axis=1)       # players seen on each frame
    print(t.xy.shape)
    print(round(100 * seen.mean(), 1), round(100 * seen[graded].mean(), 1), round(players[graded].mean(), 1))