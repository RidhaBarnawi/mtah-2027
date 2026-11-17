"""Check the answer key against the tracking, which the answer key never looked at.

    ../.venv/bin/python check_answer_key.py

If the answers are right:
  - players should move more slowly when the answer is D (play stopped) than when it is H / A
  - when the answer says home (or away) has the ball and the ball is visible, the player nearest
    the ball should usually belong to that team
"""

import csv
import os

import numpy as np
from kloppy import metrica

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "sample-data", "data")
PITCH = (105.0, 68.0)
FPS = 25


def load_tracking(game):
    folder = os.path.join(DATA, "Sample_Game_%d" % game)
    if game in (1, 2):
        dataset = metrica.load_tracking_csv(
            home_data=os.path.join(folder, "Sample_Game_%d_RawTrackingData_Home_Team.csv" % game),
            away_data=os.path.join(folder, "Sample_Game_%d_RawTrackingData_Away_Team.csv" % game))
    else:
        dataset = metrica.load_tracking_epts(
            meta_data=os.path.join(folder, "Sample_Game_%d_metadata.xml" % game),
            raw_data=os.path.join(folder, "Sample_Game_%d_tracking.txt" % game))
    return dataset


def positions(dataset):
    """(frame_ids, ball[n,2], players[n,p,2] in metres, is_home[p])."""
    df = dataset.to_df()
    home_team = next(t for t in dataset.metadata.teams if str(t.ground).lower().endswith("home"))
    ids, is_home = [], []
    for team in dataset.metadata.teams:
        for player in team.players:
            if "%s_x" % player.player_id in df.columns:
                ids.append(player.player_id)
                is_home.append(team is home_team)
    scale = np.array(PITCH)
    ball = df[["ball_x", "ball_y"]].to_numpy(float) * scale
    players = np.stack([df[["%s_x" % i, "%s_y" % i]].to_numpy(float) for i in ids], axis=1) * scale
    return df["frame_id"].to_numpy(int), ball, players, np.array(is_home)


def main():
    for game in (1, 2, 3):
        labels = [row["label"] for row in csv.DictReader(
            open(os.path.join(HERE, "answers", "sample_game_%d.csv" % game)))]
        labels = np.array(labels)
        frame_ids, ball, players, is_home = positions(load_tracking(game))
        lab = labels[frame_ids - 1]          # row i of the answer file is tracking frame i + 1

        speed = np.linalg.norm(np.diff(players, axis=0), axis=2) * FPS     # m/s per player
        speed = np.nanmean(np.where(speed < 12, speed, np.nan), axis=1)    # drop tracking jumps
        alive = np.isin(lab[1:], ["H", "A"])
        dead = lab[1:] == "D"

        dist = np.linalg.norm(players - ball[:, None, :], axis=2)
        seen = ~np.isnan(ball).any(axis=1) & ~np.isnan(dist).all(axis=1)
        nearest = np.full(len(lab), -1)
        nearest[seen] = np.nanargmin(dist[seen], axis=1)
        nearest_home = np.where(nearest >= 0, is_home[nearest], False)
        poss = seen & np.isin(lab, ["H", "A"])
        agree = np.mean(nearest_home[poss] == (lab[poss] == "H"))

        print("Sample game %d" % game)
        print("    average player speed when in play   %.2f m/s" % np.nanmean(speed[alive]))
        print("    average player speed when stopped   %.2f m/s" % np.nanmean(speed[dead]))
        print("    ball visible when in play / stopped %.0f%% / %.0f%%"
              % (100 * seen[np.isin(lab, ["H", "A"])].mean(), 100 * seen[lab == "D"].mean()))
        print("    nearest player belongs to the team the answer names: %.0f%%" % (100 * agree))
        print()


if __name__ == "__main__":
    main()
