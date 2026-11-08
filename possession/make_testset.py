"""Build a labelled test set from Metrica's public sample matches.

The competition gives ground truth for only five minutes of the provided match, and the rules allow
those minutes only for checking the pipeline -- never for choosing settings. So the settings have
to be chosen on labelled football from *other* matches. Metrica Sports publishes three anonymised
full matches with tracking and hand-annotated event data:

    https://github.com/metrica-sports/sample-data

    git clone --depth 1 https://github.com/metrica-sports/sample-data.git ~/metrica-sample-data

This script turns each match's events into the labels the competition scores, one per tracking
frame:

    H  home team in possession, ball alive
    A  away team in possession, ball alive
    D  dead ball
    X  not graded: before the first kick-off, between the halves, after the last event

and writes one `frame,label` CSV per match in the scorer's format. As in the competition, a row is
a tracking frame counted from 0: frame = frame_id - 1.

How events become labels
------------------------
Graded window. Each half is graded from its first KICK OFF to the end of its last event (the END
HALF event when there is one). Between the halves the labels are X, so the scorer treats the two
halves as separate runs, exactly as it does for the competition match.

Dead ball. Every restart -- any SET PIECE, or an event whose subtype contains GOAL KICK, which
Metrica records as a PASS, BALL LOST or BALL OUT rather than as a SET PIECE -- ends a stoppage. The
stoppage starts when the last action before the restart finished: the latest end frame among the
three events before it. RECOVERY and CARD events are skipped in that look-back, because Metrica
records the goalkeeper collecting the ball for a goal kick as a RECOVERY on the restart frame, and
bookings happen during the stoppage itself. The kick-off that opens a half starts the graded window
instead of ending a stoppage.

Possession. While the ball is alive, the team in possession is the team of the most recent PASS,
CARRY, RECOVERY, SET PIECE, SHOT or BALL LOST event. CHALLENGE events are not used: both teams
appear in them, and a won challenge that leads to control is followed by a RECOVERY anyway.

These are event-based labels, so they are close to the competition's convention but not identical
to it: the competition derives its labels from corrected tracking. Treat scores on this test set as
a guide for choosing settings, not as a prediction of the competition score.

    python possession/make_testset.py                       # all three matches
    python possession/make_testset.py --sample-dir DIR --out DIR
"""

import argparse
import json
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402

# Params
DEFAULT_SAMPLE_DIR = os.environ.get("METRICA_SAMPLE_DIR") or os.path.expanduser(
    "~/metrica-sample-data")
GAMES = (1, 2, 3)

# Event types whose team holds the ball when the event starts.
POSSESSION_TYPES = {"PASS", "CARRY", "RECOVERY", "SET PIECE", "SHOT", "BALL LOST"}
# Event types skipped when looking back from a restart for the end of play. See the docstring.
NOT_PLAY_TYPES = {"RECOVERY", "CARD"}
# How many events before a restart to look at for the end of play.
LOOKBACK = 3


def game_dir(sample_dir, game):
    return os.path.join(sample_dir, "data", "Sample_Game_{}".format(game))


def load_events_csv(sample_dir, game):
    """Games 1 and 2: one CSV row per event, Team is 'Home' or 'Away'."""
    path = os.path.join(game_dir(sample_dir, game), "Sample_Game_{}_RawEventsData.csv".format(game))
    raw = pd.read_csv(path)
    events = pd.DataFrame({
        "team": raw["Team"].map({"Home": "H", "Away": "A"}),
        "type": raw["Type"],
        "subtype": raw["Subtype"].fillna(""),
        "period": raw["Period"].astype(int),
        "start": raw["Start Frame"].astype(int),
        "end": raw["End Frame"].astype(int),
    })
    return _tidy(events)


def load_events_json(sample_dir, game):
    """Game 3: Metrica's JSON events. Subtypes come as a list, joined with '-' like the CSVs.

    The home team is the first team in the EPTS metadata -- the FIFA/EPTS convention, the same one
    the competition's baseline relies on.
    """
    folder = game_dir(sample_dir, game)
    meta = ET.parse(os.path.join(folder, "Sample_Game_{}_metadata.xml".format(game))).getroot()
    home_id = next(meta.iter("Team")).get("id")

    with open(os.path.join(folder, "Sample_Game_{}_events.json".format(game))) as handle:
        data = json.load(handle)["data"]

    rows = []
    for event in data:
        subtypes = event.get("subtypes") or []
        if isinstance(subtypes, dict):
            subtypes = [subtypes]
        start = event["start"]["frame"]
        end = event["end"]["frame"]
        team_id = (event.get("team") or {}).get("id")
        rows.append({
            "team": "" if team_id is None else ("H" if team_id == home_id else "A"),
            "type": event["type"]["name"],
            "subtype": "-".join(s["name"] for s in subtypes),
            "period": int(event["period"]),
            "start": int(start),
            "end": int(end) if end is not None else int(start),
        })
    return _tidy(pd.DataFrame(rows))


def _tidy(events):
    """Order events by start frame, keeping file order for ties, and repair empty end frames."""
    events = events.copy()
    events["team"] = events["team"].fillna("")
    events["end"] = np.maximum(events["start"], events["end"])  # a kick-off can carry End Frame 0
    events["order"] = np.arange(len(events))
    return events.sort_values(["start", "order"]).reset_index(drop=True)


def n_tracking_frames(sample_dir, game):
    """Number of tracking frames. Frame ids run 1..n in all three matches, so this is the last id."""
    folder = game_dir(sample_dir, game)
    if game in (1, 2):
        path = os.path.join(folder, "Sample_Game_{}_RawTrackingData_Home_Team.csv".format(game))
        frames = pd.read_csv(path, skiprows=2, usecols=[1]).iloc[:, 0]
    else:
        path = os.path.join(folder, "Sample_Game_{}_tracking.txt".format(game))
        with open(path) as handle:
            frames = pd.Series([int(line[:line.index(":")]) for line in handle])
    if frames.iloc[0] != 1 or not frames.diff().iloc[1:].eq(1).all():
        raise ValueError("game {}: tracking frame ids are not 1..n".format(game))
    return int(frames.iloc[-1])


def is_restart(events):
    return (events["type"] == "SET PIECE") | events["subtype"].str.contains("GOAL KICK")


def labels_from_events(events, n_frames):
    """One label per tracking frame, frame = frame_id - 1. See the module docstring for the rules."""
    labels = np.full(n_frames, "X", dtype="<U1")

    for period, half in events.groupby("period", sort=True):
        half = half.reset_index(drop=True)
        kickoffs = half[(half["type"] == "SET PIECE") & half["subtype"].str.contains("KICK OFF")]
        if kickoffs.empty:
            raise ValueError("period {} has no KICK OFF".format(period))
        first = int(kickoffs["start"].iloc[0])
        end_half = half[half["subtype"].str.contains("END HALF")]
        last = int(end_half["end"].max()) if len(end_half) else int(half["end"].max())

        # Possession: the team of the most recent possession event, carried forward frame by frame.
        team = np.full(last - first + 1, "", dtype="<U1")
        for start, side in zip(half["start"], half["team"].where(half["type"].isin(
                POSSESSION_TYPES), "")):
            if side and first <= start <= last:
                team[start - first] = side  # later events on the same frame overwrite earlier ones
        known = np.flatnonzero(team != "")
        carry = np.maximum.accumulate(np.where(team != "", np.arange(len(team)), -1))
        carry[carry < 0] = known[0]
        window = team[carry]

        # Dead ball: from the end of the last action before each restart up to the restart.
        restarts = half[is_restart(half) & (half["start"] > first)]
        play = half[~half["type"].isin(NOT_PLAY_TYPES)]
        for restart in restarts["start"]:
            before = play[play["start"] < restart].tail(LOOKBACK)
            if before.empty:
                continue
            stop = int(before["end"].max())
            if stop < restart:
                window[max(stop, first) - first:restart - first] = "D"

        labels[first - 1:last] = window
    return labels


def write_labels(path, labels):
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


def summarise(game, labels):
    """The figures to compare with the competition's public window, plus the scorer self-check."""
    codes = pm.label_codes(labels)
    runs = pm.valid_runs(codes)
    stats = pm.possession_stats(codes, runs)
    graded = codes != pm.X
    self_check = pm.evaluate(codes, codes)
    return {
        "game": game,
        "graded_min": round(graded.sum() / pm.FPS / 60.0, 1),
        "halves": len(runs),
        "possessions": stats["n_spells"],
        "per_min": round(stats["n_spells"] / (graded.sum() / pm.FPS / 60.0), 2),
        "mean_s": round(stats["mean_duration_s"], 1),
        "dead_%": round(100.0 * np.sum(codes == pm.D) / graded.sum(), 1),
        "home_%": round(100.0 * stats["home_share"], 1),
        "self_macroF1": round(self_check["macro_f1"], 4),
        "self_spellF1": round(self_check["spell_f1_mean"], 4),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample-dir", default=DEFAULT_SAMPLE_DIR,
                        help="clone of github.com/metrica-sports/sample-data")
    parser.add_argument("--out", help="where to write the label CSVs (default: <sample-dir>/labels)")
    args = parser.parse_args()

    out = args.out or os.path.join(args.sample_dir, "labels")
    os.makedirs(out, exist_ok=True)

    rows = []
    for game in GAMES:
        events = (load_events_csv if game in (1, 2) else load_events_json)(args.sample_dir, game)
        labels = labels_from_events(events, n_tracking_frames(args.sample_dir, game))
        path = os.path.join(out, "Sample_Game_{}_labels.csv".format(game))
        write_labels(path, labels)
        rows.append(summarise(game, labels))
        print("wrote", path)

    print()
    print(pd.DataFrame(rows).to_string(index=False))
    print("\ncompetition public window, for comparison: 3.4 possessions per minute, mean 14.2 s, "
          "19.6% dead")
