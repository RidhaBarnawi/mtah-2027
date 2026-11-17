"""Build the answer key for Metrica's three free sample matches.

    python3 build_answer_key.py

For every tracking frame of each sample match, decide who has the ball:

    H  home team has the ball, ball in play
    A  away team has the ball, ball in play
    D  dead ball (play is stopped)
    X  not graded (before kickoff, after the half-time / full-time whistle)

and write answers/sample_game_<n>.csv in the same format as the official answer file:
`frame,label`, one row per tracking frame, frame counted from 0 (row 0 = tracking frame 1).

The answers come from each match's event list, which people recorded while watching that match.
The competition rules allow human-made labels for OTHER games, so these are a fair mock exam and
fair training data. Nothing here uses the competition match.

How the event list becomes an answer, in short:
  - a restart (kick-off, throw-in, free kick, corner, goal kick) starts play; the restarting
    team has the ball
  - a pass, carry, shot or recovery means that team has the ball (a recovery is how the ball
    changes team)
  - ball out, a foul, a goal or a shot that goes out stops play until the next restart
  - a restart always means play had stopped, even when no event says so (a foul recorded as a
    duel, an offside, a handball): play then stopped when the last action before it ended
  - duels, 'ball lost' and cards change nothing on their own: the next on-ball event decides
"""

import csv
import json
import os
import xml.etree.ElementTree as ET
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "sample-data", "data")
OUT = os.path.join(HERE, "answers")
FPS = 25

# Metrica's own event names (see sample-data/documentation/events-definitions.pdf).
ON_BALL = {"PASS", "CARRY", "SHOT", "RECOVERY", "SET PIECE"}   # the team doing it has the ball
RESUMES_PLAY = {"PASS", "CARRY", "SHOT", "SET PIECE"}          # can end a stoppage
# Finding when play stopped before a restart that no event announces: the latest end among the
# LOOKBACK events before it. Recoveries and cards are skipped, because a goalkeeper collecting the
# ball for a goal kick is recorded as a RECOVERY and bookings happen during the stoppage itself.
LOOKBACK = 3
NOT_PLAY = {"RECOVERY", "CARD"}


# ---------------------------------------------------------------------------------------------
# Reading the three matches into one common shape
# ---------------------------------------------------------------------------------------------

def event(team, kind, sub, period, start, end, order):
    """One event in the common shape. team is 'H', 'A' or None; frames are tracking frames."""
    return {"team": team, "type": kind.upper(), "sub": (sub or "").upper(), "period": period,
            "start": start, "end": end if end and end >= start else start, "order": order}


def load_csv_match(game):
    """Sample games 1 and 2: events CSV + tracking CSV (frames numbered from 1)."""
    folder = os.path.join(DATA, "Sample_Game_%d" % game)
    events = []
    with open(os.path.join(folder, "Sample_Game_%d_RawEventsData.csv" % game)) as handle:
        for order, row in enumerate(csv.DictReader(handle)):
            team = {"Home": "H", "Away": "A"}.get(row["Team"])
            events.append(event(team, row["Type"], row["Subtype"], int(row["Period"]),
                                int(row["Start Frame"]), int(row["End Frame"]), order))

    periods = {}
    with open(os.path.join(folder, "Sample_Game_%d_RawTrackingData_Home_Team.csv" % game)) as h:
        for row in list(csv.reader(h))[3:]:
            period, frame = int(row[0]), int(row[1])
            first, last = periods.get(period, (frame, frame))
            periods[period] = (min(first, frame), max(last, frame))
    return events, periods


def load_json_match(game):
    """Sample game 3: events JSON + EPTS metadata, whose half boundaries are tracking frames."""
    folder = os.path.join(DATA, "Sample_Game_%d" % game)
    meta = ET.parse(os.path.join(folder, "Sample_Game_%d_metadata.xml" % game)).getroot()
    home_id = meta.find(".//Score").get("idLocalTeam")
    params = {p.findtext("Name"): p.findtext("Value") for p in meta.iter("ProviderParameter")}
    periods = {1: (int(params["first_half_start"]), int(params["first_half_end"])),
               2: (int(params["second_half_start"]), int(params["second_half_end"]))}

    with open(os.path.join(folder, "Sample_Game_%d_events.json" % game)) as handle:
        raw = json.load(handle)["data"]
    events = []
    for order, row in enumerate(raw):
        subs = row.get("subtypes")
        if isinstance(subs, dict):
            subs = [subs]
        sub = " / ".join(s["name"] for s in subs or [])
        team = None
        if row.get("team"):
            team = "H" if row["team"]["id"] == home_id else "A"
        events.append(event(team, row["type"]["name"], sub, row["period"],
                            row["start"]["frame"], row["end"]["frame"], order))
    return events, periods


MATCHES = {1: load_csv_match, 2: load_csv_match, 3: load_json_match}


# ---------------------------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------------------------

def stops_play_at(e):
    """The frame at which this event stops play, or None if it does not."""
    if e["type"] == "BALL OUT":
        return e["end"]                     # when the ball crosses the line
    if e["type"] == "FAULT RECEIVED":
        return e["start"]                   # the foul
    if e["type"] == "SHOT":
        goal = "GOAL" in e["sub"] and "GOAL KICK" not in e["sub"]
        if goal or "OUT" in e["sub"]:
            return e["end"]                 # ball in the net, or out for a goal kick / corner
    return None


def is_restart(e):
    return e["type"] == "SET PIECE" or (e["type"] == "PASS" and "GOAL KICK" in e["sub"])


def last_action_end(events, index):
    """The frame the last action before events[index] ended, or None if there is none."""
    before = [e for e in events[:index] if e["type"] not in NOT_PLAY][-LOOKBACK:]
    return max(e["end"] for e in before) if before else None


def label_half(events, notes):
    """(first_frame, end_frame, changes) for one half.

    changes is a list of (frame, label): from that tracking frame on, the answer is label.
    first_frame is the kick-off; end_frame is the first frame after the half ends (the whistle).
    """
    kickoffs = [e for e in events if e["type"] == "SET PIECE" and "KICK OFF" in e["sub"]]
    start_event = kickoffs[0] if kickoffs else next(e for e in events if e["type"] in ON_BALL)
    whistle = [e for e in events if "END HALF" in e["sub"] and e["start"] >= start_event["start"]]
    end_frame = whistle[0]["start"] if whistle else max(e["end"] for e in events) + 1
    if not whistle:
        notes["no END HALF event, used last event"] += 1

    alive, team = True, start_event["team"]
    changes = [(start_event["start"], team)]
    # A stoppage starts when the ball actually crosses the line (or at the foul), which can be
    # a second or more after the event that announces it. Events recorded before that moment
    # (a last touch, a carry towards the line) still happen in play, so the stop waits here.
    pending_stop = None
    resumed_at = start_event["start"]
    for index, e in enumerate(events):
        if e is start_event or e["start"] < start_event["start"] or e["start"] >= end_frame:
            continue
        if pending_stop is not None and e["start"] >= pending_stop:
            alive, pending_stop = False, None
        if alive and is_restart(e) and e["team"]:
            # A restart while play is supposedly live: play had stopped and no event said so.
            if pending_stop is not None:
                notes["restart before its scheduled stop"] += 1
            else:
                stop = last_action_end(events, index)
                if stop is not None and resumed_at < stop < e["start"]:
                    changes = [change for change in changes if change[0] < stop]
                    changes.append((stop, "D"))
                    alive = False
                    notes["stoppage found from its restart"] += 1
                else:
                    notes["restart with no room for a stoppage"] += 1
        if alive:
            if e["type"] in ON_BALL and e["team"] and e["team"] != team:
                team = e["team"]
                changes.append((e["start"], team))
            stop = stops_play_at(e)
            if stop is not None:
                stop = min(stop, end_frame - 1)
                if pending_stop is None or stop < pending_stop:
                    if pending_stop is not None:
                        changes.remove((pending_stop, "D"))
                    pending_stop = stop
                    changes.append((stop, "D"))
        elif e["type"] in RESUMES_PLAY and e["team"]:
            if not is_restart(e):
                notes["play resumed without a tagged restart"] += 1
            alive, team = True, e["team"]
            resumed_at = e["start"]
            changes.append((e["start"], team))
        else:
            notes["ignored while play was stopped: " + e["type"]] += 1

    out_of_order = sum(1 for a, b in zip(changes, changes[1:]) if b[0] < a[0])
    if out_of_order:
        notes["changes out of time order (sorted)"] += out_of_order
    changes.sort(key=lambda change: change[0])
    return start_event["start"], end_frame, changes


def build_labels(events, periods):
    """One label per tracking frame; index i is tracking frame i + 1, exactly like the official
    file, where row i is video frame i and the tracking counts from 1."""
    n_frames = max(last for _, last in periods.values())
    labels = ["X"] * n_frames
    notes = Counter()
    for period, (first, last) in sorted(periods.items()):
        half = sorted((e for e in events if e["period"] == period),
                      key=lambda e: (e["start"], e["order"]))
        kickoff, end_frame, changes = label_half(half, notes)
        end_frame = min(end_frame, last + 1)
        bounds = [frame for frame, _ in changes[1:]] + [end_frame]
        for (frame, label), stop in zip(changes, bounds):
            for f in range(max(frame, kickoff), stop):
                labels[f - 1] = label
    return labels, notes


# ---------------------------------------------------------------------------------------------
# Checking the result looks like the official answers
# ---------------------------------------------------------------------------------------------

def style(labels):
    """Possession and stoppage statistics of a label list (X splits halves, like the grader)."""
    runs, prev = [], None
    for label in labels:
        if runs and label == prev:
            runs[-1][1] += 1
        else:
            runs.append([label, 1])
        prev = label
    graded = sum(n for label, n in runs if label != "X")
    poss = [n for label, n in runs if label in ("H", "A")]
    dead = [n for label, n in runs if label == "D"]
    minutes = graded / FPS / 60.0
    return {
        "graded minutes": round(minutes, 1),
        "possessions": len(poss),
        "possessions per minute": round(len(poss) / minutes, 2),
        "mean possession (s)": round(sum(poss) / len(poss) / FPS, 1),
        "stoppages": len(dead),
        "mean stoppage (s)": round(sum(dead) / len(dead) / FPS, 1) if dead else 0,
        "share of time stopped": "%.0f%%" % (100.0 * sum(dead) / graded),
        "home share of possession": "%.0f%%" % (100.0 * labels.count("H")
                                                 / (labels.count("H") + labels.count("A"))),
    }


def write_answers(path, labels):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("%d,%s\n" % (frame, label) for frame, label in enumerate(labels))


def main():
    for game, loader in sorted(MATCHES.items()):
        events, periods = loader(game)
        labels, notes = build_labels(events, periods)
        path = os.path.join(OUT, "sample_game_%d.csv" % game)
        write_answers(path, labels)
        print("Sample game %d: %d frames -> %s" % (game, len(labels), os.path.relpath(path, HERE)))
        for key, value in style(labels).items():
            print("    %-26s %s" % (key, value))
        for key, value in sorted(notes.items()):
            print("    note: %-50s %d" % (key, value))
        print()


if __name__ == "__main__":
    main()
