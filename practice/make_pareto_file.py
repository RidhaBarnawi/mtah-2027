"""Merge one version's 3 mistake files into a single file for a Pareto analysis in Excel.

    python3 make_pareto_file.py v1_team_robot

Reads mistakes/<name>/sample_game_1..3.csv (written by practice_exam.py) and writes
mistakes/<name>/pareto_all_matches.csv with one row per mistake and these columns:

    match       1, 2 or 3
    start, end  when the mistake happened (match clock)
    seconds     how long it lasted -- sum this column in the Pareto
    mistake     wrong team / said paused, was play / said play, was paused
    situation   what happened in the match within 3 s of the mistake, in plain words
    ball        whether the tracking saw the ball during the mistake
    detail      the original event text
"""

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RESTARTS = (("THROW", "restart: throw-in"), ("CORNER", "restart: corner"),
            ("GOAL KICK", "restart: goal kick"), ("FREE KICK", "restart: free kick"),
            ("KICK OFF", "restart: kick-off"))
EVENTS = (("SET PIECE", "restart: other"), ("BALL OUT", "ball out"), ("BALL LOST", "ball lost"),
          ("FAULT RECEIVED", "foul"), ("RECOVERY", "ball won"), ("CHALLENGE", "duel"),
          ("PASS", "pass"), ("CARRY", "dribble"), ("SHOT", "shot"), ("CARD", "card"))


def situation(text):
    """'Home RECOVERY INTERCEPTION' -> 'ball won'; restarts by type; '' -> nothing nearby."""
    if not text:
        return "nothing within 3 s"
    for key, name in RESTARTS:
        if key in text and ("SET PIECE" in text or key == "GOAL KICK"):
            return name
    for key, name in EVENTS:
        if key in text:
            return name
    return "other"


def ball(seen_percent):
    seen = int(seen_percent)
    if seen < 20:
        return "ball not seen"
    if seen < 80:
        return "ball partly seen"
    return "ball seen"


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "v1_team_robot"
    folder = os.path.join(HERE, "mistakes", name)
    rows = []
    for match in (1, 2, 3):
        with open(os.path.join(folder, "sample_game_%d.csv" % match)) as handle:
            for row in csv.DictReader(handle):
                rows.append({
                    "match": match,
                    "start": row["start"],
                    "end": row["end"],
                    "seconds": row["seconds"],
                    "mistake": row["mistake"],
                    "situation": situation(row["what happened nearby"]),
                    "ball": ball(row["ball seen %"]),
                    "detail": row["what happened nearby"],
                })
    out = os.path.join(folder, "pareto_all_matches.csv")
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %d mistakes to %s" % (len(rows), os.path.relpath(out, HERE)))
    print("situations used:", ", ".join(sorted(set(r["situation"] for r in rows))))


if __name__ == "__main__":
    main()
