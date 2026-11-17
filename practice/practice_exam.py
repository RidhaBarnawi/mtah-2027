"""Grade a possession program on the practice exam, with the organisers' own scorer.

Run it from this folder with the repository's Python:

    ../.venv/bin/python practice_exam.py
    ../.venv/bin/python practice_exam.py our_program.py --name "what this version is"

With no program it grades example_program.py (the organisers' baseline).

YOUR PROGRAM is a Python file with one function:

    def predict(atd_meta, atd_raw, n_frames, halves):
        return labels    # n_frames labels, each "H", "A", "D" or "X"

    atd_meta, atd_raw  paths to the tracking pair, in the same format as the competition match
    n_frames           how many labels to return; label i is for tracking frame_id i + 1
    halves             [(first, last), ...] the graded frames of each half, both included

Run the same function on the competition match to make the real submission:

    predict(atd_meta, atd_raw, 199319, [(11185, 90651), (110808, 190197)])

so the practice exam and the real submission go through exactly the same code. The exam has no
video, Smart Tagging or homography, so a program that uses them has to cope without them here.

WHAT YOU GET
  - spellF1 (the ranking score) and macroF1 (the tie-break) for each sample match, and the average
  - a new row in scoreboard.csv, so every version you try is recorded
  - mistakes/<name>/sample_game_<n>.csv: every stretch the program got wrong, what kind of mistake
    it was and what was happening in the match -- open it in Excel for a Pareto chart

Average spellF1 differences smaller than about 0.02 between two versions are luck, not progress.
"""

import argparse
import bisect
import csv
import datetime
import importlib.util
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.environ.get("MTAH_REPO") or os.path.dirname(HERE)      # this folder sits inside the repository
sys.path.insert(0, os.path.join(REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402

from build_answer_key import MATCHES  # noqa: E402

EXAM = os.path.join(HERE, "exam")
ANSWERS = os.path.join(HERE, "answers")
MISTAKES = os.path.join(HERE, "mistakes")
SCOREBOARD = os.path.join(HERE, "scoreboard.csv")
GAMES = (1, 2, 3)
FPS = 25
NEARBY_S = 3.0      # an event this close to the start of a mistake is listed beside it
KINDS = ("wrong team", "said paused, was play", "said play, was paused", "no answer (X)")


# ---------------------------------------------------------------------------------------------
# Running a program
# ---------------------------------------------------------------------------------------------

def load_program(path):
    spec = importlib.util.spec_from_file_location("program_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "predict"):
        raise SystemExit("%s has no predict() function" % path)
    return module


def exam_match(game):
    folder = os.path.join(EXAM, "sample_game_%d" % game)
    with open(os.path.join(folder, "info.json")) as handle:
        info = json.load(handle)
    return {
        "atd_meta": os.path.join(folder, "sample_game_%d_atd.xml" % game),
        "atd_raw": os.path.join(folder, "sample_game_%d_atd.txt" % game),
        "n_frames": info["n_frames"],
        "halves": [tuple(half) for half in info["halves"]],
    }


def clean_labels(labels, n_frames):
    labels = np.array([str(v).strip().upper() for v in labels], dtype=object)
    if len(labels) != n_frames:
        raise SystemExit("predict() returned %d labels, expected %d" % (len(labels), n_frames))
    wrong = sorted(set(labels) - set(pm.LABELS))
    if wrong:
        raise SystemExit("predict() returned labels other than H/A/D/X: %s" % wrong[:5])
    return labels


def read_answers(game):
    with open(os.path.join(ANSWERS, "sample_game_%d.csv" % game)) as handle:
        return np.array([row["label"] for row in csv.DictReader(handle)], dtype=object)


# ---------------------------------------------------------------------------------------------
# The list of mistakes
# ---------------------------------------------------------------------------------------------

def ball_seen(raw_path):
    """Per frame: is there a ball position in the exam tracking?"""
    with open(raw_path) as handle:
        return np.array([line.rsplit(":", 1)[1][0] != "N" for line in handle])


def clock(row):
    seconds = row / float(FPS)
    return "%d:%04.1f" % (seconds // 60, seconds % 60)


def kind_of(correct, yours):
    if yours == "X":
        return "no answer (X)"
    if correct == "D":
        return "said play, was paused"
    if yours == "D":
        return "said paused, was play"
    return "wrong team"


def event_text(event):
    team = {"H": "Home", "A": "Away"}.get(event["team"], "")
    return " ".join(part for part in (team, event["type"], event["sub"]) if part)


def nearby_event(events, starts, row):
    """The event closest to this frame, if within NEARBY_S seconds."""
    frame = row + 1                              # events count tracking frames from 1
    i = bisect.bisect_left(starts, frame)
    best = None
    for j in (i - 1, i):
        if 0 <= j < len(events):
            gap = abs(starts[j] - frame)
            if gap <= NEARBY_S * FPS and (best is None or gap < best[0]):
                best = (gap, events[j])
    return event_text(best[1]) if best else ""


def mistakes(game, gt, pred, seen):
    """[row dict] for every stretch where the program's label is not the answer."""
    events, _ = MATCHES[game](game)
    events = sorted(events, key=lambda e: e["start"])
    starts = [e["start"] for e in events]

    wrong = (gt != "X") & (pred != gt)
    key = np.where(wrong, gt.astype(str) + pred.astype(str), "")
    bounds = np.concatenate(([0], np.flatnonzero(key[1:] != key[:-1]) + 1, [len(key)]))
    out = []
    for start, stop in zip(bounds[:-1], bounds[1:]):
        if not key[start]:
            continue
        out.append({
            "start": clock(start),
            "end": clock(stop),
            "seconds": round((stop - start) / float(FPS), 2),
            "correct": gt[start],
            "yours": pred[start],
            "mistake": kind_of(gt[start], pred[start]),
            "ball seen %": int(round(100 * seen[start:stop].mean())),
            "what happened nearby": nearby_event(events, starts, start),
            "start frame": int(start),
        })
    return out


def write_csv(path, rows, fields):
    folder = os.path.dirname(path)
    if not os.path.isdir(folder):
        os.makedirs(folder)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------------------------

def grade(program, name, save=True):
    """Score one program on every exam match. Returns {game: result} and prints as it goes."""
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_") or "program"
    results = {}
    print("%-14s %8s %8s %24s %8s" % ("", "spellF1", "macroF1", "possessions yours/real", "time"))
    for game in GAMES:
        match = exam_match(game)
        began = time.time()
        pred = clean_labels(program.predict(match["atd_meta"], match["atd_raw"],
                                            match["n_frames"], match["halves"]),
                            match["n_frames"])
        gt = read_answers(game)
        result = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred))
        rows = mistakes(game, gt, pred, ball_seen(match["atd_raw"]))
        result["mistake_seconds"] = {kind: sum(r["seconds"] for r in rows if r["mistake"] == kind)
                                     for kind in KINDS}
        results[game] = result
        if save:
            write_csv(os.path.join(MISTAKES, safe, "sample_game_%d.csv" % game), rows,
                      list(rows[0].keys()) if rows else ["start"])
        print("sample game %d  %8.4f %8.4f %16d / %-6d %6.0fs"
              % (game, result["spell_f1_mean"], result["macro_f1"], result["n_pred_spells"],
                 result["n_gt_spells"], time.time() - began))

    spell = float(np.mean([r["spell_f1_mean"] for r in results.values()]))
    macro = float(np.mean([r["macro_f1"] for r in results.values()]))
    print("%-14s %8.4f %8.4f" % ("average", spell, macro))

    print("\nwhere the mistakes are (seconds, all 3 matches)")
    total = sum(sum(r["mistake_seconds"].values()) for r in results.values()) or 1.0
    for kind in KINDS:
        seconds = sum(r["mistake_seconds"][kind] for r in results.values())
        if seconds:
            print("    %-24s %7.0f s  %3.0f%%" % (kind, seconds, 100.0 * seconds / total))

    if save:
        new = not os.path.isfile(SCOREBOARD)
        with open(SCOREBOARD, "a", newline="") as handle:
            writer = csv.writer(handle)
            if new:
                writer.writerow(["when", "name", "spellF1 game 1", "spellF1 game 2",
                                 "spellF1 game 3", "spellF1 average", "macroF1 average"])
            writer.writerow([datetime.datetime.now().strftime("%Y-%m-%d %H:%M"), name]
                            + ["%.4f" % results[g]["spell_f1_mean"] for g in GAMES]
                            + ["%.4f" % spell, "%.4f" % macro])
        print("\nadded to scoreboard.csv; mistakes in mistakes/%s/" % safe)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("program", nargs="?", default=os.path.join(HERE, "example_program.py"),
                        help="a .py file with predict() (default: example_program.py)")
    parser.add_argument("--name", help="what this version is, for the scoreboard")
    args = parser.parse_args()

    name = args.name or os.path.splitext(os.path.basename(args.program))[0]
    print("Practice exam: %s\n" % name)
    grade(load_program(args.program), name)


if __name__ == "__main__":
    main()
