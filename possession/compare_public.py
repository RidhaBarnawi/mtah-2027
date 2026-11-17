"""Compare a submission with the organisers' public 5 minutes, and write the comparison as files.

    python possession/compare_public.py SUBMISSION.csv --out DIR
    python possession/compare_public.py SUBMISSION.csv --out DIR --name "what this version is"

Writes two files into DIR and prints the scores:

    compare_5min.html   open it in a browser: the scores, a timeline of the organisers' answers
                        against ours, and a table of every stretch where they differ
    compare_5min.csv    every stretch, one row each, for Excel

The scores come from the organisers' own metric code (evaluation/possession_metrics.py in the
challenge repository), so they match run_evaluation.py.

What this is for: checking that the pipeline is right -- frames aligned, home and away the right
way round, labels meaning what we think. The rules do not allow choosing settings, or choosing
between versions, with these 5 minutes, and 17 possessions are too few to rank anything anyway.

The two files are made from the match data. Keep them on this computer.
"""

import argparse
import csv
import html
import os
import sys

import numpy as np

# Params
FPS = 25
ROW_SECONDS = 60          # one timeline row per minute
PLOT_LEFT, PLOT_RIGHT = 84.0, 996.0
LANES = (("Answer", 6, 16), ("Ours", 28, 16), ("Differs", 50, 5))   # name, y, height
MEANING = {"H": "Barcelona has the ball", "A": "Athletic has the ball", "D": "play stopped",
           "X": "no answer"}

STYLE = """
.viz-root { color-scheme: light; --surface: #fcfcfb; --page: #f9f9f7; --ink: #0b0b0b;
  --ink-2: #52514e; --muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7;
  --home: #2a78d6; --away: #eb6834; --wrong: #d03b3b; }
@media (prefers-color-scheme: dark) {
  .viz-root { color-scheme: dark; --surface: #1a1a19; --page: #0d0d0d; --ink: #f0efec;
    --ink-2: #c3c2b7; --muted: #898781; --grid: #2c2c2a; --axis: #383835;
    --home: #3987e5; --away: #d95926; --wrong: #d03b3b; }
}
html { background: var(--page); }
body.viz-root { margin: 0 auto; padding: 24px 16px 48px; max-width: 1040px; background: var(--page);
  color: var(--ink); font: 15px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
h1 { font-size: 22px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 32px 0 8px; }
p { margin: 0 0 8px; color: var(--ink-2); }
.card { background: var(--surface); border-radius: 10px; padding: 16px; margin-top: 12px; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 12px; }
.stat b { display: block; font-size: 26px; font-weight: 600; color: var(--ink); }
.stat span { color: var(--ink-2); font-size: 13px; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 20px; margin: 4px 0 12px; color: var(--ink-2); font-size: 13px; }
.legend i { display: inline-block; width: 22px; height: 10px; border-radius: 3px; margin-right: 6px; vertical-align: middle; }
.row-title { font-size: 13px; color: var(--ink-2); margin: 14px 0 2px; }
svg { display: block; width: 100%; height: auto; }
.track { fill: var(--grid); }
.home { fill: var(--home); } .away { fill: var(--away); } .wrong { fill: var(--wrong); }
.lane { fill: var(--ink-2); font-size: 11px; }
.tick { fill: var(--muted); font-size: 10px; }
.tickline { stroke: var(--axis); stroke-width: 1; }
table { border-collapse: collapse; width: 100%; font-size: 13px; font-variant-numeric: tabular-nums; }
th, td { text-align: left; padding: 5px 10px 5px 0; border-bottom: 1px solid var(--grid); }
th { color: var(--ink-2); font-weight: 600; }
.scroll { overflow-x: auto; }
.note { font-size: 13px; color: var(--ink-2); }
"""


def clock(frames):
    """Frames since the start of the public window -> 'm:ss.s'."""
    seconds = frames / float(FPS)
    return "%d:%04.1f" % (seconds // 60, seconds % 60)


def video_time(frame):
    """Video frame -> 'mm:ss' in the tactical video."""
    seconds = int(frame // FPS)
    return "%02d:%02d" % (seconds // 60, seconds % 60)


def runs(values, first, last):
    """[(start, stop, value)] constant stretches of values[first..last]; stop is exclusive."""
    part = values[first:last + 1]
    change = np.flatnonzero(part[1:] != part[:-1]) + 1
    starts = np.concatenate(([0], change))
    stops = np.concatenate((change, [len(part)]))
    return [(first + int(a), first + int(b), part[int(a)]) for a, b in zip(starts, stops)]


def kind_of(answer, ours):
    if ours == "X":
        return "we gave no answer"
    if answer == "D":
        return "we said play, it was stopped"
    if ours == "D":
        return "we said stopped, it was play"
    return "wrong team"


def lane_marks(lane_runs, row_first, row_last, y, height, who, window_first):
    """SVG rects for one lane of one row. Stoppages draw nothing: the grey track shows through."""
    per_frame = (PLOT_RIGHT - PLOT_LEFT) / (ROW_SECONDS * FPS)
    out = []
    for start, stop, label in lane_runs:
        a, b = max(start, row_first), min(stop, row_last + 1)
        if a >= b or label not in ("H", "A", "W"):
            continue
        x = PLOT_LEFT + (a - row_first) * per_frame
        width = (b - a) * per_frame
        if width > 4:                      # a 2px gap in the surface colour between touching marks
            x, width = x + 1, width - 2
        css = {"H": "home", "A": "away", "W": "wrong"}[label]
        if label == "W":
            title = "Ours differs from the answer"
        else:
            title = "%s: %s" % (who, MEANING[label])
        title += ", %s to %s (%.1f s)" % (clock(start - window_first), clock(stop - window_first),
                                          (stop - start) / float(FPS))
        out.append('<rect class="%s" x="%.1f" y="%d" width="%.1f" height="%d" rx="%d"><title>%s</title></rect>'
                   % (css, x, y, max(width, 1.0), height, 3 if width >= 8 else 1, html.escape(title)))
    return out


def timeline(answer_runs, ours_runs, wrong_runs, first, last):
    """One small SVG per minute: the answer lane, our lane, and a thin lane marking differences."""
    per_second = (PLOT_RIGHT - PLOT_LEFT) / ROW_SECONDS
    parts = []
    row_first = first
    while row_first <= last:
        row_last = min(row_first + ROW_SECONDS * FPS - 1, last)
        parts.append('<div class="row-title">Match clock %s to %s &middot; video %s to %s</div>'
                     % (clock(row_first - first)[:-2], clock(row_last + 1 - first)[:-2],
                        video_time(row_first), video_time(row_last + 1)))
        svg = ['<svg viewBox="0 0 1000 78" role="img" aria-label="Answer and our labels, %s to %s">'
               % (clock(row_first - first), clock(row_last + 1 - first))]
        for (name, y, height), lane_runs, who in zip(LANES, (answer_runs, ours_runs, wrong_runs),
                                                    ("Answer", "Ours", "")):
            svg.append('<text class="lane" x="%d" y="%d" text-anchor="end">%s</text>'
                       % (PLOT_LEFT - 8, y + height / 2 + 4, name))
            svg.append('<rect class="track" x="%.1f" y="%d" width="%.1f" height="%d" rx="2"/>'
                       % (PLOT_LEFT, y, PLOT_RIGHT - PLOT_LEFT, height))
            svg.extend(lane_marks(lane_runs, row_first, row_last, y, height, who, first))
        for second in range(0, ROW_SECONDS + 1, 10):
            x = PLOT_LEFT + second * per_second
            svg.append('<line class="tickline" x1="%.1f" x2="%.1f" y1="58" y2="62"/>' % (x, x))
            svg.append('<text class="tick" x="%.1f" y="74" text-anchor="%s">%s</text>'
                       % (x, "end" if second == ROW_SECONDS else "middle",
                          clock(row_first - first + second * FPS)[:-2]))
        svg.append("</svg>")
        parts.append("\n".join(svg))
        row_first = row_last + 1
    return "\n".join(parts)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("submission", help="a frame,label CSV for the whole match")
    parser.add_argument("--out", required=True, help="folder to write the two files into")
    parser.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        help="the challenge repository (for the public answers and the metric code)")
    parser.add_argument("--name", default="our labels", help="what this submission is, for the title")
    args = parser.parse_args()

    sys.path.insert(0, os.path.join(args.repo, "evaluation"))
    import possession_metrics as pm

    answer_frame = pm.read_label_csv(os.path.join(args.repo, "evaluation", "gt",
                                                  "BAR-ATH_possession_gt_public.csv"))
    ours_frame = pm.read_label_csv(args.submission)
    problem = pm.validate_submission(ours_frame, len(answer_frame))
    if problem:
        raise SystemExit("the submission is not valid: " + problem)
    answer_codes, ours_codes = pm.label_codes(answer_frame), pm.label_codes(ours_frame)
    result = pm.evaluate(answer_codes, ours_codes)

    letters = np.array(pm.LABELS)
    answer, ours = letters[answer_codes], letters[ours_codes]
    graded = np.flatnonzero(answer != "X")
    first, last = int(graded[0]), int(graded[-1])

    answer_runs = runs(answer, first, last)
    ours_runs = runs(ours, first, last)
    wrong = np.where((answer != ours) & (answer != "X"), "W", "-")
    wrong_runs = runs(wrong, first, last)
    pair_runs = runs(np.char.add(answer, ours), first, last)

    os.makedirs(args.out, exist_ok=True)
    csv_path = os.path.join(args.out, "compare_5min.csv")
    rows = []
    for start, stop, pair in pair_runs:
        a, o = pair[0], pair[1]
        rows.append({"start_frame": start, "end_frame": stop - 1,
                     "from": clock(start - first), "to": clock(stop - first),
                     "seconds": round((stop - start) / float(FPS), 2),
                     "answer": a, "ours": o, "same": "yes" if a == o else "no",
                     "kind": "" if a == o else kind_of(a, o)})
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    differences = [r for r in rows if r["same"] == "no"]
    total = (last - first + 1) / float(FPS)
    by_kind = {}
    for r in differences:
        by_kind[r["kind"]] = by_kind.get(r["kind"], 0.0) + r["seconds"]

    stats = [
        ("%.4f" % result["spell_f1_mean"], "spellF1, the ranking score"),
        ("%.4f" % result["macro_f1"], "macroF1, the tie-break"),
        ("%d / %d" % (result["n_pred_spells"], result["n_gt_spells"]), "possessions: ours / the answer"),
        ("%.1f%%" % (100 * result["accuracy"]), "of frames have the right label"),
        ("%.2f" % result["alive_dead_f1"], "play against stopped (F1)"),
        ("%.1f%%" % (100 * result["team_acc_both_alive"]), "right team, where both say play"),
    ]
    page = ["<!DOCTYPE html>", '<html lang="en"><head><meta charset="utf-8">',
            '<meta name="viewport" content="width=device-width, initial-scale=1">',
            "<title>Public 5 minutes</title>", "<style>%s</style></head>" % STYLE,
            '<body class="viz-root">',
            "<h1>Our labels against the organisers' public 5 minutes</h1>",
            "<p>%s &middot; video frames %d to %d &middot; %.0f seconds</p>"
            % (html.escape(args.name), first, last, total),
            '<div class="card stats">']
    page += ['<div class="stat"><b>%s</b><span>%s</span></div>' % stat for stat in stats]
    page += ["</div>", "<h2>Timeline</h2>", '<div class="card">', '<div class="legend">',
             '<span><i style="background:var(--home)"></i>Barcelona has the ball (H)</span>',
             '<span><i style="background:var(--away)"></i>Athletic has the ball (A)</span>',
             '<span><i style="background:var(--grid)"></i>Play stopped (D)</span>',
             '<span><i style="background:var(--wrong);height:5px"></i>Ours differs from the answer</span>',
             "</div>", timeline(answer_runs, ours_runs, wrong_runs, first, last), "</div>",
             "<h2>Where the differences are</h2>", '<div class="card scroll"><table>',
             "<tr><th>Kind of difference</th><th>Seconds</th><th>Share of the 5 minutes</th></tr>"]
    for kind, seconds in sorted(by_kind.items(), key=lambda item: -item[1]):
        page.append("<tr><td>%s</td><td>%.1f</td><td>%.1f%%</td></tr>"
                    % (kind, seconds, 100 * seconds / total))
    page += ["</table></div>", "<h2>Every difference</h2>", '<div class="card scroll"><table>',
             "<tr><th>From</th><th>To</th><th>Seconds</th><th>Answer</th><th>Ours</th><th>Kind</th></tr>"]
    for r in differences:
        page.append("<tr><td>%s</td><td>%s</td><td>%.1f</td><td>%s</td><td>%s</td><td>%s</td></tr>"
                    % (r["from"], r["to"], r["seconds"], MEANING[r["answer"]], MEANING[r["ours"]],
                       r["kind"]))
    page += ["</table></div>",
             '<p class="note" style="margin-top:24px">Every stretch, the same and the different, is in '
             "compare_5min.csv next to this file. This page is a check that the pipeline is right. "
             "The rules do not allow choosing settings or versions with these 5 minutes. "
             "It is made from the match data, so keep it on this computer.</p>",
             "</body></html>"]
    html_path = os.path.join(args.out, "compare_5min.html")
    with open(html_path, "w") as handle:
        handle.write("\n".join(page))

    print("spellF1 %.4f | macroF1 %.4f" % (result["spell_f1_mean"], result["macro_f1"]))
    print("possessions: ours %d, the answer %d" % (result["n_pred_spells"], result["n_gt_spells"]))
    print("right label on %.1f%% of frames | play against stopped F1 %.2f | right team %.1f%% where both say play"
          % (100 * result["accuracy"], result["alive_dead_f1"], 100 * result["team_acc_both_alive"]))
    for kind, seconds in sorted(by_kind.items(), key=lambda item: -item[1]):
        print("  %-30s %6.1f s" % (kind, seconds))
    print("wrote", html_path)
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
