# Practice exam

We can't score our program on the competition match: the answers are secret, the public 5 minutes
are only for checking the pipeline, and labelling the match by hand is forbidden. So we score
it here, on **Metrica's 3 free sample matches** (github.com/metrica-sports/sample-data). The rules
allow data about other games, as long as the technical document explains it.

Everything in this folder is built from public data. Nothing in it comes from the competition
match. The large files (`sample-data/`, `exam/`, `answers/`) are not in the repository; the
commands under *Build it* recreate them exactly.

## Build it

From this folder, with the repository's Python (`pip install -r ../possession/requirements.txt`):

```
git clone https://github.com/metrica-sports/sample-data.git
../.venv/bin/python build_answer_key.py        # events -> answers/
../.venv/bin/python make_practice_exam.py      # sample matches -> exam/, about 3 minutes
```
The random parts use a fixed seed, so the exam comes out identical every time.

## Score a program (one command)

```
../.venv/bin/python practice_exam.py our_program.py --name "what this version is"
../.venv/bin/python tracking_noise.py our_program.py
```
`our_program.py` is the team's program (`../possession`). Without a program file,
`practice_exam.py` scores `example_program.py`, the organisers' baseline.

`practice_exam.py` prints the score for each match and the average, adds a row to
`scoreboard.csv`, and writes every mistake to `mistakes/<name>/` (open it in Excel).
`tracking_noise.py` scores the program again with tracking errors added.

**The team rule:** keep a change only if the **average spellF1** goes up by more than about
**0.02**, on noisy tracking as well as clean. Smaller differences are luck. Never decide using
the public 5 minutes.

## What a program must look like

One Python file with one function (copy `example_program.py` to start):

```python
def predict(atd_meta, atd_raw, n_frames, halves):
    return labels   # n_frames labels, each "H", "A", "D" or "X"; label i = frame_id i + 1
```

The exam has no video, Smart Tagging or homography, so a program that uses them needs to work
without them here.

## What is in this folder

| File | What it does |
|---|---|
| `build_answer_key.py` | Builds the correct answers from each match's event list → `answers/` |
| `check_answer_key.py` | Checks those answers against the player movements |
| `make_practice_exam.py` | Rewrites the 3 matches in the competition's tracking format and makes them as hard → `exam/` |
| `practice_exam.py` | The grader (uses the organisers' own metric code) |
| `tracking_noise.py` | Scores a program on clean, noisy and harsh tracking; `add_noise()` makes training copies |
| `example_program.py` | The organisers' baseline, as an example program |
| `our_program.py` | Connector: runs the team's program (`../possession`) on the exam |
| `make_pareto_file.py` | Merges one version's 3 mistake files into one file for a Pareto in Excel |

## How hard the exam is

Made to match what DATA.md says about the competition match:
- 46 anonymous track slots, so a slot is not a person
- about 20.5 of 22 players tracked
- the ball is missing on 35% of play frames (our estimate)
- speed in m/s

**Check:** the organisers' baseline at 4 smoothing settings ranks in the **same order** here as
on the full competition match (published in starter/baseline.py). The public 5 minutes rank them
**wrong**.

| Setting | Practice exam (average of 3) | Full match (published) |
|---|---|---|
| no smoothing | 0.047 | 0.049 |
| ±1 s | 0.197 | 0.200 |
| **±2 s** | **0.289** (best) | **0.243** (best) |
| ±5 s | 0.278 | 0.224 |

Look at the average, not one match: match 3 alone would have picked ±5 s.

**The answers.** Built from Metrica's event lists: the team of the latest pass, carry, shot,
recovery or set piece has the ball; play is stopped from the ball going out, a foul, a goal or a
shot going out until the restart, and every restart is treated as proof that play had stopped.
They agree with the answers built separately in `../possession/make_testset.py` on 96–98% of
frames.

**Weak spot.** In these matches the ball is missing on 88–100% of stopped frames, so the exam is
a poor judge of stoppage detection. We did not tune stoppage settings on it.

## Our versions on it

Average spellF1 on the 3 matches:

| Version | Clean | Noisy | Harsh |
|---|---|---|---|
| Organisers' baseline | 0.289 | | |
| v1 | 0.443 | 0.327 | 0.271 |
| v2 | 0.483 | 0.467 | 0.423 |
| v3 | 0.531 | 0.503 | 0.430 |
| v3 + learned team clue, scored on the match left out of training | 0.568 | 0.537 | 0.468 |
