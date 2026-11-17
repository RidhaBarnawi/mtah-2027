# Our entry: who has the ball, frame by frame

Team **NRZ Ace** — More than a Hack 2027, Phase 1.

This folder is our program. Everything else in the repository is the organisers' starter kit,
unchanged, plus `practice/`, the practice exam we chose our settings on.

## Reproduce our submission

Python 3.12. Only two of the provided files are read:
`tactical/BAR-ATH_tactical_atd.xml` and `tactical/BAR-ATH_tactical_atd.txt`.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r possession/requirements.txt

# Unpack the competition bundle into ./data, or point MTAH_DATA_DIR at it
python possession/make_submission.py --no-cache --out "evaluation/results/NRZ Ace.csv"
python evaluation/validate_submission.py "evaluation/results/NRZ Ace.csv"
```

It takes a minute or two on a laptop, with no GPU. There is no randomness: the same data gives
the same file, byte for byte. SHA-256 of the file we submitted:

    4056bae7d48f5140ed2b69699ab1c0d6da02f6acd3e7d03cbe74a586ec4ef0d1

## How it works

1. **Read the tracking** (`tracking.py`) into arrays: every track slot's position, its team, and
   the ball, per frame, in metres.
2. **Compute signals** (`signals.py`), one row per frame: is the ball seen, is it off the pitch,
   how fast it moves, the nearest player of each team, how fast the ball moves relative to the
   nearest player, kinks in the ball's path, how the players move.
3. **Turn signals into two pieces of evidence** (`decode.py`):
   - *Is play stopped?* The ball off the pitch or not seen, the ball lying still, the players
     walking.
   - *Which team has it?* The nearest player, counted only when he is on the ball and the ball
     moves with him; touches (a sharp kink in the ball's path beside a player); and a small
     learned model (`team_model.py`, weights in `team_model.json`) whose strongest clue is which
     team touched the ball last.
4. **Decode each half as a whole** (`decode.py`): the best sequence of home / away / dead, where
   every change of state has a cost. A label changes only when the evidence for the change,
   added up over the following frames, outweighs that cost. This is what turns flickering
   per-frame guesses into whole possessions, which is what the ranking score counts.

`make_submission.py` runs steps 1 to 4 on the competition match and writes the CSV.

## How the settings and the model were chosen

Never on the competition match. We used Metrica Sports' three public sample matches
(github.com/metrica-sports/sample-data), which come with tracking and event data:

- `practice/build_answer_key.py` turns each match's events into H / A / D answers.
- `practice/make_practice_exam.py` rewrites the matches in the competition's tracking format and
  makes them as hard as the organisers' published facts say the competition match is: 46
  anonymous track slots, about 20.5 of 22 players tracked, the ball missing in stretches.
- `practice/practice_exam.py` grades a program with the organisers' own metric code, and
  `practice/tracking_noise.py` grades it again with tracking errors added.
- `train_team_model.py` trains the learned clue on those matches. `--check` trains on two
  matches and scores the third; that is the comparison the model was accepted on.

Average spellF1 on the three practice matches:

| Version | Clean tracking | Noisy | Harsh |
|---|---|---|---|
| Organisers' baseline (nearest player, ±2 s vote) | 0.289 | | |
| v1: stopped / team evidence + decoder | 0.443 | 0.327 | 0.271 |
| v2: tighter "on the ball" rule, player speed over 1 s | 0.483 | 0.467 | 0.423 |
| v3: + ball moving with the player, + touches | 0.531 | 0.503 | 0.430 |
| **Submitted: v3 + learned team clue** (scored on the match left out of training) | **0.568** | **0.537** | **0.468** |

The public 5 minutes of the competition match were used only to check the pipeline (frames
aligned, home and away the right way round), with `compare_public.py`.

## Files

| File | What it does |
|---|---|
| `make_submission.py` | competition match in, submission CSV out |
| `tracking.py` | reads the competition tracking, and Metrica's sample matches |
| `signals.py` | per-frame signals |
| `decode.py` | evidence, settings (`PARAMS`) and the decoder |
| `team_model.py`, `team_model.json` | the learned "who has the ball" clue and its weights |
| `train_team_model.py` | trains and checks that clue on the practice matches |
| `compare_public.py` | compares a submission with the public 5 minutes (a pipeline check) |
| `make_testset.py`, `evaluate_samples.py` | our first, simpler test set from the same sample matches |
| `requirements.txt` | exact library versions |
