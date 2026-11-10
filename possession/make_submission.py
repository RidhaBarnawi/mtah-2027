"""Write the submission for the competition match: tracking -> signals -> decoder -> CSV.

    python possession/make_submission.py               # writes evaluation/results/decoder_v1.csv
    python possession/make_submission.py --out PATH

Reads the provided match through games.py (MTAH_DATA_DIR) and labels it with the settings fixed in
possession/decode.py, which were chosen on Metrica's public sample matches. Then check the file the
same way the organisers will:

    python evaluation/validate_submission.py evaluation/results/decoder_v1.csv
    python evaluation/run_evaluation.py

evaluation/results/ is gitignored apart from the organisers' baselines, so the file -- which is
derived from the match data -- is never committed.
"""

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import games  # noqa: E402
from decode import LABELS, label_frames  # noqa: E402
from signals import compute_signals  # noqa: E402
from tracking import load_competition  # noqa: E402

# Params
DEFAULT_OUT = os.path.join(ROOT, "evaluation", "results", "decoder_v1.csv")


def write_submission(path, labels):
    """The frame,label CSV: a header, then one row per video frame from 0."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as handle:
        handle.write("frame,label\n")
        handle.writelines("{},{}\n".format(frame, label) for frame, label in enumerate(labels))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=DEFAULT_OUT, help="where to write the CSV")
    args = parser.parse_args()

    signals = compute_signals(load_competition())
    blocks = [(first, last) for _, first, last in games.half_video_blocks(games.GAME)]
    codes = label_frames(signals, blocks)
    write_submission(args.out, LABELS[codes])
    print("wrote {} ({} rows)".format(args.out, len(codes)))
