"""Download the Enron email/reply dataset into ./data.

Usage:  uv run python scripts/fetch_data.py
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from hiver_email.data import default_data_dir  # noqa: E402

KAGGLE_DATASET = "oanannv/enron-email-reply-dataset"

MANUAL_INSTRUCTIONS = rf"""
Automatic download failed.

Kaggle needs credentials for programmatic download. Either:

  1. Create an API token at https://www.kaggle.com/settings  ->  "Create New Token".
     Save the downloaded kaggle.json to:
         Linux/macOS : ~/.kaggle/kaggle.json
         Windows     : %USERPROFILE%\.kaggle\kaggle.json
     Then re-run:  uv run python scripts/fetch_data.py

  2. Or set the environment variables KAGGLE_USERNAME and KAGGLE_KEY.

  3. Or download it by hand from
         https://www.kaggle.com/datasets/{KAGGLE_DATASET}
     and place the extracted .csv anywhere inside:
         {{data_dir}}

Everything downstream only needs a CSV in that folder.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=KAGGLE_DATASET)
    parser.add_argument("--data-dir", default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir) if args.data_dir else default_data_dir()
    data_dir.mkdir(parents=True, exist_ok=True)

    existing = sorted(data_dir.rglob("*.csv"))
    if existing:
        print(f"Dataset already present ({len(existing)} CSV file(s)):")
        for p in existing:
            print(f"  {p}  ({p.stat().st_size / 1e6:.1f} MB)")
        return 0

    try:
        import kagglehub
    except ImportError:
        print("kagglehub is not installed. Run: uv sync", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS.format(data_dir=data_dir), file=sys.stderr)
        return 1

    print(f"Downloading '{args.dataset}' from Kaggle via kagglehub ...")
    try:
        cache_path = Path(kagglehub.dataset_download(args.dataset))
    except Exception as exc:  # noqa: BLE001 - any failure means "tell the human"
        print(f"ERROR: {exc}\n", file=sys.stderr)
        print(MANUAL_INSTRUCTIONS.format(data_dir=data_dir), file=sys.stderr)
        return 1

    copied = 0
    for csv in sorted(cache_path.rglob("*.csv")):
        target = data_dir / csv.name
        shutil.copy2(csv, target)
        print(f"  -> {target}  ({target.stat().st_size / 1e6:.1f} MB)")
        copied += 1
    if not copied:
        print(f"No CSV inside the downloaded archive at {cache_path}", file=sys.stderr)
        return 1
    print(f"Done. {copied} file(s) in {data_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
