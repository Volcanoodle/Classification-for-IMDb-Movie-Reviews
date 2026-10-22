"""Check the text-cleaning rules described in issue #2.

Reads the 50,000 labelled reviews, reports counts A-D, and prints
three random before/after examples (E). By default no seed is used, so
each run selects different examples; pass --seed for a reproducible
sample.

Usage:
    .venv/Scripts/python.exe -I scripts/check_text_cleaning.py
    .venv/Scripts/python.exe -I scripts/check_text_cleaning.py --seed 123
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "raw" / "aclImdb"
SPLITS = ("train", "test")
LABELS = ("pos", "neg")
WHITESPACE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Apply the three cleaning rules in order."""
    text = text.replace("<br />", " ")
    text = WHITESPACE.sub(" ", text)
    return text.strip()


def review_files() -> list[Path]:
    """Return all labelled review files in a stable order."""
    files = []
    for split in SPLITS:
        for label in LABELS:
            files.extend(sorted((DATA_DIR / split / label).glob("*.txt")))
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional seed for reproducible example selection.",
    )
    args = parser.parse_args()

    files = review_files()
    per_group = {(split, label): 0 for split in SPLITS for label in LABELS}
    containing_br = 0
    changed = 0
    still_br = 0
    empty = 0

    for path in files:
        before = path.read_text(encoding="utf-8")
        after = clean_text(before)

        per_group[(path.parent.parent.name, path.parent.name)] += 1
        if "<br />" in before:
            containing_br += 1
        if after != before:
            changed += 1
        if "<br" in after.lower():
            still_br += 1
        if after == "":
            empty += 1

    print(f"Total reviews: {len(files)}")
    for (split, label), count in per_group.items():
        print(f"  {split}/{label}: {count}")
    print()
    print(f'A. Reviews containing "<br />" before cleaning: {containing_br}')
    print(f"B. Reviews changed by cleaning: {changed}")
    print(f'C. Reviews still containing "<br" (any case) after cleaning: {still_br}')
    print(f"D. Reviews that end up empty: {empty}")
    print()

    if args.seed is None:
        sample = random.sample(files, k=3)
        print("E. Three random examples (no seed; a later run selects different reviews):")
    else:
        sample = random.Random(args.seed).sample(files, k=3)
        print(f"E. Three random examples (seed={args.seed}):")
    for index, path in enumerate(sample, start=1):
        before = path.read_text(encoding="utf-8")
        after = clean_text(before)
        relative = path.relative_to(DATA_DIR)
        print()
        print(f"--- Example {index}: {relative.as_posix()} ---")
        print("BEFORE:")
        print(before)
        print("AFTER:")
        print(after)


if __name__ == "__main__":
    main()
