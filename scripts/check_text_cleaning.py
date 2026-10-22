"""Check the text-cleaning rules described in issue #2.

Reads the 50,000 labelled reviews, reports counts A-D, and prints
three random before/after examples (E), truncated to about 200
characters by default. By default no seed is used, so each run selects
different examples; pass --seed for a reproducible sample.

Usage:
    .venv/Scripts/python.exe -I scripts/check_text_cleaning.py
    .venv/Scripts/python.exe -I scripts/check_text_cleaning.py --seed 123
    .venv/Scripts/python.exe -I scripts/check_text_cleaning.py --excerpt-chars 300
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
DEFAULT_EXCERPT_CHARS = 200


def clean_text(text: str) -> str:
    """Apply the three cleaning rules in order."""
    text = text.replace("<br />", " ")
    text = WHITESPACE.sub(" ", text)
    return text.strip()


def first_difference(left: str, right: str) -> int:
    """Return the index of the first character that differs."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return min(len(left), len(right))


def excerpt_at(text: str, start: int, limit: int) -> str:
    """Return a window of roughly `limit` characters around `start`."""
    start = max(0, start)
    end = min(len(text), start + limit)
    if start > 0:
        space = text.find(" ", start)
        if 0 <= space < start + 20:
            start = space + 1
    if end < len(text):
        space = text.rfind(" ", start, end)
        if space > start:
            end = space
    chunk = text[start:end]
    if start > 0:
        chunk = "[...] " + chunk
    if end < len(text):
        chunk = chunk + " [...]"
    return chunk


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
    parser.add_argument(
        "--excerpt-chars",
        type=int,
        default=DEFAULT_EXCERPT_CHARS,
        help="Maximum characters shown per example (default: %(default)s).",
    )
    args = parser.parse_args()
    if args.excerpt_chars <= 0:
        parser.error("--excerpt-chars must be positive")

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
        if before == after:
            start = 0
            label = "excerpt; review unchanged"
        else:
            difference = first_difference(before, after)
            start = 0 if difference < args.excerpt_chars else max(
                0, difference - args.excerpt_chars // 2
            )
            label = "excerpt around the first change"
        print(f"--- Example {index}: {relative.as_posix()} ---")
        print(f"BEFORE ({label}, max {args.excerpt_chars} chars):")
        print(excerpt_at(before, start, args.excerpt_chars))
        print(f"AFTER ({label}, max {args.excerpt_chars} chars):")
        print(excerpt_at(after, start, args.excerpt_chars))


if __name__ == "__main__":
    main()
