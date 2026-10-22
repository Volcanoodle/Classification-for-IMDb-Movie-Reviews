## Text cleaning check

### Purpose

Verify the three text-cleaning rules on all 50,000 labelled IMDb reviews and record results for comparison with the pipeline.

### Method

- Read `data/raw/aclImdb/{train,test}/{pos,neg}` as UTF-8: 12,500 reviews per folder; exclude `unsup`.
- Replace every exact `<br />` with a space, collapse `\s+` to one space, then strip leading and trailing whitespace.
- Count reviews containing `<br />` before cleaning, reviews changed, reviews still containing `<br` (case-insensitive) after cleaning, and reviews empty after cleaning.
- A review counts as changed when its cleaned text differs from the original text (exact string comparison).
- Use `scripts/check_text_cleaning.py`; no dataset files are modified.
- Select N reviews (default 3) uniformly at random from the 50,000 labelled reviews. By default no seed is used, so a later run selects different reviews; pass `--seed N` for a reproducible sample and `--count N` to change how many are shown.
- Show only an excerpt of each review, about 200 characters by default (`--excerpt-chars`). The excerpt is taken around the first change; full review text is not reproduced because the dataset licence does not permit full-text redistribution.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -I scripts/check_text_cleaning.py
.venv/Scripts/python.exe -I scripts/check_text_cleaning.py --seed 20261021
.venv/Scripts/python.exe -I scripts/check_text_cleaning.py --count 5
.venv/Scripts/python.exe -I scripts/check_text_cleaning.py --excerpt-chars 300
```

Counts A, C and D were also checked at the command line with `grep`/`find` over the same four folders.

### Results

| Check | Reviews |
| --- | ---: |
| A. Containing `<br />` before cleaning | 29,200 |
| B. Changed by cleaning | 29,376 |
| C. Still containing `<br` (any case) after cleaning | 0 |
| D. Empty after cleaning | 0 |

The 176 additional changed reviews contained no `<br />` and were changed by whitespace cleaning.

### Examples

The examples below were generated with `--seed 20261021`, the default `--count 3`, and the default 200-character excerpt. They are excerpts, not full reviews, because the dataset licence does not permit full-text reproduction. Example 3 is unchanged: it contains none of the targeted patterns, so the rules leave it as it is.

```text
--- Example 1: test/pos/8467_10.txt ---
BEFORE (excerpt around the first change, max 200 chars):
This film is an eery, but interesting film. I will tell you why it is eery later in this review.<br /><br />The film is interesting because it was the first film to ever contain sound. No, it may not [...]

AFTER (excerpt around the first change, max 200 chars):
This film is an eery, but interesting film. I will tell you why it is eery later in this review. The film is interesting because it was the first film to ever contain sound. No, it may not be a one [...]
```

```text
--- Example 2: test/pos/101_9.txt ---
BEFORE (excerpt around the first change, max 200 chars):
[...] inner sense of self-importance. Additionally, he seems tired and bored with the monotony of life.<br /><br />Later in the film, he walks into a bank robbery and manages to annoy the robbers so [...]

AFTER (excerpt around the first change, max 200 chars):
[...] inner sense of self-importance. Additionally, he seems tired and bored with the monotony of life. Later in the film, he walks into a bank robbery and manages to annoy the robbers so much that one [...]
```

```text
--- Example 3: test/neg/1047_4.txt ---
BEFORE (excerpt; review unchanged, max 200 chars):
I watched this episode with high hopes after seeing it on so many people's "Favourite episodes" list. I'm not boasting in any way, but from the start, I realised they were in some sort of toybox/can, [...]

AFTER (excerpt; review unchanged, max 200 chars):
I watched this episode with high hopes after seeing it on so many people's "Favourite episodes" list. I'm not boasting in any way, but from the start, I realised they were in some sort of toybox/can, [...]
```

### Limitations

This script implements issue #2's rules independently and does not import or reuse `scripts/pre_data.py`. That file had been read during an earlier review, so this was not a strictly blind check. The examples were generated with `--seed 20261021 --count 3` and are excerpts of about 200 characters; full review text is not reproduced because the dataset licence does not permit it. The counts do not depend on which examples are selected.
