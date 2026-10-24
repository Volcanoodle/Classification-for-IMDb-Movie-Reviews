## Movie ID spot check

### Purpose

Movie IDs are used to keep reviews of the same work in the same fold. Matching row counts alone cannot establish whether reviews have been assigned the correct IDs, so selected reviews were compared manually with their IMDb pages.

### Method

`scripts/pre_data.py` extracts the numeric review ID from each filename and uses it as a zero-based index into the corresponding URL file. Review ID N therefore corresponds to line N + 1 when counting from 1.

Samples near changes in movie ID were selected. Within a block of reviews for the same work, taking an adjacent URL might still return the same ID. Boundary samples can reveal this type of indexing error.

IMDb pages were opened manually and compared with the titles, cast, and plot details in the reviews. Review paths below are relative to `data/raw/aclImdb/`. URL line numbers start at 1.

### Reviews checked

| Review file | URL file and line | IMDb work | Matching evidence |
| --- | --- | --- | --- |
| `train/pos/2_9.txt` | `train/urls_pos.txt`, line 3 | [Bromwell High (2005)](https://www.imdb.com/title/tt0453418/) | Explicit title and London school setting match. IMDb identifies it as a TV series. |
| `train/neg/0_3.txt` | `train/urls_neg.txt`, line 1 | [Futz (1969)](https://www.imdb.com/title/tt0064354/) | The unusual plot involving a man's affection for a pig matches. |
| `train/neg/1_1.txt` | `train/urls_neg.txt`, line 2 | [Stanley & Iris (1990)](https://www.imdb.com/title/tt0100680/) | Robert De Niro, Jane Fonda, Swoosie Kurtz, and the illiterate protagonist match. |
| `test/neg/5_4.txt` | `test/urls_neg.txt`, line 6 | [The Guardian (2006)](https://www.imdb.com/title/tt0406816/) | Kevin Costner, Ashton Kutcher, and Coast Guard rescue-swimmer training match. |
| `test/neg/6_3.txt` | `test/urls_neg.txt`, line 7 | [The Dresser (1983)](https://www.imdb.com/title/tt0085461/) | Cast, director, writer, and the King Lear performance match. |
| `test/pos/31_10.txt` | `test/urls_pos.txt`, line 32 | [The Dresser (1983)](https://www.imdb.com/title/tt0085461/) | Explicit title, Albert Finney, Tom Courtenay, and plot match. |
| `test/pos/32_10.txt` | `test/urls_pos.txt`, line 33 | [Lackawanna Blues (2005)](https://www.imdb.com/title/tt0407936/) | Explicit title, director George C. Wolfe, writer, and principal cast match. |
| `test/pos/61_9.txt` | `test/urls_pos.txt`, line 62 | [Lackawanna Blues (2005)](https://www.imdb.com/title/tt0407936/) | Explicit title, boardinghouse setting, writer, and actress S. Epatha Merkerson match. |
| `test/pos/78_8.txt` | `test/urls_pos.txt`, line 79 | [Acrobatty Bunny (1946)](https://www.imdb.com/title/tt0038286/) | Explicit title, director Robert McKimson, and circus plot involving Bugs Bunny and a lion match. |

### Boundary checks

- In `train/pos`, ID 2 maps to Bromwell High, while ID 3 maps to Futz. Taking the next URL for ID 2 would assign a different work.
- In `train/neg`, IDs 0 and 1 map to Futz and Stanley & Iris respectively. Their different plots distinguish the two works.
- In `test/neg`, IDs 5 and 6 map to The Guardian and The Dresser respectively. Their casts and plots distinguish the two works.
- In `test/pos`, IDs 31 and 32 explicitly name The Dresser and Lackawanna Blues respectively. Taking the next URL for ID 31, or the previous URL for ID 32, would assign the wrong work.

### Inconclusive sample

`test/pos/17_8.txt` maps to The Dresser (`tt0085461`), but its short review names Tom Courtenay and Peter Finch. The IMDb page lists Albert Finney alongside Tom Courtenay.

This sample was not counted among the nine confirmed matches. The discrepancy does not by itself establish an indexing error; the review may contain an incorrect actor name. The later review `test/pos/31_10.txt` explicitly names The Dresser and its matching cast, but does not resolve the discrepancy in review 17.

### Conclusion and limitations

The nine checked samples support the review-to-movie-ID mapping at those positions. No definite movie-ID mismatch was identified. One additional sample remains inconclusive.

This was a targeted spot check, not an exhaustive validation of all 50,000 reviews. Some matches rely on cast and plot details rather than an explicit title.

## Duplicate count check
### Method

These counts were obtained using `load_raw()` and `clean_text()` from `scripts/pre_data.py`.

The cleaning function replaces `<br />` with spaces, collapses repeated whitespace, and strips leading and trailing whitespace. Text comparisons are exact and case-sensitive after these steps.

### Results

| Check | Count |
| --- | ---: |
| F. Number of distinct cleaned training texts | 24,902 |
| G. Distinct cleaned training texts also present in the test set | 123 |
| H. Training rows remaining after keeping one copy per text and excluding texts present in the test set | 24,779 |

The original training set contains 25,000 rows. Deduplication removes 98 repeated rows. Excluding the 123 distinct texts shared with the test set leaves 24,779 rows.

These are calculated counts only. No reviews were deleted or dataset files modified. This calculation reuses the project's cleaning function; it is not an independent implementation of text cleaning.
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
- Show only an excerpt of each review, about 200 characters by default (`--excerpt-chars`). The excerpt is taken around the first change; full review text is not reproduced because the dataset has no licence that permits redistribution.

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

This script implements issue #2's rules independently and does not import or reuse `scripts/pre_data.py`. That file had been read during an earlier review, so this was not a strictly blind check. The examples were generated with `--seed 20261021 --count 3` and are excerpts of about 200 characters; full review text is not reproduced because the dataset has no licence that permits redistribution. The counts do not depend on which examples are selected.

## Comparison with pre_data.py

| Check | Reviewer | Reviewer's result | pre_data.py | Match |
| --- | --- | ---: | ---: | :---: |
| A. Reviews containing `<br />` before cleaning | Enze ZHOU | 29,200 | 29,200 | Yes |
| B. Reviews changed by cleaning | Enze ZHOU | 29,376 | 29,376 | Yes |
| C. Reviews still containing `<br` after cleaning | Enze ZHOU | 0 | 0 | Yes |
| D. Reviews empty after cleaning | Enze ZHOU | 0 | 0 | Yes |
| F. Distinct train texts | Junqiang Zhu | 24,902 | not printed | – |
| G. Distinct train texts also in test | Junqiang Zhu | 123 | not printed | – |
| H. Train rows after de-duplication | Junqiang Zhu | 24,779 | 24,779 | Yes |
| Movie IDs at 9 boundary samples | Junqiang Zhu | 9 matches, 0 mismatches | – | – |

F and G are not printed by the script, but they are consistent with H: 24,902 − 123 = 24,779.

