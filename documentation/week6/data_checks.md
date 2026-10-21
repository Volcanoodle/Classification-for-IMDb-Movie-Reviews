## Text cleaning check

### Purpose

This section independently verifies the project's text-cleaning rules. It records whether the three rules behave as expected across all 50,000 labelled reviews and provides counts that can be compared with the pipeline's own check.

### Data

- Source: `data/raw/aclImdb/` (Large Movie Review Dataset v1.0).
- Scope: the 50,000 labelled reviews from `train`/`test` × `pos`/`neg`; the `unsup` folders are excluded.
- Each file was read as UTF-8.
- Files counted per group: `train/pos` 12,500; `train/neg` 12,500; `test/pos` 12,500; `test/neg` 12,500.
- Only reads were performed; no review or dataset file was modified.

### Method

The check was implemented independently from the three rules stated in issue #2, without importing or reusing functions from `scripts/pre_data.py`:

1. Replace every `<br />` with a space: `text.replace("<br />", " ")`.
2. Collapse runs of whitespace (`\s+`) into one space: `re.sub(r"\s+", " ", text)`.
3. Strip leading and trailing whitespace: `text.strip()`.

The rules were applied in the order above. A review was counted as changed when the cleaned string differed from the original string, using exact comparison.

The implementation is `scripts/check_text_cleaning.py` and uses only the Python standard library. It was run with Python 3.14.7:

```powershell
.venv/Scripts/python.exe -I scripts/check_text_cleaning.py
```

The same counts were checked independently at the command line: `grep -rl '<br />'` over the four labelled folders returned 29,200 files, `grep -ril '<br'` returned 29,200, and `find ... -size 0` returned 0 empty files. This confirms counts A, C and D below for the raw data.

### Results

| Check | Count |
| --- | ---: |
| A. Reviews containing `<br />` before cleaning | 29,200 |
| B. Reviews changed by cleaning | 29,376 |
| C. Reviews still containing `<br` (any case) after cleaning | 0 |
| D. Reviews that end up empty | 0 |

B is 176 higher than A. Every review containing `<br />` is changed by rule 1, and 176 further reviews are changed only by whitespace handling (leading/trailing whitespace or repeated spaces, tabs or newlines). All 29,200 occurrences of `<br` in the dataset are the exact string `<br />`, so C is 0. No review is empty before or after cleaning, so D is 0.

### Examples

Three examples were selected at random from the reviews changed by cleaning, using `random.Random(20261021)`. To keep the examples readable, sampling was limited to reviews that contain `<br />` and are at most 1,000 characters; the full text is shown before and after.

```text
--- Example 1: test/neg/2553_4.txt ---
BEFORE:
I must say I was disappointed with this film. Although it is well acted and directed, the underlying story simply plods along too slowly.<br /><br />Granted, in another mood I would have liked it better. I did chuckle a lot, but rarely laughed out loud; and there was actually a sense of suspense to discover who won. But in contrast to another movie that my wife picked up the same day (one neither of us had heard of before) this one paled in comparison.<br /><br />If you see lots of movies, then by all means see this -- it's distinctly better than your average fare. But if you (like me) have limited time and want to watch only the best and most entertaining, save this for later.<br /><br />[Rate: 7/10]

AFTER:
I must say I was disappointed with this film. Although it is well acted and directed, the underlying story simply plods along too slowly. Granted, in another mood I would have liked it better. I did chuckle a lot, but rarely laughed out loud; and there was actually a sense of suspense to discover who won. But in contrast to another movie that my wife picked up the same day (one neither of us had heard of before) this one paled in comparison. If you see lots of movies, then by all means see this -- it's distinctly better than your average fare. But if you (like me) have limited time and want to watch only the best and most entertaining, save this for later. [Rate: 7/10]
```

```text
--- Example 2: test/pos/2949_10.txt ---
BEFORE:
Watching this film caused quite an emotional reaction. This is what today's documentaries are all about. It's refreshing to watch something so personal, honest and real. Mr. Block's thoughts, opinions and disclosure are rarely seen these days and are incredibly well displayed here. It's a fine line to walk between personal truth and exploitation. This film treads very carefully and quite successfully.<br /><br />One would think that learning about how a seemingly normal couple falls short of society's expectations would give birth to pessimism...but it doesn't. Quite the opposite: it made me feel good. I feel that I now know more about marriage...about women.<br /><br />Definitely check this out, it'll make you think - exactly what a good documentary should be designed to do.

AFTER:
Watching this film caused quite an emotional reaction. This is what today's documentaries are all about. It's refreshing to watch something so personal, honest and real. Mr. Block's thoughts, opinions and disclosure are rarely seen these days and are incredibly well displayed here. It's a fine line to walk between personal truth and exploitation. This film treads very carefully and quite successfully. One would think that learning about how a seemingly normal couple falls short of society's expectations would give birth to pessimism...but it doesn't. Quite the opposite: it made me feel good. I feel that I now know more about marriage...about women. Definitely check this out, it'll make you think - exactly what a good documentary should be designed to do.
```

```text
--- Example 3: test/neg/4781_1.txt ---
BEFORE:
This movie was really awful. It was not in the least bit frightening, or even startling. I went to see it with a bunch of friends and by the end of the night we were saying "The Ruins ruined my night." <br /><br />I would not recommend seeing this movie in theaters, renting it or even watching the movie on television by accident.It is an absolute waste of an hour and a half. <br /><br />The plot was nearly non-existent, the characters were horribly underdeveloped, and they gave no back story whatsoever for anything that was happening, and then left it completely open at the end as if preparing for a sequel.

AFTER:
This movie was really awful. It was not in the least bit frightening, or even startling. I went to see it with a bunch of friends and by the end of the night we were saying "The Ruins ruined my night." I would not recommend seeing this movie in theaters, renting it or even watching the movie on television by accident.It is an absolute waste of an hour and a half. The plot was nearly non-existent, the characters were horribly underdeveloped, and they gave no back story whatsoever for anything that was happening, and then left it completely open at the end as if preparing for a sequel.
```

### Limitations

The three rules themselves were taken from issue #2, so this check verifies their implementation rather than the choice of rules. The code does not import or reuse `scripts/pre_data.py`, but `scripts/pre_data.py` had already been read during an earlier repository review, so this is an independent reimplementation rather than a strictly blind check. The random examples are illustrative only; the counts cover all 50,000 labelled reviews.
