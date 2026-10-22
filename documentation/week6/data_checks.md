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
- Select three reviews uniformly at random from the 50,000 labelled reviews. By default no seed is used, so a later run selects different reviews; pass `--seed N` for a reproducible sample.

Run from the repository root:

```powershell
.venv/Scripts/python.exe -I scripts/check_text_cleaning.py
.venv/Scripts/python.exe -I scripts/check_text_cleaning.py --seed 123  # optional: reproducible
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

The examples below are from one run without a seed. Running the script again will select different reviews; pass `--seed N` to make a draw reproducible. Example 3 contains no `<br />` and no repeated or leading/trailing whitespace, so it is unchanged; the rules do not alter reviews that are already clean.

```text
--- Example 1: train/neg/8793_1.txt ---
BEFORE:
For those deciding whether or not to watch this movie and are reading these comments for insight, I first offer these four words: Don't waste your time! "Chungking Express" was shoddily filmed, slapped together quickly and seems as though it were conceived in its entirety by someone standing in line at a Hong Kong Burger King. I can't remember ever watching a film with an assortment of such one-dimensional characters trying to work their way through a script this mundane! It's an absurd effort with philosophically ridiculous dialog (a man wanders into his flooded apartment and offers the stunning revelation that "tears can be dried with a tissue, but water takes time to mop up"). The same character is also seen carrying on a deep, meaningful rapport with his towels, soap, stuffed animals, dirty laundry, etc. The shaky, wandering, hand-held camera-work was another annoying feature I could have done without. And if that isn't enough to make you puke in your popcorn, we hear the old 60's ode "California Dreaming" by the Mamas and Papas blaring over the soundtrack over and over again during a particularly lengthy sequence.<br /><br />Quentin Tarantino was responsible for bringing this loser to America through his Rolling Thunder Productions company, though I cannot for the life of me figure out why a man with his talent would bother. He was known to have remarked, "I'm happy to love a movie this much." A lot of us, though, hope he will concentrate on making his audiences happy with more worthwhile discoveries in the future.

AFTER:
For those deciding whether or not to watch this movie and are reading these comments for insight, I first offer these four words: Don't waste your time! "Chungking Express" was shoddily filmed, slapped together quickly and seems as though it were conceived in its entirety by someone standing in line at a Hong Kong Burger King. I can't remember ever watching a film with an assortment of such one-dimensional characters trying to work their way through a script this mundane! It's an absurd effort with philosophically ridiculous dialog (a man wanders into his flooded apartment and offers the stunning revelation that "tears can be dried with a tissue, but water takes time to mop up"). The same character is also seen carrying on a deep, meaningful rapport with his towels, soap, stuffed animals, dirty laundry, etc. The shaky, wandering, hand-held camera-work was another annoying feature I could have done without. And if that isn't enough to make you puke in your popcorn, we hear the old 60's ode "California Dreaming" by the Mamas and Papas blaring over the soundtrack over and over again during a particularly lengthy sequence. Quentin Tarantino was responsible for bringing this loser to America through his Rolling Thunder Productions company, though I cannot for the life of me figure out why a man with his talent would bother. He was known to have remarked, "I'm happy to love a movie this much." A lot of us, though, hope he will concentrate on making his audiences happy with more worthwhile discoveries in the future.
```

```text
--- Example 2: train/neg/6410_1.txt ---
BEFORE:
I think a round of applause is in order for whoever pieced together the trailer for Rogue Pictures' latest release, 'The Return'. I myself, along with everyone else have been duped into believing it is in fact a horror film. On the contrary though, its actually a supernatural thriller. Too bad it is not the least bit thrilling.<br /><br />'The Return' stars Sarah Michelle Gellar as Joanna Mills, a young woman who has had personal problems since the age of eleven. It was at that age that she began having haunting visions depicting the murder of a woman whom she has never met. While in Texas on a business trip, she is led by these visions to the murdered woman's hometown, La Salle. There she comes face to face with another person who has frequently appeared in her visions. A man by the name of Terry Stahl, who is played by Peter O'Brien. Joanna is now on a desperate search for answers. A search that could in the end result in her very own murder.<br /><br />I really don't know where to begin here folks. Which should I mention first? The atrocious acting, hideous directing, or the terribly bland story? No matter which one I choose my point behind each is the same: they simply suck. Adam Sussman's screenplay is downright moronic. Its not interesting. Its not compelling. Its just plain unpleasant. I kept waiting for something to jumpstart the "film" (I've placed quotations around film because I don't believe 'The Return' deserves to be called an actual film due to it's foulness.) and at least give it some slight chance of hope, but nothing ever happened. I was left out in an unbearable cold to freeze. Not even stellar performances from the actors themselves could have saved this disaster. Of course they probably knew this having read the script then agreeing to do the "film". I assume this is why the acting was so awful. At least that's what I'm choosing to believe. I really hope the cast doesn't pride themselves on their performances. They need immediate medical attention if they do.<br /><br />Now the directing was unmistakably bad, but I can't quite crucify Asif Kapadia entirely. (Well I could, but I won't since I'm such a nice guy.) I look at 'The Return' as a test for Kapadia because for all of you that don't know, this is his first full-length feature "film". He's just getting his foot in the door and still learning. Next time around, well if there is a next time, hopefully he will have improved vastly. The only thing he was able to accomplish here was almost completely duplicating the visual style of Marcus Nispel's 2003 re-make of 'The Texas Chainsaw Massacre'. Now that's nice the "film" was given that, but unfortunately he still won't be receiving any kudos from me for that. Copying someone else's work isn't something I consider to be praise worthy. (Even if it is from a film I very much enjoyed.)<br /><br />I think Jim Sonzero's American re-make of 'Pulse' will now have to fork over the title of Worst Film of the Year to 'The Return'. It beyond question is deserving of the title in almost every imaginable way. Now I don't doubt this will make a small, and I mean very small, profit. No matter what though, it won't surpass or even parallel the fluke success of Gellar's previous acting effort, 'The Grudge'. On that note, there's one last thing I'd like to add. I can honestly say I had never been embarrassed to have been seen leaving a theater auditorium until seeing 'The Return'. That is something I never wish to experience again, along with the "film" itself.

AFTER:
I think a round of applause is in order for whoever pieced together the trailer for Rogue Pictures' latest release, 'The Return'. I myself, along with everyone else have been duped into believing it is in fact a horror film. On the contrary though, its actually a supernatural thriller. Too bad it is not the least bit thrilling. 'The Return' stars Sarah Michelle Gellar as Joanna Mills, a young woman who has had personal problems since the age of eleven. It was at that age that she began having haunting visions depicting the murder of a woman whom she has never met. While in Texas on a business trip, she is led by these visions to the murdered woman's hometown, La Salle. There she comes face to face with another person who has frequently appeared in her visions. A man by the name of Terry Stahl, who is played by Peter O'Brien. Joanna is now on a desperate search for answers. A search that could in the end result in her very own murder. I really don't know where to begin here folks. Which should I mention first? The atrocious acting, hideous directing, or the terribly bland story? No matter which one I choose my point behind each is the same: they simply suck. Adam Sussman's screenplay is downright moronic. Its not interesting. Its not compelling. Its just plain unpleasant. I kept waiting for something to jumpstart the "film" (I've placed quotations around film because I don't believe 'The Return' deserves to be called an actual film due to it's foulness.) and at least give it some slight chance of hope, but nothing ever happened. I was left out in an unbearable cold to freeze. Not even stellar performances from the actors themselves could have saved this disaster. Of course they probably knew this having read the script then agreeing to do the "film". I assume this is why the acting was so awful. At least that's what I'm choosing to believe. I really hope the cast doesn't pride themselves on their performances. They need immediate medical attention if they do. Now the directing was unmistakably bad, but I can't quite crucify Asif Kapadia entirely. (Well I could, but I won't since I'm such a nice guy.) I look at 'The Return' as a test for Kapadia because for all of you that don't know, this is his first full-length feature "film". He's just getting his foot in the door and still learning. Next time around, well if there is a next time, hopefully he will have improved vastly. The only thing he was able to accomplish here was almost completely duplicating the visual style of Marcus Nispel's 2003 re-make of 'The Texas Chainsaw Massacre'. Now that's nice the "film" was given that, but unfortunately he still won't be receiving any kudos from me for that. Copying someone else's work isn't something I consider to be praise worthy. (Even if it is from a film I very much enjoyed.) I think Jim Sonzero's American re-make of 'Pulse' will now have to fork over the title of Worst Film of the Year to 'The Return'. It beyond question is deserving of the title in almost every imaginable way. Now I don't doubt this will make a small, and I mean very small, profit. No matter what though, it won't surpass or even parallel the fluke success of Gellar's previous acting effort, 'The Grudge'. On that note, there's one last thing I'd like to add. I can honestly say I had never been embarrassed to have been seen leaving a theater auditorium until seeing 'The Return'. That is something I never wish to experience again, along with the "film" itself.
```

```text
--- Example 3: test/pos/10046_10.txt ---
BEFORE:
great historical movie, will not allow a viewer to leave once you begin to watch. View is presented differently than displayed by most school books on this subject. My only fault for this movie is it was photographed in black and white; wished it had been in color ... wow !

AFTER:
great historical movie, will not allow a viewer to leave once you begin to watch. View is presented differently than displayed by most school books on this subject. My only fault for this movie is it was photographed in black and white; wished it had been in color ... wow !
```

### Limitations

This script implements issue #2's rules independently and does not import or reuse `scripts/pre_data.py`. That file had been read during an earlier review, so this was not a strictly blind check. The examples are from one unseeded draw and cannot be reproduced exactly; pass `--seed N` to make future draws reproducible. The counts do not depend on which examples are selected.

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
