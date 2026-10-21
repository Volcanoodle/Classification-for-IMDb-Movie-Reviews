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
