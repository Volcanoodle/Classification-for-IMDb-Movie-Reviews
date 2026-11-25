# Error Analysis
According to error_finder.csv, there are 718 reviews where all 6 models' predictions contradict with the label, among which 426 negative reviews are predicted as positive andd 292 positve ones are predicted as negative.

About 30 reviews were picked randomnly from each label and analysed,

## The major reason for wrong predictions are:

1. Praise for one part can outweigh the negative arguments, or the contrary
    -  train/neg/10053: (negative label, all 6 positive)\
    The review says "The saving grace is Emma Vilarasau, who does an outstanding job", giving a positive comment, but the rest of the review is negative.\
    The models took too much weight in this compliment.
2. Limit of ngram used in training:
    - train/neg/10216: (negative label, all 6 positive)\
    The review says "until the very end, I would have given 9 out of 10 stars to the movie", then "no longer a masterpiece".\
    Most of the models used unigram, which does not allow them to learn long phrases. 
3. Sarcasms in the review:
    - train/neg/3654: (negative label, all 6 positive)\
    The review calls Shogun Mayeda a “Highly recommended cinematic masterpiece.” Its final sentence says “All of the above is opposite for the film in question.”

4. The praise or criticism belongs to other works:
    - train/neg/3383: (negative label, all 6 positive)\
    The writer praises a live musical as “absolutely mind-blowing” and wants to see it again. The film adaptation, the work being rated, left the author feeling “cheated.”\
    The models failed to understand that the praise does not belong to the movie being reviewed.
5. Maybe the label provided by the data set is not correct:
    - train/neg/11755: (negative label, all 6 positive)\
    The reviewer says “I loved the movie” and “It's a great movie!”. \
    It should be a positive review, and all 6 models predict correctly, but the label says the opposite.

