Run the following python code to check whether the data split is complete and well balanced:


```python
import pandas as pd

df = pd.read_csv("train_folds.csv")

folds = []
for _ in range(10):
    folds.append([0,0])
# create a 10-element array to keep track of how many entries each fold has been distributed
# each element is a 2-element array, 
# the first indicating the number of positive reviews, 
# and the second that of negative ones


totalPos = 0
totalNeg = 0

for row in df.itertuples():
    try:
        s = row.review_id.split('/')[1]
    except IndexError:
        s = None
    # extract the second part of the rewview_id, it should be either "pos" or "neg"
    if s!= None:
        if s == "pos":
            folds[row.fold][0] += 1
            totalPos += 1
        elif s == "neg":
            folds[row.fold][1] += 1
            totalNeg += 1

print("The following line shows the number of positive and negative reviews in each fold:")
print(folds)
print()
print(df["fold"].value_counts().sort_index())
print()
print("total positive = ", totalPos)
print("total negative = ", totalNeg)
print("total = ", totalPos + totalNeg)
```

    The following line shows the number of positive and negative reviews in each fold:
    [[1240, 1229], [1241, 1230], [1240, 1228], [1243, 1233], [1249, 1241], [1243, 1228], [1249, 1238], [1247, 1239], [1250, 1240], [1241, 1230]]
    
    fold
    0    2469
    1    2471
    2    2468
    3    2476
    4    2490
    5    2471
    6    2487
    7    2486
    8    2490
    9    2471
    Name: count, dtype: int64
    
    total positive =  12443
    total negative =  12336
    total =  24779


From the output above, 
it is clear that each fold has roughly equal numbers (around 1240) of postive and negative reviews,
and the total number of entries in each fold is consistantly around 2480,
the total number of positive and negative reviews are roughly equal (around 12400),
so it can be concluded that the data is balancedly distributed.

The total number of entries is 24779, as expected, so the data is complete.
