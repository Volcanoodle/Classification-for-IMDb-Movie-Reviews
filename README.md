# Machine Learning-Based Sentiment Classification for IMDb Movie Reviews

## Group name

DL_Group_9

## Group members

* Xiangxi Shu
* Dani Haritz Leozarin
* Zihan Yang
* Enze ZHOU
* Junqiang Zhu

## Objectives

* Classify English movie reviews as expressing either positive or negative sentiment (Binary Classification).
* Apply unsupervised clustering (k-means, EM) to observe whether reviews naturally group by sentiment or other underlying factors (e.g., movie genre).
* Evaluate the impact of different text preprocessing and representation methods (e.g., term frequency counts vs. preserving word order) on model performance.
* Compare the performance of classical machine learning models (Naive Bayes, Logistic Regression, Decision Trees, Random Forests) on the same test set.
* Compare classical baselines against neural network architectures (MLP, 1D-CNN) and analyze the reasons for performance differences.
* Explore the boundaries and failure cases of the models (e.g., sarcasm, mixed reviews, plot summaries).

## Milestones

| Requirement | Planned completion |
| --- | --- |
| R1 Topic selection, objectives, and datasets | Week 4 (Pitch) |
| Data cleaning, preprocessing, and splits | Week 5 |
| R2 Data exploration, visualization, and clustering | Week 6 - Week 7 |
| R3 Classical baseline models training and comparison | Week 7 |
| R4 Neural networks (MLP, 1D-CNN) and final comparison | Week 8 - Week 9 |
| Report drafting and code freezing | Week 10 |
| Final report and repository submission | Week 11 |

## Datasets

### Source

* **Name:** Large Movie Review Dataset v1.0
* **Origin:** Stanford University AI Lab, from the paper "Learning Word Vectors for Sentiment Analysis" (Maas et al., 2011, ACL).
* **Official URL:** [https://ai.stanford.edu/~amaas/data/sentiment/](https://ai.stanford.edu/~amaas/data/sentiment/)
* **Size:** 50,000 labeled movie reviews (25,000 for training, 25,000 for testing), perfectly balanced with 50% positive and 50% negative. An additional 50,000 unlabeled reviews are included.
* **Labeling logic:** Reviews with an IMDb rating of 7 or higher are labeled as positive (1); reviews with a rating of 4 or lower are labeled as negative (0). Neutral reviews are not included.
* **Note:** Please rely on the official URL provided above, as identically named datasets on Kaggle are third-party re-uploads.

### License

The official website does not specify a standard open-source license (e.g., MIT, GPL), but it explicitly requires citing the aforementioned Maas et al. (2011) paper when using the dataset.

### Examples

*(To be updated with actual data excerpts during the exploration phase)*

| Label | Review (excerpt) |
| --- | --- |
| positive | [To be extracted] |
| negative | [To be extracted] |

## Installation

This project requires Python 3.14 (any 3.14.x patch version is supported).

1. Clone the repository and navigate to the project folder.
2. Create a virtual environment:
   * Windows: `py -3.14 -m venv .venv`
   * macOS / Linux: `python3.14 -m venv .venv`


3. Activate the virtual environment:
   * Windows (PowerShell): `.\.venv\Scripts\Activate.ps1`
   * macOS / Linux: `source .venv/bin/activate`


4. Install dependencies: `python -m pip install -r requirements.txt`
5. Launch Jupyter Lab: `jupyter lab`

## Data preparation pipeline
1. Download `aclImdb_v1.tar.gz` from the official URL and extract it so that the folder `data/raw/aclImdb/` exists.
2. From the project root, run `python scripts/pre_data.py`. It takes a few minutes and prints a check after each step.
The script:
1. reads the 50,000 labelled reviews (the 50,000 unlabelled ones are not used);
2. replaces `<br />` with a space and collapses repeated whitespace;
3. removes repeated reviews from the training set, including training reviews that also appear in the test set; the official test set is kept unchanged;
4. assigns each training review to one of 10 folds, stratified by label and grouped by movie (seed 42);
5. writes `data/processed/train.csv` (24,779 rows) and `data/processed/test.csv` (25,000 rows), which are not tracked by git, and `data/splits/train_folds.csv`, the reference fold assignment, which is tracked by git.
The `rating` column is for analysis only: it determines the label and must never be used as a model input.
Independent checks of these steps are in `documentation/week6/`.

## R2: Data analysis and exploration

*[Pending implementation - To be completed in Week 6]*

## R3: Baseline training and evaluation

*[Pending implementation - To be completed in Week 7]*

## R4: Neural networks

*[Pending implementation - To be completed in Week 9]*