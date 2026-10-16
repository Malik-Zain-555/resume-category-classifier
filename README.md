# Resume Category Classifier

Classifies a resume into one of 24 job categories (HR, CHEF, BANKING, INFORMATION-TECHNOLOGY, TEACHER, and more) using TF-IDF and Logistic Regression. The project also looks inside the trained model to see which words push a resume toward each category.

**Status:** in progress. The model is trained and inspected. Evaluation deep-dive, an API, and containerization are still to do (see [Roadmap](#roadmap)).

## Problem

Given the text of a resume, predict its job category. This is a multi-class text classification problem with 24 classes.

## Dataset

`Resume.csv`, with the columns used here:

| Column | Meaning |
|---|---|
| `Resume_str` | Raw resume text |
| `Category` | Job category label (24 unique values) |

The dataset file is not included in this repo. Place `Resume.csv` where the notebook expects it (the notebook loads it from Google Drive in Colab).

## Approach

1. **Audit:** shape, null counts, category distribution.
2. **Cleaning:** lowercase; remove URLs and email addresses; keep letters only; collapse extra whitespace.
3. **Split:** 80/20 train/test, stratified by category (`random_state=42`).
4. **Features:** `TfidfVectorizer` with English stop words, `max_features=3000`, `ngram_range=(1, 2)`. The vectorizer is fit on the training set only and applied to the test set with `transform`, to avoid data leakage.
5. **Model:** `LogisticRegression(max_iter=1000)`.
6. **Evaluation:** accuracy and a per-class classification report.
7. **Interpretation:** the model's `coef_` matrix (24 x 3000) gives a weight for every word in every category. Sorting each row shows the top 10 words per category.

```python
for class_name, coef in zip(logReg.classes_, logReg.coef_):
    top = np.argsort(coef)[-10:]
    print(class_name)
    print(tfidf.get_feature_names_out()[top])
```

## Results

Overall test accuracy: **XX.X%** *(replace with the number from your notebook)*

### What the model learned

Many top words match intuition:

| Category | Example top words |
|---|---|
| HR | `payroll`, `recruitment`, `employee relations`, `human resources` |
| CHEF | `kitchen`, `culinary`, `sous`, `menu` |
| BANKING | `teller`, `mortgage`, `lending`, `loan` |
| TEACHER | `lesson`, `curriculum`, `classroom`, `students` |

### Odd findings

Some top words have no obvious link to their category, for example `dental` under ARTS, `high school` under SALES, `tibco` under AUTOMOBILE, and `india` and `cisco` under BPO.

I checked `dental` more closely:

- It appears in only 4 ARTS resumes, but in 10 HEALTHCARE and 10 HR resumes (counted over the full dataset).
- Its weight for ARTS is about 1.29, roughly 37% of the largest ARTS weight (about 3.48), and about 4x its HEALTHCARE weight (0.31).

This suggests the model picked up a pattern from a small group of resumes that is not a real signal for the category (possible spurious correlation or overfitting). Comparing train and test accuracy is the next check.

## Tech stack

Python, pandas, NumPy, scikit-learn, seaborn, Google Colab.

## How to run

1. Open the notebook in Google Colab (or Jupyter).
2. Upload `Resume.csv` and update the file path in the data-loading cell.
3. Run all cells in order.

## Roadmap

- [ ] Compare train vs test accuracy and investigate overfitting
- [ ] Try other models and tune the TF-IDF settings
- [ ] Predict the category of a new, unseen resume
- [ ] Serve the model with a FastAPI `/predict` endpoint
- [ ] Containerize with Docker

## Author

Malik Zain ([@Malik-Zain-555](https://github.com/Malik-Zain-555))
