# Resume Category Classifier

Classifies a resume into one of 24 job categories (HR, CHEF, BANKING, INFORMATION-TECHNOLOGY, TEACHER, and more) using TF-IDF and Logistic Regression. It also says **"UNSURE"** when the model is not confident, and the project looks inside the trained model to see which words drive each prediction.

**Status:** TF-IDF version complete. An embeddings comparison, a FastAPI endpoint and Docker are planned (see [Roadmap](#roadmap)).

## Problem

Given the text of a resume, predict its job category. This is a multi-class text classification problem with 24 classes.

## Dataset

`Resume.csv` with the columns used here:

| Column | Meaning |
|---|---|
| `Resume_str` | Raw resume text |
| `Category` | Job category label (24 unique values) |

2,484 resumes in total (1,987 train / 497 test). The dataset file is not included in this repo; add your own source link here. The notebook loads it from Google Drive in Colab.

## Approach

1. **Audit:** shape, null counts, category distribution.
2. **Cleaning:** lowercase; remove URLs and emails; keep letters only; collapse extra whitespace (`clean_resume`).
3. **Split:** 80/20 train/test, stratified by category (`random_state=42`).
4. **Features:** `TfidfVectorizer` with English stop words, `max_features=3000`, `ngram_range=(1, 2)`.
5. **Model:** `LogisticRegression(C=100, max_iter=1000)`, wrapped with the vectorizer in one scikit-learn `Pipeline`.
6. **Model selection:** tuned `C`, compared word TF-IDF with character n-gram TF-IDF using 5-fold cross-validation on the training set.
7. **Interpretation:** the model's `coef_` matrix (24 x 3000) gives a weight for every word in every category; sorting each row shows the top words per category.
8. **Inference:** `predict_resume(text)` cleans the text, gets class probabilities, and returns `UNSURE` if the top probability is below 0.50.

## Results

| Setup | Score |
|---|---|
| Word TF-IDF, `C=100` (final) | **71.8%** test accuracy, **69.0% ± 3.2** 5-fold CV |
| Character n-grams (3-5), `C=100` | 68.2% test accuracy, 64.3% ± 3.9 5-fold CV |

`C=100` was chosen by looking at test accuracy, so the test number is slightly optimistic; the cross-validation figure is the more conservative estimate.

### What the model learned

Many top words match intuition:

| Category | Example top words |
|---|---|
| HR | `payroll`, `recruitment`, `employee relations`, `human resources` |
| CHEF | `kitchen`, `culinary`, `sous`, `menu` |
| BANKING | `teller`, `mortgage`, `lending`, `loan` |
| TEACHER | `lesson`, `curriculum`, `classroom`, `students` |

### Limitations I found

- **Odd top words.** Some top words have no obvious link to their category (for example `dental` under ARTS). `dental` appears in only 4 ARTS resumes but in 10 HEALTHCARE and 10 HR resumes, yet its ARTS weight is about 1.29 (roughly 37% of the largest ARTS weight). This looks like a pattern picked up from a small group of resumes, not a real signal.
- **Words the model never saw.** `frontend` appears in only 8 resumes and `mongo` in 5, so they are not in the 3,000-feature vocabulary. A short web-developer resume was therefore predicted as CONSULTANT with very low confidence (about 16%). Raising `max_features` to 10,000 did not fix this because the words are rare in the data itself. Character n-grams did classify that resume as INFORMATION-TECHNOLOGY, but they scored lower overall, so they are not the final model.
- **Short, vague text.** A 3-line resume mixing a dental career with hobbies got only about 23% confidence, so the `UNSURE` threshold returns "UNSURE" instead of a wrong category. Clear chef and HR summaries scored above 99%.
- The 0.50 threshold is a starting value chosen from three examples, not a tuned one.

## Using the saved model

```python
import joblib
pipe = joblib.load("models/resume_pipeline.pkl")
# the pipeline expects text already passed through clean_resume()
pipe.predict_proba([clean_resume(text)])
```

The cleaning function is not inside the pipeline, so it must be applied before prediction.

## Tech stack

Python, pandas, NumPy, scikit-learn, seaborn, joblib, Google Colab.

## Roadmap

- [x] Baseline TF-IDF + Logistic Regression
- [x] Tune `C`; compare word vs character n-grams with cross-validation
- [x] Inspect model weights and investigate odd words
- [x] Save the pipeline and add a `predict_resume` function with an `UNSURE` threshold
- [ ] Compare against sentence embeddings (to see if they handle unseen words better)
- [ ] Serve the model with a FastAPI `/predict` endpoint
- [ ] Containerize with Docker

## Author

Malik Zain ([@Malik-Zain-555](https://github.com/Malik-Zain-555))
