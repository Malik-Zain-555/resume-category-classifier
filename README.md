# Resume Category Classifier

Classifies a resume into one of 24 job categories (HR, CHEF, BANKING, INFORMATION-TECHNOLOGY, TEACHER, and more). Two approaches are built and compared: **TF-IDF + Logistic Regression** and **sentence embeddings + Logistic Regression**. The TF-IDF version also says **"UNSURE"** when the model is not confident, and the project looks inside the model to see which words drive each prediction.

**Status:** both models built and compared. The TF-IDF model is served through a FastAPI endpoint and runs in Docker. Serving the embeddings model is still planned (see [Roadmap](#roadmap)).

## Problem

Given the text of a resume, predict its job category. This is a multi-class text classification problem with 24 classes.

## Dataset

`Resume.csv` with the columns used here:

| Column | Meaning |
|---|---|
| `Resume_str` | Raw resume text |
| `Category` | Job category label (24 unique values) |

2,484 resumes in total (1,987 train / 497 test). The dataset file is not included in this repo; add your own source link here. The notebooks load it from Google Drive in Colab.

## Repository contents

| File | What it is |
|---|---|
| `resume-category-classifier.ipynb` | TF-IDF version: cleaning, `C` tuning, word vs character n-grams, model weights, `predict_resume` with `UNSURE` |
| `02_embeddings.ipynb` | Embeddings version and comparison with TF-IDF |
| `resume_pipeline.pkl` | Saved TF-IDF + Logistic Regression pipeline |
| `app/main.py` | FastAPI app with a `/predict` endpoint |
| `requirements.txt` | Exact library versions the API needs |
| `Dockerfile` | Recipe to run the API in a container |
| `PROJECT_WALKTHROUGH.md` | Full step-by-step explanation of the project |

## Approach

Both versions share the same cleaning (`lowercase`, remove URLs and emails, keep letters only, collapse spaces) and the same split: 80/20, stratified by category, `random_state=42`.

**TF-IDF version**
1. `TfidfVectorizer` (English stop words, `max_features=3000`, `ngram_range=(1, 2)`) and `LogisticRegression(C=100)` in one scikit-learn `Pipeline`.
2. `C` and word vs character n-grams compared with 5-fold cross-validation on the training set.
3. The model's `coef_` matrix (24 x 3000) is used to list top words per category.
4. `predict_resume(text)` cleans the text and returns the category, or `UNSURE` if the top probability is below 0.50.

**Embeddings version**
1. Each resume is turned into 384 numbers with the pretrained `all-MiniLM-L6-v2` sentence-transformer (no training needed for this step).
2. `LogisticRegression` is trained on those 384 numbers; `C` was chosen by 5-fold cross-validation (`C=10`).

## Results

| | TF-IDF (word) | Embeddings |
|---|---|---|
| 5-fold CV (training set) | 69.0% ± 3.3 | **70.6% ± 3.5** |
| Test accuracy | 71.8% | **74.4%** |
| Short web-developer resume (`frontend`, `mongo`) | CONSULTANT (wrong), ~16% confidence | **INFORMATION-TECHNOLOGY**, 53% |
| Short dental-doctor resume | wrong, 23% | HEALTHCARE, 22% |
| Clear chef / HR summaries | 99.9% / 99.8% | 98.7% / 98.8% |
| Can inspect which words drive a prediction | **yes** (`coef_`) | no |

The test set has only 497 resumes and the CV results vary by about ±3.5 points, so the 2-3 point lead of embeddings is small. Embeddings are slightly better and handle unseen words better; TF-IDF is much easier to explain. `C=100` for TF-IDF was first chosen by looking at test accuracy, so its test number is slightly optimistic; the cross-validation numbers are the more conservative comparison.

Character n-gram TF-IDF (3-5) scored 68.2% test and 64.3% ± 3.9 in CV, so it was not kept, although it did classify the short web-developer resume correctly.

### What the TF-IDF model learned

Many top words match intuition:

| Category | Example top words |
|---|---|
| HR | `payroll`, `recruitment`, `employee relations`, `human resources` |
| CHEF | `kitchen`, `culinary`, `sous`, `menu` |
| BANKING | `teller`, `mortgage`, `lending`, `loan` |
| TEACHER | `lesson`, `curriculum`, `classroom`, `students` |

## Limitations I found

- **Odd top words.** Some top words have no obvious link to their category (for example `dental` under ARTS). `dental` appears in only 4 ARTS resumes but in 10 HEALTHCARE and 10 HR resumes, yet its ARTS weight is about 1.29 (roughly 37% of the largest ARTS weight). This looks like a pattern picked up from a small group of resumes, not a real signal.
- **Words the TF-IDF model never saw.** `frontend` appears in only 8 resumes and `mongo` in 5, so they are not in the vocabulary. Raising `max_features` to 10,000 did not help because the words are rare in the data itself. Embeddings handle this better because they capture meaning.
- **Short, vague text.** A 3-line resume with hobbies mixed in (games, sports, cooking) fooled both models (BUSINESS-DEVELOPMENT and CONSTRUCTION, about 20% confidence). Removing the hobbies made the embeddings model answer HEALTHCARE.
- **Low confidence.** The `UNSURE` threshold of 0.50 is a starting value chosen from a few examples, not a tuned one.
- **Not yet checked:** how much of a long resume the embedding model actually reads.

## Running the API

The TF-IDF model is served with FastAPI. `POST /predict` takes a resume text and returns a category and a confidence. If the top probability is below 0.50, the category is `"UNSURE"`.

**With Docker**

```bash
docker build -t resume-api .
docker run -p 8000:8000 resume-api
```

**Without Docker** (Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; on Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Then open `http://127.0.0.1:8000/docs`, choose `POST /predict`, click *Try it out*, and send:

```json
{"text": "Dedicated chef with experience in kitchen operations, menu planning and culinary techniques."}
```

Example response:

```json
{"category": "CHEF", "confidence": 0.99}
```

The versions in `requirements.txt` match the ones used to save `resume_pipeline.pkl` (scikit-learn 1.6.1). Loading the file with a different scikit-learn version may warn or behave differently.

## Using the saved TF-IDF model

```python
import joblib
pipe = joblib.load("resume_pipeline.pkl")
# the pipeline expects text already passed through the cleaning function
pipe.predict_proba([clean_resume(text)])
```

The cleaning function is not inside the pipeline, so it must be applied before prediction. The embeddings model is not saved yet.

## Tech stack

Python, pandas, NumPy, scikit-learn, sentence-transformers, joblib, FastAPI, Docker, Google Colab.

## Roadmap

- [x] Baseline TF-IDF + Logistic Regression, tuned with cross-validation
- [x] Inspect model weights and investigate odd words
- [x] Save the pipeline and add a `predict_resume` function with an `UNSURE` threshold
- [x] Compare against sentence embeddings
- [x] Serve the TF-IDF model with a FastAPI `/predict` endpoint (with the `UNSURE` rule)
- [x] Containerize with Docker
- [ ] Serve the embeddings model too (larger install, needs the sentence-transformers library)
- [ ] Tune the `UNSURE` threshold on more resumes

## Author

Malik Zain ([@Malik-Zain-555](https://github.com/Malik-Zain-555))
