# Resume Category Classifier: Full Walkthrough

This document explains the whole project from the first line of code to where it stands now: **what** was done, **how** it was done, and **why** each decision was made. It is written for two readers: my future self returning after a break, and anyone who wants to understand the project without prior knowledge of machine learning.

> **How this was built:** a learning project made with guidance from AI assistants (Claude and Gemini). I ran every experiment myself, checked the results, and wrote down what each one showed.

---

## 1. The idea in one minute

**Goal:** given the text of a resume, predict which job category it belongs to (24 categories such as HR, CHEF, BANKING, INFORMATION-TECHNOLOGY, TEACHER).

**What was built:**
1. A baseline: **TF-IDF + Logistic Regression**.
2. Experiments to improve and understand it (tuning, character n-grams, cross-validation).
3. A look *inside* the model to see which words it relies on, which exposed its weaknesses.
4. A saved model and a `predict_resume` function that says **"UNSURE"** when it is not confident.
5. A second approach using **sentence embeddings**, compared fairly with the first.
6. The TF-IDF model turned into a real service: a **FastAPI** endpoint (`POST /predict`) packaged in a **Docker** container, so anyone can run it with two commands.

**Headline results**

| | TF-IDF (word) | Embeddings |
|---|---|---|
| 5-fold cross-validation | 69.0% ± 3.3 | **70.6% ± 3.5** |
| Test accuracy | 71.8% | **74.4%** |
| Web-developer resume with unseen words (`frontend`, `mongo`) | wrong, ~16% confidence | **correct, 53% confidence** |

Embeddings are slightly better, especially on words the model has never seen. TF-IDF is easier to explain because its weights can be read directly. The TF-IDF model is the one currently served through the API (section 6).

---

## 2. The problem and the data

**Why this problem?** Companies receive huge numbers of resumes. Automatically sorting them by field is a typical, practical text classification task.

**Type of problem:** *multi-class text classification*. The input is free text, and the output is one of 24 labels. (An earlier project in this series, customer churn, was binary: only Yes/No, and the data was in table form.)

**Data:** `Resume.csv`, 2,484 resumes. Columns used:

| Column | Meaning |
|---|---|
| `Resume_str` | the resume text |
| `Category` | the job category (the label to predict) |

**Why an audit first?** Before building anything, I checked the shape, missing values, and how many resumes each category has. Some categories are small (in the test set, BPO has only 4 resumes and AUTOMOBILE has 7), which later explained why the model struggled with them.

---

## 3. The big picture

Every step below is one stage of a pipeline:

```
raw resume text
   -> clean the text
   -> split into train / test
   -> turn text into numbers (TF-IDF or embeddings)
   -> train a classifier (Logistic Regression)
   -> evaluate honestly
   -> inspect mistakes and improve
   -> save the model, wrap it in a function
```

The same loop applies to almost any ML project: **baseline, measure, look at errors, change one thing, measure again.**

---

## 4. Step by step: what, how, why

### 4.1 Cleaning the text

**What:** lowercase everything, remove URLs and emails, keep only letters, collapse repeated spaces.

```python
def resume_cleaner(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\.\S+', " ", text)
    text = re.sub(r'\S+@\S+', " ", text)
    text = re.sub(r'[^a-zA-Z\s]', " ", text)
    text = re.sub(r'\s+', " ", text).strip()
    return text
```

**Why:** "Python", "python" and "PYTHON" should be one word, and URLs, emails and symbols add noise. Cleaning shrinks the vocabulary and makes patterns easier to find.

**A side effect to know:** removing everything that is not a letter also turns `front-end` into `front end` and `mongo-db` into `mongo db`. A bug I found and fixed later: spaces accidentally placed *inside* a regular expression made the URL pattern miss matches.

### 4.2 Train/test split

**What:** 80% of resumes for training (1,987) and 20% for testing (497).

```python
train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)
```

**Why a test set?** A model that is graded on the same resumes it learned from can just memorise them. The test set simulates resumes it has never seen.

**Why `stratify=y`?** It keeps the category proportions the same in train and test, so small categories are not accidentally missing from one of them.

**Why `random_state=42`?** So the split (and every result) is reproducible.

### 4.3 Turning text into numbers with TF-IDF

Computers cannot read words, only numbers. **TF-IDF** gives every word in every resume a score made of two parts:

- **TF (term frequency):** how often the word appears in *this* resume. More often means a higher score.
- **IDF (inverse document frequency):** how *rare* the word is across all resumes. A word found in almost every resume ("experience") says little. A word found in few resumes ("recruitment") is a strong clue.

**A small example I worked out by hand.** Three tiny documents:

1. `python data science python`
2. `python web development`
3. `hr hiring recruitment`

| Word in document 1 | Score | Why |
|---|---|---|
| python | 0.73 | appears twice (high TF), but is in 2 of 3 documents (lower IDF) |
| data, science | 0.48 | appear once, but are rare (only in 1 of 3 documents) |

In document 2, `python`, `web` and `development` each appear once, yet `python` scores lower (0.47 vs 0.62) because it is the common one. If a fourth document containing `python` is added, `python` becomes even more common and its score in document 1 drops (to about 0.67).

Two details worth remembering:
- Scores are **normalised** so each document's vector has length 1. This stops long resumes from automatically getting bigger numbers.
- TF-IDF **ignores word position.** It only counts and compares.

**Settings used:** `stop_words='english'` (drop words like "the"), `ngram_range=(1, 2)` (single words and two-word phrases like "human resources"), `max_features=3000` (keep only the 3,000 most useful features, so each resume becomes a row of 3,000 numbers).

### 4.4 `fit`, `transform`, `fit_transform`

| Method | What it does | In TfidfVectorizer |
|---|---|---|
| `fit` | learns something from data | learns the vocabulary and IDF values |
| `transform` | applies what was learned | turns text into numbers |
| `fit_transform` | both at once | used on **training data only** |

```python
x_train_tfidf = tfidf.fit_transform(x_train)  # learn + convert
x_test_tfidf  = tfidf.transform(x_test)       # convert only
```

**Why not `fit` on the test set?** If the vectorizer also learned from test resumes, the test set would no longer be unseen. That mistake is called **data leakage**, and it makes results look better than they are. The same rule applies to a brand-new resume at prediction time: only `transform`.

### 4.5 The classifier: Logistic Regression

**What it does:** for every category it learns one **weight per word**. A large positive weight means "this word pushes toward this category", a negative weight pushes away. For a new resume it adds up the weighted scores and also returns a probability for every category.

**Why this model?** It is simple, fast, and its weights can be read. That makes it a good baseline and lets me see *why* it predicts what it predicts.

The learned weights live in `coef_`. Its shape is **(24, 3000)**: 24 categories (rows, in the order of `classes_`) by 3,000 features (columns, in the order of `get_feature_names_out()`). Entry `coef_[i][j]` is the weight of word `j` for category `i`.

### 4.6 Looking inside the model

Sorting each row of `coef_` shows the 10 words that push hardest toward each category:

```python
for name, coef in zip(model.classes_, model.coef_):
    top = np.argsort(coef)[-10:]
    print(name, tfidf.get_feature_names_out()[top])
```

**Words that make sense:** HR: `payroll`, `recruitment`, `employee relations`. CHEF: `kitchen`, `culinary`, `sous`. BANKING: `teller`, `mortgage`, `lending`.

**Words that do not:** ARTS had `dental`, SALES had `high school`, AUTOMOBILE had `tibco`, BPO had `india` and `cisco`.

### 4.7 Investigating the odd word "dental"

Accuracy alone would never have shown this. I checked:

| Check | Result |
|---|---|
| Resumes containing "dental" | HEALTHCARE 10, HR 10, ADVOCATE 7, ACCOUNTANT 6, **ARTS only 4** |
| ARTS weight for "dental" | 1.29 (the largest ARTS weight was 3.48) |
| HEALTHCARE weight for "dental" | 0.31 |

So `dental` is not a real signal for ARTS. The model latched onto a pattern in a tiny group of resumes. That is a **spurious correlation** and a sign of **overfitting**.

### 4.8 Tuning `C`: overfitting and underfitting

`C` controls how much freedom Logistic Regression has. Small `C` means strong restriction. Large `C` means it may fit the training data almost perfectly.

| C | Train | Test |
|---|---|---|
| 0.1 | 63.0% | 60.4% |
| 1 | 81.0% | 66.2% |
| 10 | 99.6% | 70.6% |
| 100 | 100% | **71.8%** |
| 1000 | 100% | 71.2% |

**The lesson:** a big train-test gap does not automatically mean a bad model. The most "overfitted" settings had the *best* test accuracy, while the most restricted setting (`C=0.1`) was **underfitting**: poor even on its own training data. Judge a model by test (or cross-validation) accuracy, not by the gap alone.

**A caution:** choosing `C` by looking at test accuracy slightly "uses up" the test set, so that number is a little optimistic. This is why cross-validation was added.

### 4.9 Cross-validation and `Pipeline`

**Cross-validation:** the training data is split into 5 parts. The model trains on 4 and is checked on 1, five times, rotating which part is held out. The average (mean) is the expected accuracy, and the standard deviation (std) shows how much it moves between folds. This compares settings **without touching the test set**.

**`Pipeline`:** puts the vectorizer and the model into one object, so in every fold the vectorizer learns only from that fold's training part. This prevents leakage and later lets the whole thing be saved as a single file.

```python
word_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(C=100, max_iter=1000, random_state=42))
])
```

| Variant | CV mean ± std |
|---|---|
| Word TF-IDF | **68.95% ± 3.2** |
| Character n-gram TF-IDF | 64.27% ± 3.9 |

### 4.10 The problem of unseen words

I wrote a short resume of my own: web development, frontend, backend, MongoDB, MySQL. The word model predicted **CONSULTANT** with only about 16% confidence. Checking the vocabulary showed why:

| Word | In the model's vocabulary? | Resumes containing it |
|---|---|---|
| `frontend` | no | 8 |
| `mongo` | no | 5 |
| `web`, `database`, `mysql` | yes | many |

TF-IDF keeps only the most common words, and these words are rare in this dataset. Increasing `max_features` from 3,000 to 10,000 did **not** help, because they are rare in the data itself. A model only knows what its training data showed it.

**Attempted fix: character n-grams.** Instead of whole words, the vectorizer learns 3-5 letter fragments (`fro`, `ront`, `nten`), so `frontend`, `front end` and typos look similar. This did classify my web-developer resume correctly (INFORMATION-TECHNOLOGY), but overall accuracy was lower (68.2% test, 64.3% CV), so it was not kept.

### 4.11 Probabilities and the "UNSURE" rule

`predict_proba` returns a probability for each category. Examples:

| Resume | Prediction | Top probability |
|---|---|---|
| Clear chef summary | CHEF | 99.9% |
| Clear HR summary | HR | 99.8% |
| Short dental-doctor text with hobbies | BUSINESS-DEVELOPMENT | **23.4%** |

When the model has no real idea, its top probability is low. So the final function refuses to answer in that case:

```python
def predict_resume(text):
    cleaned = clean_resume(text)
    probs = pipe.predict_proba([cleaned])[0]
    best = np.argmax(probs)
    if probs[best] < 0.50:
        return "UNSURE", probs[best]
    return pipe.classes_[best], probs[best]
```

**Why 0.50?** It cleanly separates the confident cases (about 99%) from the vague one (about 23%). It is a starting value chosen from a few examples, not a tuned threshold. A product that says "I'm not sure" is more trustworthy than one that always answers.

### 4.12 Saving the model

```python
joblib.dump(word_pipe, "resume_pipeline.pkl")
loaded = joblib.load("resume_pipeline.pkl")
```

Because the vectorizer is inside the pipeline, one file contains everything. The cleaning function is *not* inside it and must be applied before prediction.

---

## 5. Second approach: sentence embeddings

### 5.1 What an embedding is

An embedding model turns text into a fixed-length list of numbers (here **384**), so that texts with similar **meaning** end up close together. It is already trained on huge amounts of text, so no training is needed for this step. Model used: `all-MiniLM-L6-v2`.

A quick experiment showed the difference. Cosine similarity (1 means very close, near 0 means unrelated):

| Pair | Similarity |
|---|---|
| frontend developer vs web developer | **0.66** |
| web developer vs teacher | 0.36 |
| frontend developer vs teacher | 0.18 |
| frontend developer vs chef | 0.11 |

TF-IDF could not see the link between "frontend" and "web developer", because they are different words. Embeddings capture it.

### 5.2 Using embeddings for the classifier

Each resume becomes 384 numbers (instead of 3,000 TF-IDF scores): `x_train_emb` has shape (1987, 384). The same Logistic Regression is then trained on them. There is no `fit` for the embedding step, since the model is pretrained.

```python
x_train_emb = model.encode(x_train.tolist())
x_test_emb  = model.encode(x_test.tolist())
```

### 5.3 Results

| C | CV mean ± std |
|---|---|
| 1 | 68.04% ± 3.2 |
| **10** | **70.61% ± 3.5** |
| 100 | 68.19% ± 2.2 |
| 1000 | 67.59% ± 2.0 |

`C=10` was chosen from cross-validation (not from the test set). Test accuracy: **74.4%** (train 90.2%).

| Check | TF-IDF | Embeddings |
|---|---|---|
| Web-dev resume (`frontend`, `mongo`) | CONSULTANT, ~16% | **INFORMATION-TECHNOLOGY, 53%** |
| Dental text with hobbies | BUSINESS-DEVELOPMENT, 23% | CONSTRUCTION, 21% (both wrong) |
| Dental text without hobbies | wrong | HEALTHCARE, 22% (right, but unsure) |
| Chef / HR | 99.9% / 99.8% | 98.7% / 98.8% |

**Trade-off:** embeddings are a little better and handle unseen words, but their 384 numbers have no names, so I cannot read which word drove a prediction the way I can with TF-IDF's `coef_`.

### 5.4 A reusable function, and its speed

The trained embeddings classifier was saved with `joblib` (`embeddings_clf.pkl`), and a function `predict_embeddings(text)` was written in the same way as the TF-IDF one: clean the text, encode it, get probabilities, return `UNSURE` below 0.50. It returns plain Python values (`('CHEF', 0.987)`), not numpy objects, because an API cannot send numpy types as JSON.

Speed matters for a service, so I timed it: about **10.7 ms per resume** (roughly 90 resumes per second). Embeddings are therefore not too slow for an API. The real cost is installation size: the `sentence-transformers` library brings a large dependency (PyTorch), which makes a Docker image much bigger. That is why the first API serves the lighter TF-IDF model.

---

## 6. From notebook to a working service: FastAPI and Docker

### 6.1 Why do this at all?

A model inside a notebook can only be used by someone who opens the notebook and runs the cells. A website, a mobile app or another program cannot do that. To make the model *usable*, it needs a stable "door" that other programs can knock on: an **API**.

### 6.2 What an API is (the restaurant picture)

Think of a restaurant. You (a client) do not walk into the kitchen. You give your order to a waiter, and the waiter brings back the food. An API is the waiter:

- The **request** is the order: "here is a resume text".
- The **server** is the kitchen: it runs the model.
- The **response** is the dish: "category CHEF, confidence 0.99".

Key words:

| Word | Meaning |
|---|---|
| **Endpoint** | one specific address on the server, such as `/predict` |
| **GET** | "give me something" (what a browser address bar does) |
| **POST** | "here is some data, process it" (needed to send a resume) |
| **JSON** | the text format used to exchange data, e.g. `{"text": "I am a chef"}` |
| **Status code** | `200` means OK, `422` means "your data has the wrong shape", `405` means "wrong method" |

### 6.3 FastAPI and uvicorn

**FastAPI** is the library used to describe the endpoints. **uvicorn** is the program that actually runs the application and listens for requests. The smallest possible app:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "hello"}
```

- `app = FastAPI()` creates the application.
- `@app.get("/")` says: when someone sends a GET request to `/`, run the function below.
- The function returns a dictionary, and FastAPI turns it into JSON automatically.

It is started with:

```bash
python -m uvicorn app.main:app --reload
```

The part `app.main:app` always has the form **`folder.file:variable`**: the folder `app`, the file `main.py` (without `.py`), and the variable `app` created by `app = FastAPI()`. `--reload` restarts the server automatically whenever the code changes.

FastAPI also builds a free test page at **`/docs`**, where each endpoint can be tried from the browser. No extra tool is needed.

### 6.4 Receiving text: the `Resume` class

```python
from pydantic import BaseModel

class Resume(BaseModel):
    text: str

@app.post("/predict")
def predict(resume: Resume):
    return {"length": len(resume.text)}
```

`class Resume` describes the shape of the request: one field called `text` that must be a string. FastAPI checks every incoming request against it and answers with a `422` error if the shape is wrong. This was tested first with a harmless function that only counts characters (`"I'm a chef"` gave `{"length": 10}`), to confirm that the text really arrives before the model is involved.

### 6.5 Connecting the model

The final `app/main.py` does five things:

```python
pipe = joblib.load("resume_pipeline.pkl")          # 1. load the model once, at start-up

@app.post("/predict")
def predict(resume: Resume):
    cleaned = clean_resume(resume.text)            # 2. same cleaning as in training
    probs = pipe.predict_proba([cleaned])[0]       # 3. probability for each category
    best = np.argmax(probs)                        # 4. position of the highest one
    if probs[best] < 0.5:                          # 5. the UNSURE rule
        return {"category": "UNSURE", "confidence": float(probs[best])}
    return {"category": str(pipe.classes_[best]), "confidence": float(probs[best])}
```

Why each choice:

- **Load the model once, outside the function.** Loading takes time. Doing it inside `predict` would repeat it for every request.
- **The same cleaning function must be used.** The model was trained on cleaned text. Different cleaning means the model sees different input than it learned from.
- **`probs[best]`, not `best`.** `best` is only the *position* of the winning category (for example 10). The confidence is `probs[best]`. An early version compared the position with 0.5 by mistake, so the `UNSURE` rule never triggered.
- **`float(...)` and `str(...)`.** numpy numbers cannot be turned into JSON directly, so they are converted to plain Python types.

Results through the API (same as in the notebook):

| Request text | Response |
|---|---|
| Long chef summary | `CHEF`, 0.999 |
| `"I'm a chef"` | `CHEF`, 0.99999 |
| `"string"` (Swagger's default text) | `UNSURE`, 0.42 |

A reminder from this: `"I'm a chef"` gets 99.999% confidence because the word "chef" alone is very strong. **High confidence does not mean the answer is correct**, it only means the model is sure of itself. The `UNSURE` rule only catches vague text, not confident mistakes.

### 6.6 Virtual environment and `requirements.txt`

A **virtual environment** (`.venv`) is a private folder of Python libraries for one project, so that projects do not disturb each other.

`requirements.txt` lists the exact libraries and versions the project needs:

```
fastapi==0.162.0
uvicorn==0.54.0
pydantic==2.14.1
scikit-learn==1.6.1
numpy==2.5.1
joblib==1.5.3
```

`==` means "exactly this version". This matters especially for scikit-learn: a `.pkl` file saved with one version may warn or misbehave when loaded with another. The versions listed are the ones that the saved pipeline was tested with. The list comes from `python -m pip freeze`, which prints what is installed.

### 6.7 Docker

**The problem Docker solves:** the API only worked on my laptop because I had installed the right versions by hand. On another computer, it would fail. Docker packs the app, Python and all libraries into one **container**, a sealed box that runs the same way anywhere.

Three words:

| Word | Meaning |
|---|---|
| **Dockerfile** | the recipe: step-by-step instructions to build the box |
| **Image** | the finished, frozen box built from the recipe |
| **Container** | a running copy of an image |

The recipe used:

```dockerfile
FROM python:3.12-slim                          # start from a small Linux box with Python 3.12

WORKDIR /code                                  # the working folder inside the box

COPY requirements.txt .                        # copy the library list first
RUN pip install --no-cache-dir -r requirements.txt   # install the libraries

COPY app ./app                                 # then copy the code
COPY resume_pipeline.pkl .                     # and the model

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]   # command to run on start
```

Why the order matters: Docker remembers each step (a **layer**). The libraries are installed *before* the code is copied, because the code changes often and the libraries rarely do. When only the code changes, Docker reuses the remembered layers. The first build took about 680 seconds (mostly installing libraries); after fixing one line it took 2 seconds, with every step marked `CACHED`.

Why `--host 0.0.0.0`: `127.0.0.1` means "only from inside this machine/box". Inside a container, that would block the outside world. `0.0.0.0` means "accept connections from anywhere".

Running it:

```bash
docker build -t resume-api .             # build the image and name it resume-api
docker run -p 8000:8000 resume-api       # start a container
```

`-p 8000:8000` connects port 8000 of the laptop (left) to port 8000 of the container (right), so the browser can reach the server inside the box.

The check that it works: the same long chef text gave the **identical** confidence, 0.9990010734144485, to the last digit, in the container and on the laptop. The same model, the same libraries and the same cleaning were running.

### 6.8 Problems I met and what each one taught

| What happened | Cause | Lesson |
|---|---|---|
| `No module named 'app'` | the command was run from *inside* the `app` folder | run it from the project root |
| `Import string must be in format <module>:<attribute>` | wrote `app.main` or `main.py` | the form is `folder.file:variable` |
| `No module named 'fastapi'` although `(.venv)` was active | the first `uvicorn` was a *global* one; I had also typed `unicorn` instead of `uvicorn` when installing | use `python -m uvicorn` and `python -m pip`, which always use the active environment |
| `No module named 'pydanctic'` | a typo | read the **last line** of the error: it names the problem |
| `GET /predict` gave `405` and `/doc` gave `404` | `/predict` accepts POST only; the page is `/docs` | wrong method or wrong address, not a broken app |
| The API kept returning `{"length": ...}` | `@app.post("/predict")` was defined twice and the first one won (FastAPI warned about "Duplicate Operation ID") | one endpoint, one definition |
| `UNSURE` never appeared | `best < 0.5` compared a position with a probability | compare `probs[best]` |
| It worked with `(.venv)` missing `scikit-learn` | the server was secretly running on the global Python | check `pip freeze` inside the environment, and install the exact versions there |
| `git add` stopped with `pathspec ... did not match` | `requirements.txt` did not exist yet, so nothing was added | read git's message: it says which file is missing |
| `docker build` succeeded but the Dockerfile still had `____` | `CMD` is not executed at build time, only when the container starts | a successful build is not a working container: always run it |

**The main habit:** the last line of an error message usually states the cause. Read it before changing anything.

### 6.9 What this section achieved

- The model can be called by any program, with a plain HTTP request.
- It refuses to answer when it is not confident.
- It runs identically on any machine that has Docker.
- Everything needed to reproduce it is in the repository: `app/main.py`, `requirements.txt`, `Dockerfile` and `resume_pipeline.pkl`.

---

## 7. Conclusion

- Embeddings beat TF-IDF by about 2-3 points on both cross-validation and test, and handled unseen words better.
- The gap is small. The test set has only 497 resumes and the cross-validation spread is about ±3.5, so I would not call it a decisive win.
- Both models struggle with short, vague resumes and small categories.
- TF-IDF remains valuable because it is explainable, which is how the `dental` problem was found.

---

## 8. What I learned

1. **Accuracy hides problems.** Looking at the model's weights found problems accuracy never showed.
2. **Be suspicious of numbers that look too good.** At one point a test score of 96.8% appeared, 26 points above cross-validation. Re-running the cell cleanly gave the real number, 74.4%. It had come from a stale notebook state (cells run out of order).
3. **Always restart and run all.** A notebook only deserves trust if it produces the same results from top to bottom in one go.
4. **A model only knows its data.** Rare or missing words cannot be recovered by tuning.
5. **Change one thing at a time**, then measure.
6. **Choose the metric and the comparison carefully.** Comparing models fairly means giving each the same tuning effort and the same split.
7. **A system should know when it does not know.** The UNSURE rule is cheap and valuable.
8. **Read the last line of an error first.** In the API work, almost every error named its own cause.
9. **A model in a notebook is not yet a product.** Serving it forces new questions: environments, versions, cleaning, request shape, packaging.
10. **Confidence is not correctness.** A model can be 99.999% sure and wrong, or right and unsure.
11. **"It runs" is not "it is verified".** A Docker build can succeed with a broken start command. Run it and test it.

---

## 9. A related experiment: fake vs real news

A separate side project used the same tools on a fake/real news dataset and reached 98.8% accuracy. Inspecting the weights and testing with my own texts showed the model had learned the **writing style of the source** (for example, the word "Reuters" appeared in 99.8% of real articles), not whether the news was true. It failed 2 of 3 handwritten tests. It is the same lesson as `dental`, at a larger scale: **high accuracy does not prove the model learned the right thing.**

---

## 10. Reproducing the project

**The notebooks**

1. Put `Resume.csv` in a Google Drive folder and update the path in the notebooks.
2. Run `resume-category-classifier.ipynb` (TF-IDF version) with *Runtime, Restart and run all*.
3. Run `02_embeddings.ipynb` (needs `pip install sentence-transformers`).

**The API with Docker** (needs Docker installed and running)

```bash
docker build -t resume-api .
docker run -p 8000:8000 resume-api
```

Then open `http://127.0.0.1:8000/docs`, choose `POST /predict`, *Try it out*, and send `{"text": "your resume text"}`.

**The API without Docker** (Python 3.12)

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; Mac/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

| File | Contents |
|---|---|
| `resume-category-classifier.ipynb` | cleaning, TF-IDF model, tuning, weights, character n-grams, `predict_resume` |
| `02_embeddings.ipynb` | embeddings, the comparison with TF-IDF, saved embeddings classifier and its speed test |
| `resume_pipeline.pkl` | saved TF-IDF + Logistic Regression pipeline (used by the API) |
| `app/main.py` | the FastAPI application |
| `requirements.txt` | exact library versions |
| `Dockerfile` | recipe for the container |
| `README.md` | short project summary |
| `PROJECT_WALKTHROUGH.md` | this document |

---

## 11. What is next

- [x] Serve the TF-IDF model with FastAPI (`POST /predict`, with the `UNSURE` rule)
- [x] Package it with Docker
- [ ] Serve the embeddings model as well (larger install because of PyTorch; the classifier is already saved as `embeddings_clf.pkl` in the notebook work)
- [ ] Check how much of a long resume the embedding model actually reads (not yet verified)
- [ ] Test on more resumes, ideally written by other people, to tune the `UNSURE` threshold
- [ ] Add a few automatic tests for the API
- [ ] Publish the container (for example on a free hosting service) so the API has a public address

---

## 12. Glossary

| Term | Meaning |
|---|---|
| **Classification** | predicting a category for an input |
| **Feature** | one number the model looks at (here a word's TF-IDF score) |
| **TF-IDF** | word score = how often in this text x how rare across all texts |
| **n-gram** | a sequence of 1, 2, ... words (or letters) treated as one feature |
| **Vocabulary** | the set of words/features the vectorizer knows |
| **Embedding** | a fixed-length list of numbers capturing the meaning of a text |
| **Train / test set** | data to learn from / data held back for an honest exam |
| **Data leakage** | information from the test set influencing training |
| **Overfitting** | great on training data, worse on new data |
| **Underfitting** | poor even on training data |
| **Regularisation (`C`)** | restriction on how freely the model can fit; small `C` = more restriction |
| **Cross-validation** | repeated train/check rounds inside the training set to compare settings |
| **Pipeline** | preprocessing and model chained into one object |
| **`predict_proba`** | the probability the model gives to each category |
| **Spurious correlation** | a pattern that appears in the data but is not a real cause |
| **API** | a program that waits for requests and sends back answers, so other programs can use it |
| **Endpoint** | one specific address of an API, such as `/predict` |
| **GET / POST** | request types: GET asks for something, POST sends data to be processed |
| **JSON** | text format for exchanging data, like `{"text": "..."}` |
| **FastAPI** | Python library for building APIs |
| **uvicorn** | the server program that runs a FastAPI app |
| **Pydantic model** | a class that describes the shape of a request and validates it |
| **Virtual environment (`.venv`)** | a private set of libraries for one project |
| **`requirements.txt`** | list of the exact libraries and versions a project needs |
| **Docker** | a tool that packs an app and everything it needs into one runnable box |
| **Dockerfile / image / container** | the recipe / the frozen box built from it / a running copy of the box |
| **Port** | a numbered "door" on a machine; `-p 8000:8000` connects a door on the laptop to a door in the container |
| **Layer cache** | Docker's memory of unchanged build steps, which makes rebuilds fast |
