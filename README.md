# 📩 SMS Spam vs Ham Classification using Word2Vec + Random Forest

A machine learning project that classifies SMS messages as **spam** or **ham** (legitimate) by converting text into dense semantic vectors with **Word2Vec** and training a **Random Forest** classifier on top of them.

---

## 📌 Overview

Traditional bag-of-words approaches treat every word as an independent token and ignore meaning. This project instead represents each SMS as the **average of its word embeddings**, so messages with similar meaning end up close together in vector space. A tuned Random Forest then separates spam from ham.

**Pipeline:**

```
Raw SMS → Tokenize (simple_preprocess) → Word2Vec (300-d) → Average word vectors → Random Forest → Spam / Ham
```

---

## 📂 Dataset

- **Source:** [SMS Spam Collection Dataset](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) (UCI / Kaggle)
- **Size:** 5,572 messages
- **Format:** tab-separated, two columns: `label` (`ham` / `spam`) and `messages`
- **Class balance:** imbalanced, with ham heavily outnumbering spam (test split: 1,207 ham vs 186 spam)

Place the file as `SMSSpamCollection.csv` in your working directory (the notebook reads it from `/content/` when run on Google Colab).

---

## 🛠️ Tech Stack

| Area | Tools |
|---|---|
| Language | Python 3 |
| NLP / Embeddings | Gensim (Word2Vec), NLTK |
| ML | scikit-learn (RandomForestClassifier, GridSearchCV) |
| Data | Pandas, NumPy |
| Utilities | tqdm |

---

## 🔬 Methodology

1. **Exploring embeddings:** Loaded Google's pretrained `word2vec-google-news-300` to inspect 300-dimensional word vectors and nearest neighbours (e.g. `king` → `queen`, `monarch`, `prince`).
2. **Preprocessing:** Tokenized and lowercased each SMS with `gensim.utils.simple_preprocess`.
3. **Training Word2Vec on the SMS corpus:** `vector_size=300`, `min_count=2`, `epochs=10`, trained directly on the 5,572 messages.
4. **Sentence vectors:** Each message is represented by the **mean of its word vectors** (zero vector if no known words).
5. **Labels:** `ham → 1`, `spam → 0`.
6. **Split:** 75% train / 25% test (`random_state=42`).
7. **Model:** `RandomForestClassifier(n_estimators=200)` as the baseline.
8. **Tuning:** `GridSearchCV` (5-fold CV) over `n_estimators ∈ {50, 100, 250, 500}`.

---

## 📊 Results

**Baseline Random Forest (200 trees)**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Spam (0) | 0.91 | 0.89 | 0.90 | 186 |
| Ham (1) | 0.98 | 0.99 | 0.98 | 1207 |

- Accuracy: **0.97**
- ROC-AUC: **0.939**

**After GridSearchCV tuning**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Spam (0) | 0.92 | 0.89 | 0.90 | 186 |
| Ham (1) | 0.98 | 0.99 | 0.99 | 1207 |

- Accuracy: **0.97**
- ROC-AUC: **0.940**

Tuning gave a small but consistent improvement over the baseline.

---

## 🚀 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>
```

### 2. Install dependencies

```bash
pip install gensim nltk scikit-learn pandas numpy tqdm jupyter
```

### 3. Add the dataset

Download the SMS Spam Collection and save it as `SMSSpamCollection.csv` in the project folder. Update the path in the notebook if you are not using Colab:

```python
df = pd.read_csv('SMSSpamCollection.csv', sep='\t', names=['label', 'messages'])
```

### 4. Run the notebook

```bash
jupyter notebook Spam_Ham_Classification_Project_ANN.ipynb
```

> **Note:** The pretrained `word2vec-google-news-300` model is ~1.6 GB and downloads on first use. It is only used for the embedding exploration; the classifier uses the Word2Vec model trained on the SMS data.

---

## 📁 Project Structure

```
├── Spam_Ham_Classification_Project_ANN.ipynb   # Full workflow
├── SMSSpamCollection.csv                       # Dataset (add manually)
└── README.md
```

---

## 🔭 Future Improvements

- Use the **pretrained Google News vectors** (or GloVe / FastText) for sentence embeddings. A Word2Vec model trained on only ~5.5k short texts gives weak neighbours (e.g. `king` returns unrelated words), so pretrained vectors will likely generalize better.
- Handle **class imbalance** with `class_weight='balanced'`, SMOTE, or threshold tuning to improve spam recall.
- Tune more hyperparameters (`max_depth`, `min_samples_split`, `max_features`) and report the best parameters found.
- Add a stratified split and cross-validated metrics (precision/recall/F1, PR-AUC).
- Compare against TF-IDF + Logistic Regression / Naive Bayes baselines.
- Implement an actual **Artificial Neural Network** (Keras/PyTorch) on the averaged embeddings, and compare it with the Random Forest.
- Deploy as a small web app (Streamlit / Flask) for live SMS prediction.

---

## 🙋 Author

**Ekangsh**
B.Tech, Production & Industrial Engineering, NIT Jamshedpur
Interested in data science and analytics.

Feel free to ⭐ the repo if you found it useful!