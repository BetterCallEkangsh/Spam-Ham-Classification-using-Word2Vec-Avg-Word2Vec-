# Spam-Ham Classification using Word2Vec (Avg Word2Vec)

An NLP project that classifies SMS messages as **Spam** or **Ham (Not Spam)** using custom-trained Word2Vec word embeddings and a Random Forest classifier.

## 📂 Project Structure

```
.
├── 28_And_29_-Spam_Ham_Projects_Using_Word2vec_AvgWord2vec.ipynb   # Main notebook
├── smsspamcollection/
│   └── SMSSpamCollection            # Dataset (tab-separated: label, message)
└── README.md
```

## 🧠 Approach

**1. Text Preprocessing**
- Removed non-alphabetic characters using regex
- Lowercased all text
- Lemmatized tokens using NLTK's `WordNetLemmatizer`
- Tokenized cleaned sentences using Gensim's `simple_preprocess`

**2. Word Embeddings**
- Explored pre-trained **Google News Word2Vec (300-dim)** embeddings via `gensim.downloader`
- Trained a **custom Word2Vec model from scratch** on the SMS corpus using `gensim.models.Word2Vec`
- Converted each message into a fixed-length vector using **Average Word2Vec (AvgWord2Vec)** — averaging the embedding vectors of all in-vocabulary words in a sentence

**3. Model Training**
- Built the final feature matrix from averaged word vectors
- Encoded target labels (`spam`/`ham`) using one-hot encoding
- Split data into train/test sets (80/20)
- Trained a **Random Forest Classifier** on the vectorized messages

**4. Evaluation**
- Evaluated using **accuracy score** and **classification report** (precision, recall, F1-score)

## 🛠️ Tech Stack

`Python` `Gensim` `Word2Vec` `NLTK` `Scikit-learn` `Random Forest` `Pandas` `NumPy` `tqdm`

## ⚙️ Installation

```bash
pip install gensim nltk scikit-learn pandas numpy tqdm
```

Also download the required NLTK corpus:
```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
```

## ▶️ Usage

1. Place the `SMSSpamCollection` dataset inside a `smsspamcollection/` folder (dataset available from the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection)).
2. Open and run the notebook `28_And_29_-Spam_Ham_Projects_Using_Word2vec_AvgWord2vec.ipynb` cell by cell.
3. The notebook will:
   - Preprocess and tokenize the SMS text
   - Train a Word2Vec model on the corpus
   - Generate Average Word2Vec feature vectors
   - Train and evaluate a Random Forest spam classifier

## 📊 Results

The model outputs an accuracy score and a full classification report (precision/recall/F1) on the held-out test set, distinguishing spam messages from legitimate (ham) messages.

## 📝 Notes

- The pipeline demonstrates both **pre-trained embeddings** (Google News Word2Vec) and **custom-trained embeddings** on domain-specific (SMS) text, useful for comparing embedding quality on short, informal text.
- Average Word2Vec is a simple but effective way to convert variable-length text into fixed-size numeric vectors for use with traditional ML classifiers.

## 📌 License

This project is open source and available for personal or educational use.
