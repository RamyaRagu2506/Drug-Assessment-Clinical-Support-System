# Patient Sentiment Classification for Drug Assessment

Classifying patient drug reviews as positive, negative or neutral, as a first step towards a clinical decision support system for pharmacovigilance and patient-centred care. Presented at NCTS'23 (Best Paper, session recognition).

## Overview
Patients describe their experiences with medication in online reviews. This project:
1. labels the sentiment of each review with two lexicon-based tools (TextBlob and VADER),
2. compares how well traditional ML models and an LSTM can learn those labels, and
3. adds a simple usefulness-weighted score to rank drugs per condition.

## Dataset
[UCI Drug Review Dataset (Drugs.com)](https://archive.ics.uci.edu/dataset/462/drug+review+dataset+drugs+com): about 215,000 patient reviews, each with the drug name, condition, a 1–10 rating and a "useful" count.

## Preprocessing
- Lowercasing; removal of HTML entities (e.g. `&#039;`), special characters and non-ASCII characters
- Stop-word removal and stemming
- Removal of empty reviews
- Exploratory analysis of ratings, conditions, drugs and review length

## Sentiment labelling
| Tool | Rule | Positive | Negative | Neutral |
|---|---|---|---|---|
| TextBlob | polarity > 0 / = 0 / < 0 | 101,064 | 42,899 | 16,429 |
| VADER | compound ≥ 0.05 / ≤ −0.05 / in between | 77,081 | 73,372 | 9,939 |

Both label sets are imbalanced, with neutral as the minority class, so results are reported with macro F1 as well as accuracy.

## Models
- **Traditional ML:** Logistic Regression, Linear SVC and a hard-voting ensemble (LR + SVC) on TF-IDF features; 80/20 train/test split.
- **Deep learning:** LSTM with a learned embedding layer (vocabulary 50,000, sequence length 250, embedding size 100, 100 LSTM units, dropout 0.2, 5 epochs, batch size 64, early stopping); 90/10 train/test split.

## Results
| Model | Features | TextBlob: accuracy (macro F1) | VADER: accuracy (macro F1) | VADER: neutral recall |
|---|---|---|---|---|
| Logistic Regression | TF-IDF | 0.95 (0.94) | 0.92 (0.84) | 0.49 |
| Linear SVC | TF-IDF | **0.97 (0.96)** | 0.93 (0.84) | 0.48 |
| Ensemble (LR + SVC) | TF-IDF | 0.96 (0.96) | 0.93 (0.85) | 0.51 |
| LSTM | Learned embeddings | 0.96 (–) | 0.93 **(0.86)** | **0.61** |

## Key findings
- **With TextBlob labels**, all models reach high agreement; Linear SVC performs best (0.97).
- **With VADER labels**, accuracy levels off at 0.92–0.93 for every model, but the traditional models miss about half of the neutral reviews. The LSTM raises neutral recall to 0.61 and reaches the highest macro F1, which suggests word order helps with in-between sentiment.
- **Accuracy alone hides this difference.** On imbalanced data, macro F1 and per-class recall tell the real story.

## Limitations
- **Labels come from lexicon tools,** so accuracy measures agreement with TextBlob or VADER, not true patient sentiment.
- **Sentiment labels were computed on preprocessed text** (stop words removed, stemmed), which can weaken sentiment cues.
- **The LSTM used a separate pipeline and split,** so its results are not directly comparable with the TF-IDF models.
- **The TF-IDF vectoriser was fitted before the train/test split** (a minor leak).
- **The conference paper's LSTM figures** (0.97 for TextBlob, 0.96 for VADER) could not be reproduced. The table above reports the re-run results.

## Future work
- Use the patient ratings (1–10) as independent sentiment labels
- Evaluate all models on the same split, with the vectoriser fitted on training data only
- Compare with transformer models (e.g. BERT) and other embeddings (e.g. Word2Vec)
- Extend the drug-ranking score with condition-specific information

## Repository structure
- `<preprocessing notebook>`: cleaning, feature engineering and sentiment labelling
- `<logistic regression notebook>`, `<linear SVC notebook>`, `<ensemble notebook>`: traditional models
- `<LSTM notebook>`: LSTM training, evaluation and drug ranking

## How to run
1. Install the requirements: `pip install -r requirements.txt`
2. Download the dataset from the UCI link above into `data/`
3. Run the preprocessing notebook first, then the model notebooks

## Tech stack
Python · pandas · scikit-learn · TensorFlow/Keras · NLTK · TextBlob · VADER · Matplotlib · Seaborn

## Conference
Presented at **NCTS'23**, National Conference on Technology for the Society, October 2023, SRMIST, Chennai, India. Awarded **Best Paper (session recognition)**.
