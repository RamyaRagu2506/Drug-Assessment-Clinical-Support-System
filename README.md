## Developing a Personalized Clinical Support System Utilizing Classification Models for Drug Assessment

## Overview:
A clinical decision support system that tailors therapy recommendations based on aspect-based sentiment analysis of patient drug reviews. 
The system initially classifies patient attitudes toward medications - positive, negative, or neutral and uses them to generate drug recommendations providing an overview on individual's medical conditions and their suggested drug side effects supporting pharmacovigilance and patient-centred clinical decision-making.

## Dataset

UCI Drug Review Dataset — patient reviews paired with medical conditions and drug ratings.

## Methodology 
Two sentiment labelling strategies were compared across multiple classification models: **Labelling Approaches:** 
- **TextBlob** - polarity scores ranging from -1 to 1, subjectivity from 0.0 to 1.0 
- **Vader** - polarity-based: positive (≥ 0.05), negative (≤ -0.05), neutral (between)

**Models Compared:**

| Model | TextBlob Accuracy | Vader Accuracy |
|-------|-------------------|----------------|
| Linear SVC | Best performer (traditional) | Ranks above ensemble by 1% |
| Logistic Regression | Baseline | Baseline |
| Ensemble (LR + SVC) | Competitive | 0.93 |
| LSTM + TF-IDF | **0.97** | **0.96** |

## Key Results 

- LSTM-based modelling outperforms all traditional and ensemble classifiers under both labelling strategies.
- TextBlob labelling yields higher accuracy than Vader in traditional ML settings; LSTM narrows this gap significantly 
- Deep learning captures intricate sentiment patterns in medical language that surface-level lexicon methods miss - **Best accuracy: 0.97 (LSTM + TF-IDF, TextBlob labelling)**

## Limitations & Future Work
- Deep learning models requires higher computation necessitating optimisation
- Identifying condition-specific information in reviews could further improve efficiency
- Promising extension: fine-tuning BERT or other LLMs for richer hidden pattern extraction
- Alternative embeddings (Word2Vec) worth exploring for comparative analysis

## Tech Stack
Python · LSTM · TF-IDF · TextBlob · Vader · Scikit-learn · TensorFlow · NLP Libraries

## Conference
Presented at **NCTS'23** — National Conference on Technology for the Society, October 2023, SRMIST, Chennai, India.

