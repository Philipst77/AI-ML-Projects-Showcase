# Latent Dirichlet Allocation (LDA) Topic Modeling on Quora Questions

This project applies **Latent Dirichlet Allocation (LDA)** to a dataset of **Quora questions** to extract latent topics.  
It includes preprocessing, lemmatization, visualization (word clouds & topic-term distributions), and interactive topic exploration.  

---

## 📂 Project Structure

- **`ldascript.py`** – Main script for preprocessing, training LDA, and visualization  
- **`quora_sample.csv`** – Sample dataset of Quora questions  
- **Outputs (generated during execution):**
  - WordCloud visualizations  
  - Topic-term distributions  
  - Topic keywords table  

---

## 🚀 Features

- **Text Preprocessing**
  - Lowercasing, punctuation & number removal  
  - Lemmatization with SpaCy (`en_core_web_sm`)  
  - Stopword removal  

- **Visualizations**
  - **WordCloud** of cleaned and lemmatized text  
  - **pyLDAvis** interactive topic exploration  
  - **Top-N keyword extraction** per topic  

- **LDA Modeling**
  - Implements `sklearn.decomposition.LatentDirichletAllocation`  
  - Configurable parameters:
    - Number of topics (`n_components`)  
    - Learning method (`online`)  
    - Vocabulary size (`max_features`)  

- **Outputs**
  - Dataframe of extracted topic keywords  
  - Topic-word distributions  

---

## 🔧 Requirements

Install dependencies:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn spacy wordcloud pyldavis plotly
python -m spacy download en_core_web_sm
