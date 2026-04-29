# NLP Theses Search

Semantic search engine for French PhD theses using data from [theses.fr](https://theses.fr).

The goal of this project is to compare classical lexical retrieval methods with modern semantic retrieval methods on French academic text data.

Given a natural language query such as:

```text
deep learning for medical imaging
```

the system returns the most relevant PhD theses with their title, abstract, year, discipline, institution, URL, and similarity score.

## Project motivation

Search engines based only on lexical matching can fail when the query and the document use different words to express similar ideas.

For example, a thesis about *medical image analysis* may be relevant to a query about *deep learning for radiology*, even if the exact keywords do not perfectly match.

This project compares two retrieval approaches:

1. **TF-IDF + cosine similarity**, used as a classical lexical baseline.
2. **Transformer-based sentence embeddings**, used for semantic retrieval.

The objective is to evaluate whether transformer embeddings improve retrieval quality on French PhD thesis metadata.

## Dataset

The data is collected from the public theses.fr API.

The current dataset contains around 18,000 French PhD theses collected from several broad queries related to artificial intelligence, machine learning, data science, statistics, computer science, and related fields.

Each thesis contains the following fields:

| Column | Description |
|---|---|
| `id` | Thesis identifier on theses.fr |
| `title` | Thesis title |
| `abstract` | Thesis abstract when available |
| `year` | Defense year |
| `discipline` | Academic discipline |
| `institution` | Institution or university |
| `url` | Link to the thesis page |

Raw and processed datasets are not committed to the repository to keep it lightweight. They can be regenerated using the collection script.

## Repository structure

```text
nlp-theses-search/
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│── src/
│   ├── __init__.py
│   ├── collect.py
│   ├── preprocess.py
│   ├── tfidf_search.py
│   ├── embedding_search.py
│   ├── evaluation.py
│   └── utils.py
│── report/
│── app.py
│── requirements.txt
│── README.md
│── .gitignore
```

## Installation

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Data collection

To collect theses from the theses.fr API:

```bash
python -m src.collect
```

A larger dataset can be collected with:

```bash
python -m src.collect --queries "intelligence artificielle" "apprentissage automatique" "deep learning" "traitement automatique des langues" "vision par ordinateur" "science des données" "modélisation statistique" "réseaux de neurones" "fouille de données" "informatique" --max-results-per-query 2500 --page-size 100 --sleep-seconds 0.1
```

This creates:

```text
data/raw/theses_raw.csv
```

## Preprocessing

To clean the raw data and build the searchable text field:

```bash
python -m src.preprocess
```

This creates:

```text
data/processed/theses_clean.csv
```

The `text` column is built by concatenating the title, abstract, and discipline.

## Planned methods

### 1. TF-IDF retrieval

The first baseline represents each thesis as a sparse TF-IDF vector and ranks theses by cosine similarity with the query.

This method is simple, fast, interpretable, and commonly used as a strong baseline for document retrieval.

### 2. Transformer embedding retrieval

The second method encodes theses and queries into dense semantic vectors using a pretrained Sentence Transformer model.

Documents are ranked by cosine similarity in the embedding space.

This method is expected to better capture semantic similarity, especially when the query and the thesis abstract use different but related terms.

## Evaluation

Since the dataset does not directly provide labeled relevance judgments, the evaluation will combine:

- qualitative analysis of top retrieved theses for selected queries;
- comparison of TF-IDF and embedding rankings;
- overlap between top-k results;
- manual relevance judgments on a small set of representative queries.

Possible example queries:

```text
deep learning for medical imaging
natural language processing for historical documents
machine learning for credit risk
computer vision for medical diagnosis
statistical models for finance
```

## Report

The final report will follow the NeurIPS template and will include:

1. Introduction
2. Dataset and exploratory analysis
3. Related work
4. Methods
5. Experimental setup
6. Results
7. Discussion
8. Conclusion
