# NLP Theses Search

Semantic and lexical search engine for French PhD theses using data from [theses.fr](https://theses.fr).

The goal of this project is to compare a classical lexical retrieval baseline with a semantic retrieval approach on French academic text data.

Given a natural language query such as:

```text
apprentissage profond pour imagerie médicale
```

or:

```text
deep learning for medical imaging
```

the system returns the most relevant PhD theses with their title, English title when available, abstract, year, discipline, institution, theses.fr URL, and similarity score.

---

## Project motivation

Search engines based only on lexical matching can fail when the query and the document use different words to express similar ideas.

For example, a thesis about medical image analysis may be relevant to a query about deep learning for radiology, even if the exact keywords do not perfectly match.

This project compares two retrieval approaches:

1. **TF-IDF + cosine similarity**, used as a classical lexical baseline.
2. **Transformer-based sentence embeddings**, used for semantic retrieval.

The objective is to evaluate whether transformer embeddings improve retrieval quality compared with a sparse lexical baseline on enriched theses.fr metadata and abstracts.

---

## Dataset

The data is collected from the public theses.fr search API:

```text
https://theses.fr/api/v1/theses/recherche/
```

The current dataset contains:

| Item | Value |
|---|---:|
| Unique theses after deduplication | 18,242 |
| Non-empty abstracts after enrichment | 16,831 |
| Abstract coverage | 92.27% |
| Processed columns | 11 |

The data was collected using broad queries related to artificial intelligence, machine learning, data science, statistics, computer science, and related fields.

Each thesis contains the following fields:

| Column | Description |
|---|---|
| `id` | Thesis identifier on theses.fr |
| `title` | Main thesis title |
| `title_en` | English title when available |
| `abstract` | Abstract extracted from the thesis page when available |
| `year` | Defense year or first registration year |
| `discipline` | Academic discipline |
| `subjects` | Keywords and topics returned by theses.fr |
| `institution` | Institution or university |
| `status` | Thesis status |
| `url` | Link to the thesis page |
| `text` | Final searchable text field |

Raw and processed datasets are not committed to the repository to keep it lightweight. They can be regenerated using the provided scripts.

---

## Repository structure

```text
nlp-theses-search/
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   ├── 01_exploration.ipynb
│   └── 02_tfidf_results.ipynb
│── src/
│   ├── __init__.py
│   ├── collect.py
│   ├── enrich_abstracts.py
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

---

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

---

## Data pipeline

The project follows a three-step data pipeline:

1. metadata collection from the theses.fr API;
2. abstract enrichment from individual thesis pages;
3. preprocessing and construction of the final searchable field.

---

## 1. Metadata collection

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

The collection script deduplicates theses by thesis identifier.

---

## 2. Abstract enrichment

The theses.fr search endpoint provides structured metadata such as titles, disciplines, subjects, institutions and dates. However, abstracts are not included directly in the search endpoint.

To enrich the dataset with abstracts, the project includes a separate script that visits each thesis detail page and extracts the visible abstract from the HTML.

For a quick test on 20 theses:

```bash
python -m src.enrich_abstracts --limit 20
```

For the first 1,000 theses:

```bash
python -m src.enrich_abstracts --limit 1000 --sleep-seconds 0.3
```

To enrich the full dataset:

```bash
python -m src.enrich_abstracts --limit -1 --sleep-seconds 0.3
```

This creates:

```text
data/raw/theses_raw_enriched.csv
```

The current enriched dataset contains 16,831 non-empty abstracts out of 18,242 theses, corresponding to a 92.27% abstract coverage.

---

## 3. Preprocessing

To clean the enriched data and build the searchable text field:

```bash
python -m src.preprocess --input data/raw/theses_raw_enriched.csv --output data/processed/theses_clean.csv
```

This creates:

```text
data/processed/theses_clean.csv
```

The final `text` column is built by concatenating:

```text
title + title_en + abstract + discipline + subjects + institution
```

This representation makes retrieval robust to missing abstracts: if an abstract is unavailable, the thesis can still be retrieved through its title, English title, discipline, subjects and institution.

---

## Exploratory data analysis

The notebook:

```text
notebooks/01_exploration.ipynb
```

analyzes the processed dataset.

Main EDA results:

| Metric | Value |
|---|---:|
| Number of theses | 18,242 |
| Number of columns | 11 |
| Missing abstracts | 1,411 |
| Abstract coverage | 92.27% |
| Missing subjects | 4,027 |
| Missing English titles | 1,395 |
| Mean text length | 2,068.83 characters |
| Median text length | 1,910 characters |

The most frequent discipline is `Informatique`, with 3,734 theses.

---

## TF-IDF retrieval

The first retrieval baseline represents each thesis as a sparse TF-IDF vector and ranks theses by cosine similarity with the query.

Run a TF-IDF search with:

```bash
python -m src.tfidf_search --query "apprentissage profond pour imagerie médicale" --top-k 10
```

Example with output saved as CSV:

```bash
python -m src.tfidf_search --query "apprentissage profond pour imagerie médicale" --top-k 10 --output data/processed/tfidf_results_deep_learning_medical_imaging.csv
```

The TF-IDF vectorizer uses:

| Parameter | Value |
|---|---|
| Lowercasing | yes |
| Accent stripping | unicode |
| N-grams | unigrams and bigrams |
| Minimum document frequency | 2 |
| Maximum features | 50,000 |
| Similarity | cosine similarity |

TF-IDF is simple, fast and interpretable. It works well when the query and the document share explicit terms, but it can retrieve partial matches when only one part of the query is present.

---

## TF-IDF result analysis

The notebook:

```text
notebooks/02_tfidf_results.ipynb
```

evaluates the TF-IDF baseline on 8 queries.

French queries:

```text
apprentissage profond pour imagerie médicale
traitement automatique des langues pour documents historiques
apprentissage automatique pour risque de crédit
vision par ordinateur pour diagnostic médical
modèles statistiques pour la finance
```

English queries:

```text
deep learning for medical imaging
natural language processing for historical documents
machine learning for credit risk
```

The English queries are included to test the sensitivity of lexical retrieval to query language.

A manual inspection was performed on the top 5 results for each query, for a total of 40 inspected results.

| Label | Count | Rate |
|---|---:|---:|
| Relevant | 11 | 27.5% |
| Partial | 25 | 62.5% |
| Irrelevant | 4 | 10.0% |

The results show that TF-IDF rarely returns completely unrelated documents, but most retrieved results are only partially relevant. This confirms that TF-IDF is a strong lexical baseline, while also motivating the use of semantic embeddings.

---

## Planned semantic retrieval

The next step is to implement transformer-based semantic retrieval.

The planned method is:

1. encode each thesis `text` field with a pretrained Sentence Transformer model;
2. encode the user query with the same model;
3. rank theses by cosine similarity in the embedding space;
4. compare results with the TF-IDF baseline.

This method is expected to better handle vocabulary mismatch and query language variation.

---

## Evaluation strategy

Since the dataset does not provide ground-truth relevance labels, the evaluation combines:

- qualitative analysis of top retrieved theses for selected queries;
- manual relevance judgments on a small set of representative queries;
- comparison of TF-IDF and embedding rankings;
- overlap between top-k results;
- discussion of success and failure cases.

Manual relevance labels use three categories:

| Label | Meaning |
|---|---|
| `relevant` | The thesis clearly matches the full query intent |
| `partial` | The thesis matches only part of the query |
| `irrelevant` | The thesis is not related to the query intent |

---

## Report

The final report follows the NeurIPS template and includes:

1. Introduction
2. Dataset and data collection
3. Exploratory data analysis
4. Methods
5. Experimental setup
6. TF-IDF retrieval results
7. Semantic retrieval results
8. Discussion
9. Conclusion

The report is limited to 10 pages. If necessary, detailed examples and extended result tables will be moved to an appendix or kept in the notebooks.

---

## Current status

Implemented:

- repository structure;
- metadata collection from theses.fr;
- abstract enrichment from thesis pages;
- preprocessing pipeline;
- EDA notebook;
- TF-IDF search script;
- TF-IDF result analysis notebook;
- first manual relevance analysis.

Next steps:

- update and push the repository;
- implement transformer embedding retrieval;
- compare TF-IDF and embedding results;
- finalize the report.

---

## About

This project was developed for the ENSAE Machine Learning for NLP course.

It also serves as practical training for applied machine learning engineering workflows: data collection, preprocessing, retrieval models, evaluation, reproducible scripts, notebooks and GitHub-ready project structure.