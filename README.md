# NLP Theses Search

Lexical and semantic search engine for French PhD theses using data from [theses.fr](https://theses.fr).

This project compares two information retrieval approaches on French academic text data:

1. **TF-IDF + cosine similarity** as a lexical baseline.
2. **Multilingual Sentence Transformer embeddings** as a semantic retrieval method.

Given a natural language query such as:

```text
apprentissage profond pour imagerie médicale
```

or:

```text
deep learning for medical imaging
```

the system returns the most relevant PhD theses with their title, English title when available, abstract, year, discipline, institution, theses.fr URL and similarity score.

---

## Project motivation

A lexical search engine works well when the query and the document share the same words. However, it can fail when the same idea is expressed with different vocabulary.

For example, a query about deep learning for medical imaging may be relevant to theses mentioning neural networks, image registration, segmentation, radiology or computer-aided diagnosis, even if the exact query terms are not all present.

The goal of this project is to study whether semantic embeddings improve retrieval quality compared with a sparse lexical baseline on enriched theses.fr titles and abstracts.

---

## Dataset

The data is collected from the public theses.fr search API:

```text
https://theses.fr/api/v1/theses/recherche/
```

The current processed dataset contains:

| Item | Value |
|---|---:|
| Unique theses after deduplication | 18,242 |
| Non-empty abstracts after enrichment | 16,831 |
| Abstract coverage | 92.27% |
| Processed columns | 11 |

The data was collected using broad queries related to artificial intelligence, machine learning, data science, statistics, computer science and related fields.

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
| `text` | Final searchable text field built from title, English title and abstract |

Raw and processed datasets are not committed to the repository to keep it lightweight. They can be regenerated using the scripts in `src/`.

---

## Repository structure

```text
nlp-theses-search/
│── data/
│   ├── raw/
│   └── processed/
│── notebooks/
│   ├── 01_exploration.ipynb
│   ├── 02_tfidf_results.ipynb
│   └── 03_embedding_results.ipynb
│── src/
│   ├── __init__.py
│   ├── collect.py
│   ├── enrich_abstracts.py
│   ├── preprocess.py
│   ├── tfidf_search.py
│   ├── embedding_search.py
│── report/
│── requirements.txt
│── README.md
│── .gitignore
```

---

## Main files

| File | Role |
|---|---|
| `src/collect.py` | Collects thesis metadata from the theses.fr search API. |
| `src/enrich_abstracts.py` | Visits individual thesis pages and extracts visible abstracts from HTML. |
| `src/preprocess.py` | Builds the final searchable `text` field from title, English title and abstract. |
| `src/tfidf_search.py` | Implements TF-IDF retrieval with cosine similarity. |
| `src/embedding_search.py` | Implements semantic retrieval using pretrained multilingual Sentence Transformer embeddings. |
| `notebooks/01_exploration.ipynb` | Exploratory data analysis of the processed corpus. |
| `notebooks/02_tfidf_results.ipynb` | Manual analysis of TF-IDF retrieval results. |
| `notebooks/03_embedding_results.ipynb` | Manual analysis of embedding retrieval and comparison with TF-IDF. |
| `report/` | Contains the final report files. |

---

## Requirements

Recommended Python version:

- Python 3.10 or 3.11

Main libraries used:

- pandas
- numpy
- scikit-learn
- requests
- beautifulsoup4
- lxml
- sentence-transformers
- torch
- matplotlib
- jupyter

---

## Installation

Clone the repository:

```bash
git clone https://github.com/ZakaryaAD/nlp-theses-search.git
cd nlp-theses-search
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data pipeline

The project follows a three-step pipeline:

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

To enrich the dataset with abstracts, the project includes a separate script that visits each thesis page and extracts the visible abstract from the HTML.

Quick test:

```bash
python -m src.enrich_abstracts --limit 20
```

First 1,000 theses:

```bash
python -m src.enrich_abstracts --limit 1000 --sleep-seconds 0.3
```

Full dataset:

```bash
python -m src.enrich_abstracts --limit -1 --sleep-seconds 0.3
```

This creates:

```text
data/raw/theses_raw_enriched.csv
```

The current enriched dataset contains 16,831 non-empty abstracts out of 18,242 theses.

---

## 3. Preprocessing

To clean the enriched data and build the final searchable text field:

```bash
python -m src.preprocess --input data/raw/theses_raw_enriched.csv --output data/processed/theses_clean.csv
```

This creates:

```text
data/processed/theses_clean.csv
```

The final `text` column is built by concatenating:

```text
title + title_en + abstract
```

Other fields such as `discipline`, `subjects`, `institution`, `year`, `status` and `url` are kept as metadata for exploratory analysis, result display and possible future filtering. They are not included in the `text` field used in the reported retrieval experiments.

If an abstract is unavailable, the thesis is represented mainly through its title and English title when available, with less textual information.

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

Run a TF-IDF search:

```bash
python -m src.tfidf_search --query "apprentissage profond pour imagerie médicale" --top-k 10
```

Save results as CSV:

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

TF-IDF is computed on the `text` field, which contains the title, English title and abstract.

No manual stopword list is applied. The inverse document frequency component already reduces the weight of very common terms, and keeping stopwords avoids removing potentially useful French academic expressions.

TF-IDF is simple, fast and interpretable. It works well when the query and the document share explicit terms, but it can retrieve partial matches when only one part of the query is present.

---

## Semantic embedding retrieval

The semantic retrieval method is implemented in:

```text
src/embedding_search.py
```

It uses the pretrained multilingual Sentence Transformer model:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

This model maps both theses and queries into dense vectors of dimension 384.

No fine-tuning is performed because the dataset does not contain supervised query-document relevance labels.

The document embeddings are computed once from the same `text` field used by TF-IDF and stored locally as a NumPy matrix:

```text
data/processed/embeddings/theses_embeddings.npy
```

For the current corpus, the embedding matrix has shape:

```text
(18242, 384)
```

On CPU, computing all document embeddings took approximately 13 minutes and 35 seconds.

First run, compute and save document embeddings:

```bash
python -m src.embedding_search --input data/processed/theses_clean.csv --embeddings-path data/processed/embeddings/theses_embeddings.npy --batch-size 32 --max-chars 4000 --recompute
```

Later runs, load existing embeddings and search:

```bash
python -m src.embedding_search --input data/processed/theses_clean.csv --embeddings-path data/processed/embeddings/theses_embeddings.npy --query "apprentissage profond pour imagerie médicale" --top-k 10
```

The expensive step is computing document embeddings. It is done only once. At query time, only the query is encoded and compared with the stored document embeddings using cosine similarity.

---

## Evaluation

Since the dataset does not provide ground-truth relevance labels, the evaluation is qualitative and diagnostic.

The retrieval methods were evaluated on the same 8 queries:

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

The English queries are included to test sensitivity to query language.

Manual relevance labels use three categories:

| Label | Meaning |
|---|---|
| `relevant` | The thesis clearly matches the full query intent |
| `partial` | The thesis matches only part of the query |
| `irrelevant` | The thesis is not related to the query intent |

Manual relevance was assessed on the top 5 results for each query, for a total of 40 inspected results per method.

| Method | Relevant | Partial | Irrelevant | Relevant rate |
|---|---:|---:|---:|---:|
| TF-IDF | 11 | 25 | 4 | 27.5% |
| Embeddings | 22 | 18 | 0 | 55.0% |

Using `relevant = 1`, `partial = 0.5` and `irrelevant = 0`, the diagnostic graded precision over the inspected top-5 results is:

| Method | Strict Precision@5 | Graded Precision@5 | Irrelevant rate |
|---|---:|---:|---:|
| TF-IDF | 0.275 | 0.588 | 10.0% |
| Embeddings | 0.550 | 0.775 | 0.0% |

The detailed analysis is provided in the notebooks and in the final report.

---

## Main findings

The TF-IDF baseline is strong when query terms appear explicitly in the indexed fields. However, it often retrieves partial matches when only one part of the query is present.

In this small manual evaluation, the embedding method retrieves more fully relevant results than TF-IDF. It is better at capturing the global semantic intent of queries, especially when a query combines a method and an application domain.

The overlap@5 between TF-IDF and embeddings is generally low. This suggests that lexical and semantic retrieval capture different signals and could be combined in a hybrid retrieval system.

BM25 was not implemented in order to keep the comparison focused and reproducible before the deadline. It is a natural extension, together with hybrid retrieval combining lexical and semantic scores.

---

## Report

The final report follows the NeurIPS template and includes:

1. Introduction
2. Dataset and data collection
3. Exploratory data analysis
4. Methods
5. Experimental setup
6. Retrieval results
7. Discussion
8. Conclusion

The report is located in:

```text
report/
```

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
- semantic embedding search script;
- embedding result analysis notebook;
- manual relevance comparison between TF-IDF and embeddings;
- NeurIPS-style report.

Main results:

| Metric | Value |
|---|---:|
| Unique theses collected | 18,242 |
| Non-empty abstracts extracted | 16,831 |
| Abstract coverage | 92.27% |
| TF-IDF relevant rate | 27.5% |
| Embedding relevant rate | 55.0% |

---

## Reproducibility notes

The data and embedding files are excluded from GitHub to keep the repository lightweight.

The following files and folders should not be committed:

```text
data/
.venv/
*.npy
*.pkl
__pycache__/
.ipynb_checkpoints/
```

The final PDF report should be committed in the `report/` folder.

To reproduce the project from scratch:

1. install the dependencies;
2. collect metadata with `src.collect`;
3. enrich abstracts with `src.enrich_abstracts`;
4. preprocess the dataset with `src.preprocess`;
5. run TF-IDF search with `src.tfidf_search`;
6. compute embeddings once with `src.embedding_search`;
7. analyze results in the notebooks.

---

## About

This project was developed for the ENSAE Machine Learning for NLP course.

It also serves as practical training for applied machine learning engineering workflows: data collection, preprocessing, retrieval models, evaluation, reproducible scripts, notebooks and GitHub-ready project structure.