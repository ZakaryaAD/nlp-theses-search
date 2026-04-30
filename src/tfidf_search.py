import argparse
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DISPLAY_COLUMNS = [
    "title",
    "title_en",
    "abstract",
    "year",
    "discipline",
    "subjects",
    "institution",
    "url",
]


def load_data(input_path: str) -> pd.DataFrame:
    """
    Load the processed theses dataset.

    Expected input:
        CSV file with at least a 'text' column.

    Output:
        pandas DataFrame with missing text replaced by empty strings.
    """
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input file must contain a 'text' column.")

    df["text"] = df["text"].fillna("").astype(str)
    return df


def build_tfidf_matrix(texts: pd.Series, max_features: int = 50000):
    """
    Fit a TF-IDF vectorizer and transform documents into sparse vectors.

    Input:
        texts: Series of documents, shape (n_documents,)

    Output:
        vectorizer: fitted TfidfVectorizer
        matrix: sparse TF-IDF matrix, shape (n_documents, n_features)

    Mental model:
        Each document becomes a sparse vector.
        Most entries are zero because each thesis uses only a small part of the vocabulary.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=2,
    )

    matrix = vectorizer.fit_transform(texts)
    return vectorizer, matrix


def search_tfidf(
    query: str,
    df: pd.DataFrame,
    vectorizer: TfidfVectorizer,
    matrix,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Search the most relevant theses for a query using TF-IDF + cosine similarity.

    Input:
        query: user query as a string
        df: theses dataframe
        vectorizer: fitted TF-IDF vectorizer
        matrix: document TF-IDF matrix, shape (n_documents, n_features)
        top_k: number of results to return

    Output:
        DataFrame with the top_k theses and their similarity scores.
    """
    query_vector = vectorizer.transform([query])
    scores = cosine_similarity(query_vector, matrix).ravel()

    top_indices = scores.argsort()[::-1][:top_k]

    available_columns = [col for col in DISPLAY_COLUMNS if col in df.columns]
    results = df.iloc[top_indices][available_columns].copy()
    results.insert(0, "score", scores[top_indices])

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/processed/theses_clean.csv")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--output", default=None)

    args = parser.parse_args()

    df = load_data(args.input)
    vectorizer, matrix = build_tfidf_matrix(df["text"], args.max_features)

    results = search_tfidf(
        query=args.query,
        df=df,
        vectorizer=vectorizer,
        matrix=matrix,
        top_k=args.top_k,
    )

    print(results.to_string(index=False))

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results.to_csv(output_path, index=False)
        print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()