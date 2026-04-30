import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer


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
        CSV file with a non-empty 'text' column.

    Output:
        DataFrame where the 'text' column is converted to string.
    """
    df = pd.read_csv(input_path)

    if "text" not in df.columns:
        raise ValueError("Input file must contain a 'text' column.")

    df["text"] = df["text"].fillna("").astype(str)
    return df


def prepare_texts(df: pd.DataFrame, max_chars: int = 4000) -> list[str]:
    """
    Prepare texts before embedding.

    Why truncate?
        Some theses have very long abstracts.
        Sentence Transformers have a maximum token length anyway.
        Truncating very long strings avoids wasting time on text that will be cut later.

    Input:
        df: DataFrame with a 'text' column
        max_chars: maximum number of characters kept per document

    Output:
        List of strings, length = number of documents
    """
    return df["text"].fillna("").astype(str).str.slice(0, max_chars).tolist()


def get_device() -> str:
    """
    Return the best available device.
    """
    return "cuda" if torch.cuda.is_available() else "cpu"


def compute_embeddings(
    texts: list[str],
    model_name: str,
    batch_size: int,
    device: str,
) -> np.ndarray:
    """
    Encode all documents with a pretrained Sentence Transformer.

    Input:
        texts: list of documents, length n_documents
        model_name: Hugging Face / Sentence Transformers model name
        batch_size: number of documents encoded at once
        device: 'cuda' or 'cpu'

    Output:
        embeddings: numpy array of shape (n_documents, embedding_dim)

    Important:
        normalize_embeddings=True means cosine similarity can be computed
        as a simple dot product.
    """
    model = SentenceTransformer(model_name, device=device)

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return embeddings.astype("float32")


def save_embeddings(
    embeddings: np.ndarray,
    embeddings_path: Path,
    metadata: dict,
) -> None:
    """
    Save embeddings and metadata.
    """
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_path, embeddings)

    metadata_path = embeddings_path.with_suffix(".json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_embeddings(embeddings_path: Path) -> np.ndarray:
    """
    Load precomputed embeddings.
    """
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")

    return np.load(embeddings_path)


def search_embeddings(
    query: str,
    df: pd.DataFrame,
    embeddings: np.ndarray,
    model_name: str,
    device: str,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Search the most relevant theses using dense embeddings.

    Input:
        query: user query
        df: theses DataFrame
        embeddings: document embeddings, shape (n_documents, embedding_dim)
        model_name: same model used for document embeddings
        device: 'cuda' or 'cpu'
        top_k: number of results

    Output:
        DataFrame with top_k theses and similarity scores.
    """
    model = SentenceTransformer(model_name, device=device)

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    scores = embeddings @ query_embedding[0]
    top_indices = scores.argsort()[::-1][:top_k]

    available_columns = [col for col in DISPLAY_COLUMNS if col in df.columns]
    results = df.iloc[top_indices][available_columns].copy()
    results.insert(0, "score", scores[top_indices])

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        default="data/processed/theses_clean.csv",
        help="Path to processed theses CSV.",
    )
    parser.add_argument(
        "--embeddings-path",
        default="data/processed/embeddings/theses_embeddings.npy",
        help="Path where embeddings are saved or loaded.",
    )
    parser.add_argument(
        "--model-name",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Sentence Transformer model name.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=4000,
        help="Maximum number of characters kept per document before encoding.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional query to test search after embedding computation/loading.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of retrieved results.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Force recomputation of embeddings even if file exists.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save search results as CSV.",
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    embeddings_path = Path(args.embeddings_path)

    df = load_data(input_path)
    device = get_device()

    print(f"Loaded dataset: {df.shape}")
    print(f"Device: {device}")
    print(f"Model: {args.model_name}")
    print(f"Embeddings path: {embeddings_path}")

    if embeddings_path.exists() and not args.recompute:
        print("Loading existing embeddings...")
        embeddings = load_embeddings(embeddings_path)
    else:
        print("Computing embeddings...")
        texts = prepare_texts(df, max_chars=args.max_chars)

        embeddings = compute_embeddings(
            texts=texts,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=device,
        )

        metadata = {
            "model_name": args.model_name,
            "input_path": str(input_path),
            "n_documents": int(embeddings.shape[0]),
            "embedding_dim": int(embeddings.shape[1]),
            "batch_size": args.batch_size,
            "max_chars": args.max_chars,
            "device": device,
        }

        save_embeddings(
            embeddings=embeddings,
            embeddings_path=embeddings_path,
            metadata=metadata,
        )

        print(f"Saved embeddings to {embeddings_path}")
        print(f"Saved metadata to {embeddings_path.with_suffix('.json')}")

    if embeddings.shape[0] != len(df):
        raise ValueError(
            f"Mismatch: embeddings contain {embeddings.shape[0]} rows, "
            f"but dataset contains {len(df)} rows."
        )

    print(f"Embeddings shape: {embeddings.shape}")

    if args.query:
        results = search_embeddings(
            query=args.query,
            df=df,
            embeddings=embeddings,
            model_name=args.model_name,
            device=device,
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