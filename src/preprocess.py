import argparse
import re
from pathlib import Path

import pandas as pd


TEXT_COLUMNS = ["title", "title_en", "abstract", "discipline", "subjects", "institution"]

REQUIRED_COLUMNS = [
    "id",
    "title",
    "title_en",
    "abstract",
    "year",
    "discipline",
    "subjects",
    "institution",
    "status",
    "url",
]

def clean_text(text: object) -> str:
    """
    Nettoie légèrement un champ texte.

    On garde un nettoyage minimal :
    - suppression des retours à la ligne ;
    - suppression des espaces multiples ;
    - conversion en string.

    On évite un nettoyage trop agressif car les embeddings transformer
    fonctionnent mieux avec du texte naturel.
    """
    if pd.isna(text):
        return ""

    text = str(text)
    text = text.replace("\n", " ")
    text = text.replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def ensure_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """
    Vérifie que toutes les colonnes attendues existent.
    Si une colonne manque, elle est créée vide.
    """
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            df[col] = ""

    return df


def build_search_text(df: pd.DataFrame) -> pd.Series:
    """
    Construit le texte utilisé par les modèles de recherche.

    Chaque ligne correspond à un document :
    titre + résumé + discipline.
    """
    parts = []

    for col in TEXT_COLUMNS:
        parts.append(df[col].fillna("").map(clean_text))

    search_text = parts[0] + ". " + parts[1] + ". " + parts[2]
    return search_text.map(clean_text)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prépare les données pour la recherche d'information.
    """
    df = ensure_columns(df, REQUIRED_COLUMNS)

    for col in REQUIRED_COLUMNS:
        df[col] = df[col].fillna("").map(clean_text)

    df["text"] = build_search_text(df)

    # On retire les lignes inutilisables.
    df = df[df["id"].str.len() > 0].copy()
    df = df[df["title"].str.len() > 0].copy()
    df = df[df["text"].str.len() > 20].copy()

    df = df.drop_duplicates(subset=["id"])
    df = df.reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/theses_raw.csv")
    parser.add_argument("--output", default="data/processed/theses_clean.csv")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df_raw = pd.read_csv(input_path)
    df_clean = preprocess_dataframe(df_raw)

    df_clean.to_csv(output_path, index=False, encoding="utf-8")

    print(f"Loaded {len(df_raw)} raw theses")
    print(f"Saved {len(df_clean)} clean theses to {output_path}")
    print()
    print(df_clean[["id", "title", "year", "discipline"]].head())


if __name__ == "__main__":
    main()
    