import argparse
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


API_URL = "https://theses.fr/api/v1/theses/recherche/"


def safe_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """
    Essaie plusieurs clés possibles dans un dictionnaire.

    Utile car les APIs peuvent avoir des noms de champs différents
    selon les endpoints ou les versions.
    """
    for key in keys:
        if key in data and data[key] not in [None, "", []]:
            return data[key]
    return default


def normalize_text_field(value: Any) -> str:
    """
    Convertit un champ texte API en string propre.

    Cas possibles :
    - string simple
    - liste de strings
    - dictionnaire
    - None
    - autre type inattendu
    """
    if value is None:
        return ""

    if isinstance(value, str):
        return value.strip()

    if isinstance(value, list):
        return " ".join(normalize_text_field(x) for x in value if x)

    if isinstance(value, dict):
        return " ".join(normalize_text_field(x) for x in value.values() if x)

    return str(value).strip()


def fetch_page(query: str, start: int, page_size: int) -> dict[str, Any]:
    """
    Récupère une page de résultats depuis l'API theses.fr.

    start = indice du premier résultat.
    page_size = nombre de résultats à récupérer.
    """
    params = {
        "q": query,
        "debut": start,
        "nombre": page_size,
        "tri": "pertinence",
    }

    response = requests.get(
        API_URL,
        params=params,
        timeout=30,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "application/json,text/plain,*/*",
        },
    )

    response.raise_for_status()
    return response.json()


def extract_results(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Récupère la liste de résultats dans la réponse JSON.

    On anticipe plusieurs structures possibles :
    - {"theses": [...]}
    - {"resultats": [...]}
    - {"results": [...]}
    - {"data": [...]}
    - {"response": {"docs": [...]}}
    """
    candidates = [
        "theses",
        "resultats",
        "results",
        "documents",
        "docs",
        "data",
        "items",
    ]

    for key in candidates:
        value = payload.get(key)

        if isinstance(value, list):
            return value

        if isinstance(value, dict):
            for subkey in candidates:
                subvalue = value.get(subkey)
                if isinstance(subvalue, list):
                    return subvalue

    for value in payload.values():
        if isinstance(value, list) and all(isinstance(x, dict) for x in value):
            return value

        if isinstance(value, dict):
            for subvalue in value.values():
                if isinstance(subvalue, list) and all(
                    isinstance(x, dict) for x in subvalue
                ):
                    return subvalue

    return []


def parse_thesis(item: dict[str, Any]) -> dict[str, str]:
    """
    Transforme un résultat brut de l'API en ligne tabulaire propre.

    Sortie cible :
    id, title, abstract, year, discipline, institution, url
    """
    thesis_id = safe_get(
        item,
        [
            "id",
            "nnt",
            "num",
            "numero",
            "identifiant",
            "idThese",
            "these_id",
            "id_these",
        ],
        default="",
    )

    title = safe_get(
        item,
        [
            "titre",
            "title",
            "titrePrincipal",
            "title_s",
            "titres",
        ],
        default="",
    )

    abstract = safe_get(
        item,
        [
            "resume",
            "abstract",
            "resumes",
            "abstracts",
            "resume_fr",
            "resume_en",
            "resumes_fr",
            "resumes_en",
        ],
        default="",
    )

    discipline = safe_get(
        item,
        [
            "discipline",
            "disciplines",
            "domaine",
            "domaines",
            "discipline_s",
        ],
        default="",
    )

    institution = safe_get(
        item,
        [
            "etablissement",
            "etablissements",
            "institution",
            "universite",
            "etablissement_soutenance",
            "etabSoutenance",
        ],
        default="",
    )

    date = safe_get(
        item,
        [
            "dateSoutenance",
            "date_soutenance",
            "date",
            "anneeSoutenance",
            "annee",
        ],
        default="",
    )

    thesis_id = normalize_text_field(thesis_id)
    title = normalize_text_field(title)
    abstract = normalize_text_field(abstract)
    discipline = normalize_text_field(discipline)
    institution = normalize_text_field(institution)

    date_str = normalize_text_field(date)
    year = date_str[:4] if len(date_str) >= 4 and date_str[:4].isdigit() else ""

    return {
        "id": thesis_id,
        "title": title,
        "abstract": abstract,
        "year": year,
        "discipline": discipline,
        "institution": institution,
        "url": f"https://theses.fr/{thesis_id}" if thesis_id else "",
    }


def collect_theses(
    queries: list[str],
    max_results_per_query: int,
    page_size: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    """
    Collecte des thèses pour plusieurs requêtes.

    Mental model :
    - chaque requête donne plusieurs pages ;
    - chaque page contient plusieurs résultats ;
    - chaque résultat brut est converti en ligne propre ;
    - on supprime les doublons par id.
    """
    rows: list[dict[str, str]] = []

    for query in queries:
        print(f"\nCollecte pour la requête : {query}")

        for start in tqdm(range(0, max_results_per_query, page_size)):
            payload = fetch_page(
                query=query,
                start=start,
                page_size=page_size,
            )

            results = extract_results(payload)

            if not results:
                print(f"Aucun résultat trouvé à partir de start={start}")
                break

            for item in results:
                rows.append(parse_thesis(item))

            time.sleep(sleep_seconds)

    expected_columns = [
        "id",
        "title",
        "abstract",
        "year",
        "discipline",
        "institution",
        "url",
    ]

    if not rows:
        return pd.DataFrame(columns=expected_columns)

    df = pd.DataFrame(rows)

    for col in expected_columns:
        if col not in df.columns:
            df[col] = ""

    df = df[expected_columns].copy()

    df = df.drop_duplicates(subset=["id"])
    df = df[df["title"].str.len() > 0].copy()
    df = df.reset_index(drop=True)

    return df


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--queries",
        nargs="+",
        default=[
            "intelligence artificielle",
            "apprentissage automatique",
            "deep learning",
            "traitement automatique des langues",
            "vision par ordinateur",
        ],
    )

    parser.add_argument("--max-results-per-query", type=int, default=100)
    parser.add_argument("--page-size", type=int, default=25)
    parser.add_argument("--sleep-seconds", type=float, default=0.2)

    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/theses_raw.csv",
    )

    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = collect_theses(
        queries=args.queries,
        max_results_per_query=args.max_results_per_query,
        page_size=args.page_size,
        sleep_seconds=args.sleep_seconds,
    )

    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"\nSaved {len(df)} theses to {output_path}")

    if not df.empty:
        print("\nAperçu :")
        print(df[["id", "title", "year", "discipline"]].head())
    else:
        print("\nAucun résultat collecté.")
        print("Il faut inspecter la structure JSON retournée par l'API.")


if __name__ == "__main__":
    main()