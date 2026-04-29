import argparse
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm


API_URL = "https://theses.fr/api/v1/theses/recherche/"


def safe_get(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    """
    Try several possible keys in a dictionary and return the first non-empty value.
    """
    for key in keys:
        if key in data and data[key] not in [None, "", []]:
            return data[key]
    return default


def normalize_text_field(value: Any) -> str:
    """
    Convert an API field into a clean string.

    Handles:
    - strings
    - lists
    - dictionaries
    - None
    - unexpected types
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


def extract_year(item: dict[str, Any]) -> str:
    """
    Extract a year from the available date fields.

    For defended theses, dateSoutenance is usually the relevant date.
    For ongoing theses, datePremiereInscriptionDoctorat can be available instead.

    The API may return dates like:
    - "2024-10-01"
    - "08/10/2025"
    - None
    """
    candidate_keys = [
        "dateSoutenance",
        "datePremiereInscriptionDoctorat",
        "date_soutenance",
        "anneeSoutenance",
        "annee",
        "year",
    ]

    for key in candidate_keys:
        text = normalize_text_field(item.get(key))
        match = re.search(r"\b(19|20)\d{2}\b", text)
        if match:
            return match.group(0)

    # Fallback: inspect every date-like field.
    for key, value in item.items():
        key_lower = str(key).lower()
        if "date" in key_lower or "annee" in key_lower or "year" in key_lower:
            text = normalize_text_field(value)
            match = re.search(r"\b(19|20)\d{2}\b", text)
            if match:
                return match.group(0)

    return ""


def extract_subjects(item: dict[str, Any]) -> str:
    """
    Extract subject keywords from theses.fr fields.

    In the search API, subjects often look like:
    [
        {"langue": "fr", "libelle": "Intelligence Artificielle"},
        {"langue": "en", "libelle": "Artificial Intelligence"}
    ]
    """
    subjects = safe_get(
        item,
        ["sujets", "subjects", "keywords", "motsCles"],
        default=[],
    )

    labels = []

    if isinstance(subjects, list):
        for subject in subjects:
            if isinstance(subject, dict):
                label = subject.get("libelle") or subject.get("label") or subject.get("value")
                if label:
                    labels.append(str(label).strip())
            elif isinstance(subject, str):
                labels.append(subject.strip())

    elif isinstance(subjects, dict):
        for value in subjects.values():
            text = normalize_text_field(value)
            if text:
                labels.append(text)

    elif isinstance(subjects, str):
        labels.append(subjects.strip())

    # Remove duplicates while preserving order.
    seen = set()
    unique_labels = []

    for label in labels:
        label_lower = label.lower()
        if label_lower not in seen:
            unique_labels.append(label)
            seen.add(label_lower)

    return " ; ".join(unique_labels)


def fetch_page(query: str, start: int, page_size: int) -> dict[str, Any]:
    """
    Fetch one result page from the theses.fr search API.
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
    Extract the list of theses from the API response.
    """
    theses = payload.get("theses")

    if isinstance(theses, list):
        return theses

    candidates = [
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

    return []


def parse_thesis(item: dict[str, Any]) -> dict[str, str]:
    """
    Convert one raw API result into one clean tabular row.
    """
    thesis_id = safe_get(
        item,
        ["id", "nnt", "num", "numero", "identifiant", "idThese", "these_id", "id_these"],
        default="",
    )

    title = safe_get(
        item,
        ["titrePrincipal", "titre", "title", "title_s", "titres"],
        default="",
    )

    title_en = safe_get(
        item,
        ["titreEN", "title_en", "titleEN"],
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
        ["discipline", "disciplines", "domaine", "domaines", "discipline_s"],
        default="",
    )

    institution = safe_get(
        item,
        [
            "etabSoutenanceN",
            "etablissement",
            "etablissements",
            "institution",
            "universite",
            "etablissement_soutenance",
            "etabSoutenance",
        ],
        default="",
    )

    status = safe_get(
        item,
        ["status", "statut"],
        default="",
    )

    year = extract_year(item)
    subjects = extract_subjects(item)

    thesis_id = normalize_text_field(thesis_id)

    return {
        "id": thesis_id,
        "title": normalize_text_field(title),
        "title_en": normalize_text_field(title_en),
        "abstract": normalize_text_field(abstract),
        "year": year,
        "discipline": normalize_text_field(discipline),
        "subjects": subjects,
        "institution": normalize_text_field(institution),
        "status": normalize_text_field(status),
        "url": f"https://theses.fr/{thesis_id}" if thesis_id else "",
    }


def collect_theses(
    queries: list[str],
    max_results_per_query: int,
    page_size: int,
    sleep_seconds: float,
) -> pd.DataFrame:
    """
    Collect theses for several search queries.

    Each query returns paginated results.
    Results are parsed into rows and deduplicated by thesis id.
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
        "title_en",
        "abstract",
        "year",
        "discipline",
        "subjects",
        "institution",
        "status",
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
    df = df[df["id"].str.len() > 0].copy()
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
        print(
            df[
                [
                    "id",
                    "title",
                    "title_en",
                    "year",
                    "discipline",
                    "subjects",
                    "institution",
                    "status",
                ]
            ].head()
        )
    else:
        print("\nAucun résultat collecté.")
        print("Il faut inspecter la structure JSON retournée par l'API.")


if __name__ == "__main__":
    main()