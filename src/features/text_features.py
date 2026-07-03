"""
Build Federal Reserve narrative/text features.

Input:
    data/raw/fed_text_documents.parquet

Output:
    data/interim/fed_text_features_monthly.parquet

Main features:
    fed_embedding_pc1
    fed_embedding_pc2
    fed_embedding_pc3
    fed_embedding_shift_1m
    fed_embedding_shift_3m
    fed_similarity_to_inflation_theme
    fed_similarity_to_recession_theme
    fed_similarity_to_financial_stability_theme
    fed_similarity_to_tightening_theme
    fed_similarity_to_easing_theme

Notes:
    - Documents are chunked before embedding to reduce information loss from long Fed texts.
    - Monthly embeddings are built only from documents dated on or before each month-end.
    - PCA scores use an expanding historical window to reduce lookahead bias.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = PROJECT_ROOT / "data" / "raw" / "fed_text_documents.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "interim" / "fed_text_features_monthly.parquet"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

MAX_CHUNK_WORDS = 180
MIN_TEXT_LENGTH = 50
BATCH_SIZE = 32
EXPANDING_PCA_MIN_MONTHS = 12

THEME_SENTENCES = {
    "inflation": "Inflation remains elevated and persistent.",
    "recession": "Economic growth is slowing and recession risks are increasing.",
    "financial_stability": "Financial conditions are tightening.",
    "labor_market": "The labor market remains strong.",
    "tightening": "Policy rates may remain higher for longer.",
    "easing": "The central bank may begin easing monetary policy.",
}


def setup_logging() -> None:
    """Configure console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


def normalize_whitespace(text: str) -> str:
    """Collapse repeated whitespace."""
    return re.sub(r"\s+", " ", str(text)).strip()


def chunk_text(text: str, max_words: int = MAX_CHUNK_WORDS) -> list[str]:
    """
    Split long text into word-based chunks.

    Sentence-transformer models are usually optimized for sentence/paragraph-scale
    text, so this gives a more stable document representation than passing a very
    long document directly to the model.
    """
    text = normalize_whitespace(text)

    if not text:
        return []

    words = text.split()
    chunks = []

    for start in range(0, len(words), max_words):
        chunk = " ".join(words[start : start + max_words]).strip()
        if chunk:
            chunks.append(chunk)

    return chunks


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a matrix."""
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms


def load_fed_documents(path: Path) -> pd.DataFrame:
    """Load and validate Fed text documents."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing input file: {path}\n"
            "Run src/data/scrape_fed_text.py before running this script."
        )

    df = pd.read_parquet(path)

    required_columns = {
        "document_date",
        "document_type",
        "title",
        "url",
        "raw_text",
        "clean_text",
    }
    missing = required_columns.difference(df.columns)

    if missing:
        raise ValueError(f"Input file is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["document_date"] = pd.to_datetime(df["document_date"], errors="coerce")
    df = df.dropna(subset=["document_date"])

    df["clean_text"] = df["clean_text"].fillna("").astype(str)
    df["raw_text"] = df["raw_text"].fillna("").astype(str)

    df["text_for_embedding"] = np.where(
        df["clean_text"].str.strip().str.len() > 0,
        df["clean_text"],
        df["raw_text"],
    )

    df["text_for_embedding"] = df["text_for_embedding"].map(normalize_whitespace)
    df = df[df["text_for_embedding"].str.len() >= MIN_TEXT_LENGTH].copy()

    if df.empty:
        raise ValueError("No usable Fed documents found after cleaning.")

    df = df.sort_values(["document_date", "document_type", "title"]).reset_index(drop=True)
    df["date"] = df["document_date"].dt.to_period("M").dt.to_timestamp("M")

    return df


def build_chunk_table(df: pd.DataFrame) -> pd.DataFrame:
    """Create a chunk-level table with document positions."""
    records = []

    for doc_pos, text in enumerate(df["text_for_embedding"].tolist()):
        chunks = chunk_text(text)

        for chunk_id, chunk in enumerate(chunks):
            records.append(
                {
                    "doc_pos": doc_pos,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk,
                }
            )

    if not records:
        raise ValueError("No text chunks were created from the Fed documents.")

    return pd.DataFrame(records)


def embed_documents(
    df: pd.DataFrame,
    model: SentenceTransformer,
) -> np.ndarray:
    """
    Embed Fed documents.

    Method:
        1. Split each document into chunks.
        2. Embed all chunks.
        3. Average chunk embeddings within each document.
        4. L2-normalize document embeddings.
    """
    chunk_df = build_chunk_table(df)

    logging.info("Embedding %d text chunks from %d Fed documents.", len(chunk_df), len(df))

    chunk_embeddings = model.encode(
        chunk_df["chunk_text"].tolist(),
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    chunk_embeddings = l2_normalize(chunk_embeddings)

    embedding_dim = chunk_embeddings.shape[1]
    doc_embeddings = np.zeros((len(df), embedding_dim), dtype=float)

    for doc_pos, group_index in chunk_df.groupby("doc_pos").groups.items():
        doc_embeddings[int(doc_pos), :] = chunk_embeddings[list(group_index)].mean(axis=0)

    doc_embeddings = l2_normalize(doc_embeddings)

    return doc_embeddings


def aggregate_monthly_embeddings(
    df: pd.DataFrame,
    doc_embeddings: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Aggregate document embeddings to month-end embeddings.

    Months without new Fed documents inherit the latest known Fed narrative state.
    This creates a continuous monthly feature table without using future documents.
    """
    temp = df[["date"]].copy()
    temp["embedding"] = list(doc_embeddings)

    monthly_series = temp.groupby("date")["embedding"].apply(
        lambda values: l2_normalize(np.vstack(list(values))).mean(axis=0)
    )

    monthly_embeddings_observed = l2_normalize(np.vstack(monthly_series.to_numpy()))

    full_month_index = pd.date_range(
        start=monthly_series.index.min(),
        end=monthly_series.index.max(),
        freq="ME",
    )

    monthly_df = pd.DataFrame(
        {
            "date": monthly_series.index,
            "monthly_embedding": list(monthly_embeddings_observed),
        }
    ).set_index("date")

    monthly_df = monthly_df.reindex(full_month_index)
    monthly_df.index.name = "date"

    monthly_df["monthly_embedding"] = monthly_df["monthly_embedding"].ffill()

    document_count = temp.groupby("date").size().reindex(full_month_index, fill_value=0)
    monthly_df["fed_document_count"] = document_count.astype(int)
    monthly_df["fed_has_document_this_month"] = (monthly_df["fed_document_count"] > 0).astype(int)

    monthly_embeddings = l2_normalize(np.vstack(monthly_df["monthly_embedding"].to_numpy()))

    return monthly_df, monthly_embeddings


def calculate_embedding_shift(monthly_embeddings: np.ndarray, lag_months: int) -> np.ndarray:
    """
    Calculate cosine-distance shift from lagged monthly embeddings.

    Value interpretation:
        0.00 means no directional change.
        Larger values mean larger semantic shift.
    """
    shifts = np.full(monthly_embeddings.shape[0], np.nan, dtype=float)

    for i in range(lag_months, monthly_embeddings.shape[0]):
        similarity = cosine_similarity(
            monthly_embeddings[i : i + 1],
            monthly_embeddings[i - lag_months : i - lag_months + 1],
        )[0, 0]
        shifts[i] = 1.0 - similarity

    return shifts


def calculate_theme_similarities(
    monthly_embeddings: np.ndarray,
    model: SentenceTransformer,
) -> pd.DataFrame:
    """Calculate monthly cosine similarity to predefined macro-policy themes."""
    theme_names = list(THEME_SENTENCES.keys())
    theme_texts = [THEME_SENTENCES[name] for name in theme_names]

    theme_embeddings = model.encode(
        theme_texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    theme_embeddings = l2_normalize(theme_embeddings)

    similarity_matrix = cosine_similarity(monthly_embeddings, theme_embeddings)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        columns=[f"fed_similarity_to_{theme_name}_theme" for theme_name in theme_names],
    )

    return similarity_df


def calculate_expanding_pca_scores(
    monthly_embeddings: np.ndarray,
    n_components: int = 3,
    min_months: int = EXPANDING_PCA_MIN_MONTHS,
) -> np.ndarray:
    """
    Calculate expanding-window PCA scores.

    This avoids using future months to compute PCA scores for earlier months.
    Early rows remain NaN until there is enough history.
    """
    n_months = monthly_embeddings.shape[0]
    scores = np.full((n_months, n_components), np.nan, dtype=float)

    required_history = max(min_months, n_components + 1)

    for i in range(n_months):
        history = monthly_embeddings[: i + 1]

        if history.shape[0] < required_history:
            continue

        active_components = min(n_components, history.shape[0], history.shape[1])

        pca = PCA(n_components=active_components)
        transformed = pca.fit_transform(history)

        scores[i, :active_components] = transformed[-1, :active_components]

    return scores


def build_text_features() -> pd.DataFrame:
    """Build the final monthly Fed text feature table."""
    setup_logging()

    logging.info("Loading Fed text documents from %s", INPUT_PATH)
    fed_docs = load_fed_documents(INPUT_PATH)

    logging.info("Loading sentence-transformer model: %s", MODEL_NAME)
    model = SentenceTransformer(MODEL_NAME)

    doc_embeddings = embed_documents(fed_docs, model)

    monthly_df, monthly_embeddings = aggregate_monthly_embeddings(
        fed_docs,
        doc_embeddings,
    )

    pca_scores = calculate_expanding_pca_scores(monthly_embeddings)

    theme_similarity_df = calculate_theme_similarities(monthly_embeddings, model)

    output = pd.DataFrame(
        {
            "date": monthly_df.index,
            "fed_document_count": monthly_df["fed_document_count"].to_numpy(),
            "fed_has_document_this_month": monthly_df["fed_has_document_this_month"].to_numpy(),
            "fed_embedding_pc1": pca_scores[:, 0],
            "fed_embedding_pc2": pca_scores[:, 1],
            "fed_embedding_pc3": pca_scores[:, 2],
            "fed_embedding_shift_1m": calculate_embedding_shift(monthly_embeddings, 1),
            "fed_embedding_shift_3m": calculate_embedding_shift(monthly_embeddings, 3),
        }
    )

    output = pd.concat([output, theme_similarity_df], axis=1)

    required_columns = [
        "date",
        "fed_document_count",
        "fed_has_document_this_month",
        "fed_embedding_pc1",
        "fed_embedding_pc2",
        "fed_embedding_pc3",
        "fed_embedding_shift_1m",
        "fed_embedding_shift_3m",
        "fed_similarity_to_inflation_theme",
        "fed_similarity_to_recession_theme",
        "fed_similarity_to_financial_stability_theme",
        "fed_similarity_to_tightening_theme",
        "fed_similarity_to_easing_theme",
    ]

    missing = [col for col in required_columns if col not in output.columns]
    if missing:
        raise ValueError(f"Missing required output columns: {missing}")

    output = output.sort_values("date").reset_index(drop=True)

    return output


def main() -> None:
    """Script entry point."""
    features = build_text_features()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(OUTPUT_PATH, index=False)

    logging.info("Wrote Fed text features to %s", OUTPUT_PATH)
    logging.info("Output shape: %s", features.shape)
    logging.info("Columns: %s", list(features.columns))


if __name__ == "__main__":
    main()
