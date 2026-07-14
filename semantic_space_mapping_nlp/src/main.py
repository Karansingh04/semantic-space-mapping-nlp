"""Semantic Space Mapping - minimal reproducible NLP project.

Run from project root:
    python src/main.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "concepts.csv"
RESULTS_DIR = ROOT / "results"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Load concept dataset."""
    return pd.read_csv(path)


def build_text_embeddings(df: pd.DataFrame):
    """Create normalized embeddings using a pretrained Sentence-BERT model."""
    documents = (
        df["concept"] + ". " + df["description"]
    ).tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")

    embeddings = model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    return embeddings


def save_cosine_similarity(
    df: pd.DataFrame,
    embeddings,
    output_path: Path,
):
    """Compute and save pairwise cosine similarities between concepts."""
    similarity_matrix = cosine_similarity(embeddings)

    similarity_df = pd.DataFrame(
        similarity_matrix,
        index=df["concept"],
        columns=df["concept"],
    )

    similarity_df.to_csv(output_path)

    return similarity_matrix


def save_nearest_neighbors(
    df: pd.DataFrame,
    similarity_matrix,
    output_path: Path,
    top_n: int = 5,
):
    """Save the most semantically similar concepts for each concept."""
    rows = []

    for concept_index, concept in enumerate(df["concept"]):
        scores = similarity_matrix[concept_index]

        sorted_indices = scores.argsort()[::-1]

        nearest_indices = [
            index
            for index in sorted_indices
            if index != concept_index
        ][:top_n]

        for rank, neighbor_index in enumerate(
            nearest_indices,
            start=1,
        ):
            rows.append(
                {
                    "concept": concept,
                    "rank": rank,
                    "neighbor": df.iloc[neighbor_index]["concept"],
                    "neighbor_category": df.iloc[neighbor_index]["category"],
                    "cosine_similarity": scores[neighbor_index],
                }
            )

    pd.DataFrame(rows).to_csv(
        output_path,
        index=False,
    )


def cluster_embeddings(
    embeddings,
    n_clusters: int,
):
    """Cluster embeddings with K-Means."""
    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
    )

    return model.fit_predict(embeddings)


def reduce_to_2d(embeddings):
    """Reduce high-dimensional embeddings to two dimensions."""
    return PCA(
        n_components=2,
        random_state=42,
    ).fit_transform(embeddings)


def evaluate(
    true_labels,
    predicted_labels,
    embeddings,
):
    """Compute simple clustering evaluation metrics."""
    encoder = LabelEncoder()
    y_true = encoder.fit_transform(true_labels)

    return {
        "Adjusted Rand Index": adjusted_rand_score(
            y_true,
            predicted_labels,
        ),
        "Normalized Mutual Information": normalized_mutual_info_score(
            y_true,
            predicted_labels,
        ),
        "Silhouette Score": silhouette_score(
            embeddings,
            predicted_labels,
            metric="cosine",
        ),
    }


def save_plot(
    df: pd.DataFrame,
    coordinates,
    output_path: Path,
):
    """Create a PCA scatter plot of the semantic map."""
    plot_df = df.copy()
    plot_df["x"] = coordinates[:, 0]
    plot_df["y"] = coordinates[:, 1]

    plt.figure(figsize=(10, 7))

    highlight_words = {
        "football",
        "tennis",
        "pizza",
        "coffee",
        "doctor",
        "programmer",
        "happiness",
        "anger",
        "computer",
        "internet",
    }

    for category in sorted(
        plot_df["category"].unique()
    ):
        subset = plot_df[
            plot_df["category"] == category
        ]

        plt.scatter(
            subset["x"],
            subset["y"],
            label=category,
            s=70,
        )

        for _, row in subset.iterrows():
            if row["concept"] in highlight_words:
                plt.text(
                    row["x"] + 0.01,
                    row["y"] + 0.01,
                    row["concept"],
                    fontsize=8,
                )

    plt.title(
        "Semantic Map of Concepts using "
        "Sentence-BERT Embeddings and PCA"
    )
    plt.xlabel("PCA component 1")
    plt.ylabel("PCA component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()

    embeddings = build_text_embeddings(df)

    similarity_matrix = save_cosine_similarity(
        df,
        embeddings,
        RESULTS_DIR / "cosine_similarity.csv",
    )

    save_nearest_neighbors(
        df,
        similarity_matrix,
        RESULTS_DIR / "nearest_neighbors.csv",
    )

    n_clusters = df["category"].nunique()

    clusters = cluster_embeddings(
        embeddings,
        n_clusters,
    )

    coordinates = reduce_to_2d(
        embeddings
    )

    scores = evaluate(
        df["category"],
        clusters,
        embeddings,
    )

    df["cluster"] = clusters

    df.to_csv(
        RESULTS_DIR / "clusters.csv",
        index=False,
    )

    save_plot(
        df,
        coordinates,
        RESULTS_DIR / "semantic_map.png",
    )

    with open(
        RESULTS_DIR / "evaluation.txt",
        "w",
        encoding="utf-8",
    ) as file:
        for metric, value in scores.items():
            file.write(
                f"{metric}: {value:.4f}\n"
            )

    print(
        "Experiment completed. "
        "Results saved in the results/ folder."
    )

    for metric, value in scores.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()
