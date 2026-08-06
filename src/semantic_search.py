"""Semantic search tools for the NewsBot Intelligence System."""

from typing import Dict, List

from sentence_transformers import SentenceTransformer, util


_model = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model only when needed."""
    global _model

    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")

    return _model


def semantic_search(
    query: str,
    documents: List[str],
    top_k: int = 3,
) -> List[Dict]:
    """Find documents that are most similar to a search query."""
    clean_documents = [
        document.strip()
        for document in documents
        if isinstance(document, str) and document.strip()
    ]

    if not query.strip() or not clean_documents:
        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        convert_to_tensor=True,
    )

    document_embeddings = model.encode(
        clean_documents,
        convert_to_tensor=True,
    )

    scores = util.cos_sim(
        query_embedding,
        document_embeddings,
    )[0]

    number_of_results = min(top_k, len(clean_documents))

    best_results = scores.topk(
        k=number_of_results,
    )

    results = []

    for score, index in zip(
        best_results.values,
        best_results.indices,
    ):
        results.append(
            {
                "document": clean_documents[index],
                "score": round(float(score), 3),
            }
        )

    return results
