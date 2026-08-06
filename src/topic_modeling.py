"""Topic modeling tools using LDA and NMF."""

from typing import Dict, List

from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def _extract_topics(
    model,
    feature_names,
    number_of_words: int = 8,
) -> Dict[str, List[str]]:
    """Extract the most important words from each topic."""
    topics = {}

    for topic_index, topic_weights in enumerate(model.components_):
        top_indices = topic_weights.argsort()[-number_of_words:][::-1]
        words = [feature_names[index] for index in top_indices]
        topics[f"Topic {topic_index + 1}"] = words

    return topics


def create_lda_topics(
    documents: List[str],
    number_of_topics: int = 3,
    number_of_words: int = 8,
) -> Dict[str, List[str]]:
    """Discover topics using Latent Dirichlet Allocation."""
    clean_documents = [
        document.strip()
        for document in documents
        if isinstance(document, str) and document.strip()
    ]

    if len(clean_documents) < 2:
        return {"Error": ["At least two documents are required."]}

    vectorizer = CountVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=1,
    )

    document_matrix = vectorizer.fit_transform(clean_documents)

    topic_count = min(number_of_topics, len(clean_documents))

    model = LatentDirichletAllocation(
        n_components=topic_count,
        random_state=42,
    )

    model.fit(document_matrix)

    return _extract_topics(
        model,
        vectorizer.get_feature_names_out(),
        number_of_words,
    )


def create_nmf_topics(
    documents: List[str],
    number_of_topics: int = 3,
    number_of_words: int = 8,
) -> Dict[str, List[str]]:
    """Discover topics using Non-negative Matrix Factorization."""
    clean_documents = [
        document.strip()
        for document in documents
        if isinstance(document, str) and document.strip()
    ]

    if len(clean_documents) < 2:
        return {"Error": ["At least two documents are required."]}

    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_df=0.95,
        min_df=1,
    )

    document_matrix = vectorizer.fit_transform(clean_documents)

    topic_count = min(
        number_of_topics,
        document_matrix.shape[0],
        document_matrix.shape[1],
    )

    model = NMF(
        n_components=topic_count,
        random_state=42,
        init="nndsvda",
        max_iter=500,
    )

    model.fit(document_matrix)

    return _extract_topics(
        model,
        vectorizer.get_feature_names_out(),
        number_of_words,
    )
