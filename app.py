"""Main Streamlit application for the NewsBot Intelligence System."""

import streamlit as st

from src.content_analysis import analyze_sentiment
from src.multilingual import detect_language, translate_text
from src.summarization import summarize_text
from src.topic_modeling import create_lda_topics, create_nmf_topics
from src.semantic_search import semantic_search

st.set_page_config(
    page_title="NewsBot Intelligence System",
    page_icon="📰",
    layout="wide",
)


st.title("📰 NewsBot Intelligence System 2.0")
st.write(
    "Analyze news articles using sentiment analysis, summarization, "
    "topic modeling, language detection, and translation."
)


article_text = st.text_area(
    "Paste a news article below:",
    height=300,
    placeholder="Enter or paste the full article here...",
)


col1, col2, col3 = st.columns(3)


with col1:
    analyze_button = st.button(
        "Analyze Sentiment",
        use_container_width=True,
    )

with col2:
    summarize_button = st.button(
        "Generate Summary",
        use_container_width=True,
    )

with col3:
    language_button = st.button(
        "Detect Language",
        use_container_width=True,
    )


if analyze_button:
    if article_text.strip():
        result = analyze_sentiment(article_text)

        st.subheader("Sentiment Analysis")

        metric1, metric2, metric3 = st.columns(3)

        metric1.metric("Sentiment", result["label"])
        metric2.metric("Polarity", result["polarity"])
        metric3.metric("Subjectivity", result["subjectivity"])
    else:
        st.warning("Please enter a news article first.")


if summarize_button:
    if article_text.strip():
        with st.spinner("Generating summary..."):
            summary = summarize_text(article_text)

        st.subheader("Article Summary")
        st.write(summary)
    else:
        st.warning("Please enter a news article first.")


if language_button:
    if article_text.strip():
        language = detect_language(article_text)

        st.subheader("Detected Language")
        st.success(language.upper())
    else:
        st.warning("Please enter a news article first.")


st.divider()


st.subheader("🌍 Translation")

language_options = {
    "English": "en",
    "Portuguese": "pt",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
}

selected_language = st.selectbox(
    "Select the target language:",
    options=list(language_options.keys()),
)

target_language = language_options[selected_language]

if st.button("Translate Article"):
    if article_text.strip():
        with st.spinner("Translating..."):
            translated_text = translate_text(
                article_text,
                target_language=target_language,
            )

        st.text_area(
            "Translated Article",
            value=translated_text,
            height=250,
        )
    else:
        st.warning("Please enter a news article first.")


st.divider()


st.subheader("📊 Topic Modeling")

topic_documents = st.text_area(
    "Enter multiple articles, separated by three dashes (---):",
    height=250,
    placeholder=(
        "First article...\n"
        "---\n"
        "Second article...\n"
        "---\n"
        "Third article..."
    ),
)

number_of_topics = st.slider(
    "Number of topics",
    min_value=2,
    max_value=5,
    value=3,
)


topic_col1, topic_col2 = st.columns(2)


with topic_col1:
    if st.button(
        "Run LDA",
        use_container_width=True,
    ):
        documents = [
            document.strip()
            for document in topic_documents.split("---")
            if document.strip()
        ]

        results = create_lda_topics(
            documents,
            number_of_topics,
        )

        st.subheader("LDA Topics")

        for topic, words in results.items():
            st.write(f"*{topic}:* {', '.join(words)}")


with topic_col2:
    if st.button(
        "Run NMF",
        use_container_width=True,
    ):
        documents = [
            document.strip()
            for document in topic_documents.split("---")
            if document.strip()
        ]

        results = create_nmf_topics(
            documents,
            number_of_topics,
        )

        st.subheader("NMF Topics")

        for topic, words in results.items():
            st.write(f"*{topic}:* {', '.join(words)}")
st.divider()

st.subheader("🔎 Semantic Search")

search_documents = st.text_area(
    "Enter news articles separated by three dashes (---):",
    height=250,
    key="semantic_documents",
)

search_query = st.text_input(
    "What topic are you looking for?"
)

top_k = st.slider(
    "Number of results",
    1,
    5,
    3,
)

if st.button("Search Articles"):
    documents = [
        doc.strip()
        for doc in search_documents.split("---")
        if doc.strip()
    ]

    if search_query and documents:
        results = semantic_search(
            search_query,
            documents,
            top_k,
        )

        st.subheader("Results")

        for i, result in enumerate(results, start=1):
            st.write(f"### Result {i}")
            st.write(f"Similarity: {result['score']}")
            st.write(result["document"])
    else:
        st.warning("Please enter a query and at least one article.")
st.divider()

st.caption(
    "NewsBot Intelligence System 2.0 — ITAI 2373 Final Project"
)
