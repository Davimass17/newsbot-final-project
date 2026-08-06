"""Text summarization tools for the NewsBot Intelligence System."""

from typing import List

from transformers import pipeline


_summarizer = None


def get_summarizer():
    """Load the summarization model only when it is needed."""
    global _summarizer

    if _summarizer is None:
        _summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6",
        )

    return _summarizer


def split_text(text: str, max_words: int = 450) -> List[str]:
    """Split long text into smaller chunks."""
    words = text.split()

    return [
        " ".join(words[index:index + max_words])
        for index in range(0, len(words), max_words)
    ]


def summarize_text(
    text: str,
    max_length: int = 130,
    min_length: int = 40,
) -> str:
    """Generate a concise summary of a news article."""
    if not text or not text.strip():
        return "No text was provided."

    if len(text.split()) < 50:
        return text.strip()

    summarizer = get_summarizer()
    chunks = split_text(text)

    summaries = []

    for chunk in chunks:
        result = summarizer(
            chunk,
            max_length=max_length,
            min_length=min_length,
            do_sample=False,
            truncation=True,
        )

        summaries.append(result[0]["summary_text"])

    return " ".join(summaries)
