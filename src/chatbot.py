"""Simple conversational assistant for NewsBot."""

from src.content_analysis import analyze_sentiment
from src.multilingual import detect_language
from src.summarization import summarize_text


def answer_question(question: str, article: str) -> str:
    """Answer simple questions about a news article."""
    if not question.strip():
        return "Please enter a question."

    if not article.strip():
        return "Please provide a news article first."

    question_lower = question.lower()

    if "summary" in question_lower or "summarize" in question_lower:
        return summarize_text(article)

    if "sentiment" in question_lower or "positive" in question_lower:
        result = analyze_sentiment(article)

        return (
            f"The article sentiment is {result['label']}. "
            f"Polarity: {result['polarity']}. "
            f"Subjectivity: {result['subjectivity']}."
        )

    if "language" in question_lower:
        language = detect_language(article)
        return f"The detected language is {language.upper()}."

    if "what is this article about" in question_lower:
        return summarize_text(article)

    return (
        "I can answer questions about the article summary, "
        "sentiment, and detected language."
    )
