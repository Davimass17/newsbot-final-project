"""Content analysis tools for the NewsBot Intelligence System."""

from textblob import TextBlob


def analyze_sentiment(text: str) -> dict:
    """
    Analyze the sentiment of a news article.

    Returns:
        dict: Sentiment label, polarity, and subjectivity.
    """
    if not text or not text.strip():
        return {
            "label": "Neutral",
            "polarity": 0.0,
            "subjectivity": 0.0,
        }

    analysis = TextBlob(text)

    polarity = round(analysis.sentiment.polarity, 3)
    subjectivity = round(analysis.sentiment.subjectivity, 3)

    if polarity > 0.1:
        label = "Positive"
    elif polarity < -0.1:
        label = "Negative"
    else:
        label = "Neutral"

    return {
        "label": label,
        "polarity": polarity,
        "subjectivity": subjectivity,
    }
