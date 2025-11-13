import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from utils.logger import log_info, log_warning, log_success

#
# nltk.download("punkt", quiet=True)
# nltk.download("punkt_tab", quiet=True)
# nltk.download("stopwords", quiet=True)

def clean_and_tokenize(text: str) -> list[str]:
    """
    Lowercase, remove punctuation/numbers, remove stopwords, and tokenize text.
    Returns a list of words (tokens).
    """

    text = text.lower()

    text = re.sub(r"[^a-z\s]", "", text)

    tokens = word_tokenize(text)

    stop_words = set(stopwords.words("english"))
    clean_tokens = [t for t in tokens if t not in stop_words and len(t) > 1]

    return clean_tokens
