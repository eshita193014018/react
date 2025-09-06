
import re
from typing import List
import nltk

# If you have not downloaded these, run once:
# nltk.download('stopwords')
# nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOPWORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def basic_clean(text: str) -> str:
    """Lowercase and remove URLs, mentions, non-letters, extra spaces."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove urls
    text = re.sub(r"@\w+", "", text)             # remove mentions
    text = re.sub(r"[^a-z\s]", " ", text)        # remove non-letters
    text = re.sub(r"\s+", " ", text).strip()     # normalize whitespace
    return text


def tokenize_and_lemmatize(text: str, remove_stopwords: bool = True) -> str:
    """Clean, tokenize, remove stopwords (optional), and lemmatize."""
    text = basic_clean(text)
    tokens = text.split()
    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    tokens = [LEMMATIZER.lemmatize(t) for t in tokens]
    return " ".join(tokens)


if __name__ == "__main__":
    demo = "I loved the product! It's amazing :) http://a.com @user"
    print(tokenize_and_lemmatize(demo))
