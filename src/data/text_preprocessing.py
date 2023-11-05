import sys
import os

sys.path.append(os.getcwd())

import string
from nltk import word_tokenize
from nltk.corpus import stopwords


def get_words_only(sentence: str):
    tokens = word_tokenize(sentence)
    tokens = [token.lower() for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in stopwords.words("english")]
    tokens = [token for token in tokens if token.isalpha()]
    return tokens


def get_bad_words():
    bad_words = set()
    with open(
        "./data/external/toxic_words.txt",
        "r",
        encoding="utf-8",
    ) as file:
        for l in file:
            try:
                bad_words.add(l.rstrip())
            except UnicodeDecodeError:
                continue
    return bad_words


def extract_bad_words_only(words: list[str], bad_words: set[str]):
    extracted = [word for word in words if word in bad_words]
    return extracted


if __name__ == "__main__":
    t = get_bad_words()
    t_iter = iter(t)
    for _ in range(10):
        print(next(t_iter))
    print(len(t))
