import os
import sys

sys.path.append(os.getcwd())

from src.data.text_preprocessing import get_bad_words

bad_words = get_bad_words()


def predict(sentence: str):
    words = sentence.split()
    words = [word for word in words if word.lower() not in bad_words]
    return " ".join(words)


if __name__ == "__main__":
    s = "Fucking text detoxification I am tired as fuck"
    print(predict(s))
