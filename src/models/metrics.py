import os
import sys

sys.path.append(os.getcwd())

from nltk.translate.bleu_score import sentence_bleu
from src.models.logreg_classifier import toxic_prob, get_model


tokenizer, bert, model = get_model(True)


def get_blue_score(reference, translation):
    return sentence_bleu(reference, translation)


def toxicity(sentence, tokenize=False):
    _, prob = toxic_prob(sentence, tokenizer, bert, model, tokenize)
    return prob


if __name__ == "__main__":
    pass
