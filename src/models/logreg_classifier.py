import sys
import os

sys.path.append(os.getcwd())

import joblib
import sklearn
import numpy
import torch
import warnings
import transformers
from transformers import DistilBertTokenizer, DistilBertModel
from src.data.text_preprocessing import get_words_only

""" I trained this classifier in kaggle due to my lack of computational resourses
    https://www.kaggle.com/saidkamalov/pmldl-classifier
"""

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
transformers.logging.set_verbosity_error()
device = "cuda" if torch.cuda.is_available() else "cpu"


def get_model(verbose: True):
    if verbose:
        print(f"Your device is {device}")

    model = joblib.load("./data/external/classifier_weights.joblib")
    if verbose:
        print("Classifier is loaded!")
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    bert.to(device)
    if verbose:
        print("DistilBertModel is loaded!")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    if verbose:
        print("Tokenizer is loaded!")

    return tokenizer, bert, model


def toxic_prob(sentence, tokenizer, bert, model, tokenize=False):
    if tokenize:
        sentence = " ".join(get_words_only(sentence))
    inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    outputs = bert(input_ids, attention_mask=attention_mask)

    embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().detach().numpy()
    prob = model.predict_proba([embedding])
    neutral_prob, toxic_prob = round(prob[0][0], 4), round(prob[0][1], 4)
    return neutral_prob, toxic_prob


if __name__ == "__main__":
    s1 = "Sun is shining!"
    s2 = "Damn, sun is shining"
    s3 = "Fucking sun is shining bitch"

    t, b, m = get_model(True)

    n_p, t_p = toxic_prob(s1, t, b, m)
    print(f"{s1}: toxic score = {t_p}")
    n_p, t_p = toxic_prob(s2, t, b, m)
    print(f"{s2}: toxic score = {t_p}")
    n_p, t_p = toxic_prob(s3, t, b, m)
    print(f"{s3}: toxic score = {t_p}")
    print("#### with tokenization")
    n_p, t_p = toxic_prob(s1, t, b, m, True)
    print(f"{s1}: toxic score = {t_p}")
    n_p, t_p = toxic_prob(s2, t, b, m, True)
    print(f"{s2}: toxic score = {t_p}")
    n_p, t_p = toxic_prob(s3, t, b, m, True)
    print(f"{s3}: toxic score = {t_p}")
