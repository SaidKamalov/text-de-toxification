from src.models.predict_model import predict
from src.models.metrics import toxicity

while True:
    print("Write your toxic sentence")
    sentence = input()
    print("Specify the number of translated sentences")
    n_trn = int(input())
    res = predict(sentence, n_trn)
    print(f"initial toxicity score: {toxicity(sentence)}")
    print(f"results for: {sentence}")
    for s in res:
        tox = toxicity(s)
        print(f"{s}: score = {tox}")
    print("############")
