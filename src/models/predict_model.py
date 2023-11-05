import os
import sys

sys.path.append(os.getcwd())

import torch
from src.models.model import get_model


device = "cuda" if torch.cuda.is_available() else "cpu"

model, tokenizer, paraphraser = get_model(device)


def predict(sentence, num_outputs):
    results = []
    inputs = tokenizer.encode(sentence, return_tensors="pt").to(paraphraser.device)
    result = model.generate(
        inputs,
        do_sample=True,
        num_return_sequences=num_outputs,
        temperature=1.0,
        repetition_penalty=3.0,
        num_beams=1,
    )
    for r in result:
        results.append(tokenizer.decode(r, skip_special_tokens=True))
    return results


if __name__ == "__main__":
    s1 = "Damn, world is a piece of shit"
    s2 = "Fucking sun is shining bitch"
    print(f"results for: {s1}")
    for r in predict(s1, 3):
        print(r)
    print(f"results for: {s2}")
    for r in predict(s2, 3):
        print(r)
