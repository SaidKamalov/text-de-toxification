import sys
import os

sys.path.append(os.getcwd())

from src.models.gedi_adapter import GediAdapter
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoTokenizer
import torch
import transformers

""" Reference:
    https://github.com/s-nlp/detox/tree/main
"""


t5name = "s-nlp/t5-paraphrase-paws-msrp-opinosis-paranmt"
model_path = "s-nlp/gpt2-base-gedi-detoxification"
clf_name = "s-nlp/roberta_toxicity_classifier_v1"

transformers.logging.set_verbosity_error()


def get_model(device):
    tokenizer = AutoTokenizer.from_pretrained(t5name)
    print("Tokenizer loaded successfully!")

    para = AutoModelForSeq2SeqLM.from_pretrained(t5name)
    para.resize_token_embeddings(len(tokenizer))
    para.to(device)
    para.eval()
    print("Seq2Seq model loaded successfully!")

    gedi_dis = AutoModelForCausalLM.from_pretrained(model_path)

    NEW_POS = tokenizer.encode("normal", add_special_tokens=False)[0]
    NEW_NEG = tokenizer.encode("toxic", add_special_tokens=False)[0]

    gedi_dis.bias = torch.tensor([[0.08441592, -0.08441573]])
    gedi_dis.logit_scale = torch.tensor([[1.2701858]])

    print("Classifier loaded successfully!")

    adapter_v2 = GediAdapter(
        model=para,
        gedi_model=gedi_dis,
        tokenizer=tokenizer,
        gedi_logit_coef=10,
        target=0,
        neg_code=NEW_NEG,
        pos_code=NEW_POS,
        reg_alpha=3e-5,
        ub=0.01,
    )

    return adapter_v2, tokenizer, para


if __name__ == "__main__":
    """Just simulating an example from authors notebook"""

    device = torch.device("cpu")
    print(f"transformers version: {transformers.__version__}")

    model, tokenizer, paraphraser = get_model(device)

    text = "Shut up your fucking mouth."
    print("====BEFORE====")
    print(text)
    print("====AFTER=====")
    inputs = tokenizer.encode(text, return_tensors="pt").to(paraphraser.device)
    result = model.generate(
        inputs,
        do_sample=True,
        num_return_sequences=3,
        temperature=1.0,
        repetition_penalty=3.0,
        num_beams=1,
    )
    for r in result:
        print(tokenizer.decode(r, skip_special_tokens=True))
