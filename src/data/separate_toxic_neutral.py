import pandas as pd
import csv
import gc
from tqdm import tqdm


if __name__ == "__main__":
    dataset = pd.read_csv("../../data/raw/filtered.tsv", sep="\t")
    gc.collect()
    neutral = open("../../data/interim/neutral.tsv", "w", newline="")
    toxic = open("../../data/interim/toxic.tsv", "w", newline="")

    neutral_writer = csv.writer(neutral, delimiter="\t")
    toxic_writer = csv.writer(toxic, delimiter="\t")

    neutral_writer.writerow(["sentence", "toxicity_score"])
    toxic_writer.writerow(["sentence", "toxicity_score"])
    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        ref = row["reference"]
        trn = row["translation"]
        ref_tox = row["ref_tox"]
        trn_tox = row["trn_tox"]
        if trn_tox - ref_tox >= 0.25:
            neutral_writer.writerow([ref, ref_tox])
            toxic_writer.writerow([trn, trn_tox])
        elif ref_tox - trn_tox >= 0.25:
            neutral_writer.writerow([trn, trn_tox])
            toxic_writer.writerow([ref, ref_tox])

    neutral.close()
    toxic.close()
