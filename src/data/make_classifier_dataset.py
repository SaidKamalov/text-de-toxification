import pandas as pd
import csv
import gc
from tqdm import tqdm

if __name__ == "__main__":
    dataset = pd.read_csv("../../data/raw/filtered.tsv", sep="\t")
    gc.collect()
    classifier_dataset = open("../../data/raw/classifier_dataset.tsv", "w", newline="")

    writer = csv.writer(classifier_dataset, delimiter="\t")

    writer.writerow(["sentence", "label"])
    c = 0
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        if row["ref_tox"] < 0.01:
            writer.writerow([row["reference"], 0])
            c += 1
        if row["trn_tox"] < 0.01:
            writer.writerow([row["translation"], 0])
            c += 1
        if row["ref_tox"] > 0.99:
            writer.writerow([row["reference"], 1])
            c += 1
        if row["trn_tox"] > 0.99:
            writer.writerow([row["translation"], 1])
            c += 1
    print(f"{c} rows were saved!")
    classifier_dataset.close()
