import argparse
import os
import sys

sys.path.append(os.getcwd())

import src.models.metrics as metrics
import src.models.predict_model as predictor
import src.models.baseline as baseline
import matplotlib.pyplot as plt

# Create the argument parser
parser = argparse.ArgumentParser(description="Process a file.")

# Add the file path argument
parser.add_argument("file_path", type=str, help="Path to the file")

# Parse the command-line arguments
args = parser.parse_args()

# Validate if the file exists
if not os.path.isfile(args.file_path):
    print(f"Error: File '{args.file_path}' does not exist.")
    exit(1)


def visualize(save: True, file_path):
    ref_toxicity = []
    trn_toxicity = []
    baseline_toxicity = []
    blue_scores = []
    blue_scores_base = []
    with open(file_path, "r") as file:
        for l in file:
            ref_toxicity.append(metrics.toxicity(l))
            trn = predictor.predict(l, 1)
            baseline_trn = baseline.predict(l)
            trn_toxicity.append(metrics.toxicity(trn))
            baseline_toxicity.append(metrics.toxicity(baseline_trn))
            blue_scores.append(metrics.get_blue_score(l, trn))
            blue_scores_base.append(metrics.get_blue_score(l, baseline_trn))

    plt.figure(1, figsize=(12, 12))
    plt.plot(ref_toxicity, "b", label="initial toxicity")
    plt.plot(trn_toxicity, "r", label="translated toxicity")
    plt.plot(baseline_toxicity, "g", label="baseline toxicity")
    plt.ylabel("toxicity_score")
    plt.title("Compare toxicity score")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    if save:
        plt.savefig("./reports/figures/toxicity_scores.png")
    else:
        plt.show()
    plt.figure(2, figsize=(12, 12))
    plt.boxplot(blue_scores)
    plt.ylabel("blue_score")
    plt.title("Blue score for references and model predictions")
    if save:
        plt.savefig("./reports/figures/blue_scores_ref_trn.png")
    else:
        plt.show()
    plt.figure(3, figsize=(12, 12))
    plt.boxplot(blue_scores_base)
    plt.ylabel("blue_score")
    plt.title("Blue score for references and baseline results")
    if save:
        plt.savefig("./reports/figures/blue_scores_ref_base.png")
    else:
        plt.show()


if __name__ == "__main__":
    visualize(True, args.file_path)
