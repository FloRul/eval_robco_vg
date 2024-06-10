import argparse
import json
import re
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np


def format_line(line: str):
    pattern = (
        r'"scores": \[\{"name": "classification_accuracy_score", "value": (\d+)\}\]'
    )
    replacement = r'"scores": \1'
    return re.sub(pattern, replacement, line)


def get_wrong_lines_from_jsonl(file):
    with open(file, "r", encoding="utf8") as f:
        lines = f.readlines()
        lines = [json.loads(format_line(line)) for line in lines if line]
        wrong_lines = [line for line in lines if line["scores"] == 0]
    return (wrong_lines, len(lines))


def extract_intent_tuples(line: dict):
    def fetch_word_between_hashes(string):
        pattern = r"<intention>(.*?)</intention>"
        matches = re.findall(pattern, string)[-1]
        return matches

    return f"label : {line['target_output']} - infered : {fetch_word_between_hashes(line['model_output'])}"


def main(args):
    # TODO: add wrong lines ratio to total lines

    wrong_lines, total_lines = get_wrong_lines_from_jsonl(args.data_path)
    tuples = [extract_intent_tuples(line) for line in wrong_lines]
    counter = Counter(tuples)

    labels, values = zip(*counter.items())

    indexes = np.arange(len(labels))
    width = 1

    # Create a pie chart
    plt.figure(figsize=(8, 8))  # Optional: Adjust the figure size
    plt.pie(values, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.legend()

    # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.axis("equal")

    # Display the pie chart
    plt.savefig(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize script")
    parser.add_argument(
        "--data_path",
        type=str,
        default="eval_results.jsonl",
        help="Path where the eval results are stored (jsonl)",
        metavar="PATH",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="visualization.png",
        help="Path to save the visualization results (png)",
        metavar="PATH",
    )
    args = parser.parse_args()
    main(args)
