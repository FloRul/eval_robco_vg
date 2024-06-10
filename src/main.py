import os
import re
import argparse
import json
from fmeval.eval_algorithms.qa_accuracy import QAAccuracy, QAAccuracyConfig
from fmeval.eval_algorithms.classification_accuracy import (
    ClassificationAccuracy,
    ClassificationAccuracyConfig,
)
from fmeval.data_loaders.data_config import DataConfig
from utils import combine_from_folder, format_results
from runners.robco_runner import RobcoRunner


def classification_converter(input: str, labels: list[str]) -> str:
    matches = re.findall("<intention>(.*?)</intention>", input)
    if len(matches) == 0:
        return "irrelevant"
    else:
        for match in matches:
            if match in labels:
                return match
    return "irrelevant"


def main(args):
    os.environ["EVAL_RESULTS_PATH"] = args.eval_results_folder
    os.environ["PARALLELIZATION_FACTOR"] = str(args.parallelization_factor)
    os.environ["WS_THROTTLE"] = str(args.ws_throttle)

    path = "data/eval_data.jsonl"

    combine_from_folder(
        folder="data/master_datasets",
        n=args.sample_size,
        output_path=path,
    )

    valid_labels = [
        "irrelevant",
        "pii",
        "dqgeneral",
        "greeting",
        "redirection",
        "contact",
    ]

    classif_config = DataConfig(
        dataset_name="eval_data",
        dataset_uri=path,
        dataset_mime_type="application/jsonlines",
        model_input_location="Question",
        target_output_location="Intent",
    )

    redir_classif_config = DataConfig(
        dataset_name="eval_data",
        dataset_uri="data/master_datasets/redirection.jsonl",
        dataset_mime_type="application/jsonlines",
        model_input_location="Question",
        target_output_location="Intent",
    )

    classifier_algo = ClassificationAccuracy(
        ClassificationAccuracyConfig(
            valid_labels=valid_labels,
            converter_fn=classification_converter,
        )
    )

    qa_accuracy_config = DataConfig(
        dataset_name="eval_data",
        dataset_uri=path,
        dataset_mime_type="application/jsonlines",
        model_input_location="Question",
        target_output_location="Reponse",
    )

    redirection_config = DataConfig(
        dataset_name="redirection",
        dataset_uri="data/master_datasets/redirection.jsonl",
        dataset_mime_type="application/jsonlines",
        model_input_location="Question",
        target_output_location="Reponse",
    )

    qa_algo = QAAccuracy(QAAccuracyConfig())

    classif_eval_result = classifier_algo.evaluate(
        model=RobcoRunner(
            ws_address=args.ws_address,
            output_intent=True,
            ws_origin=args.ws_origin,
        ),
        save=True,
        dataset_config=redir_classif_config,
        num_records=1000,
    )

    qa_eval_result = qa_algo.evaluate(
        model=RobcoRunner(
            ws_address=args.ws_address,
            ws_origin=args.ws_origin,
        ),
        save=True,
        dataset_config=qa_accuracy_config,
        num_records=1000,
    )

    # redirection_result = qa_algo.evaluate(
    #     model=RobcoRunner(
    #         ws_address=args.ws_address,
    #         ws_origin=args.ws_origin,
    #     ),
    #     save=True,
    #     dataset_config=redirection_config,
    #     num_records=1000,
    # )

    with open(f"{args.eval_results_folder}/eval_results_summary.json", "w") as f:
        f.write(
            json.dumps(
                format_results([*classif_eval_result, *qa_eval_result]), indent=4
                # format_results([*redirection_result, *classif_eval_result]),
                # indent=4,
            )
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script")
    parser.add_argument(
        "--sample_size", type=int, default=1, help="Sample size per intent dataset"
    )
    parser.add_argument(
        "--parallelization_factor",
        type=int,
        default=1,
        help="Parallelization factor (CPU only)",
    )
    parser.add_argument("--ws_throttle", type=int, default=1, help="WebSocket throttle")
    parser.add_argument(
        "--ws_address", type=str, default=None, help="WebSocket address", required=True
    )
    parser.add_argument(
        "--eval_results_folder",
        type=str,
        default="eval_results",
        help="The folder to put the eval results in (model input + output)",
    )
    parser.add_argument(
        "--ws_origin",
        type=str,
        default=None,
        help="The origin of the WebSocket connection (used for CORS)",
        required=True,
    )
    args = parser.parse_args()
    main(args)
