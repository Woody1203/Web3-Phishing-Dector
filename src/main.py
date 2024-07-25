from pathlib import Path
import argparse


import os
import sys

# Add the src directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.train import train_baselines, train_llms
try:
    from src.infer import make_inference
except:
    pass


def getArgs():
    parser = argparse.ArgumentParser(description="Parse arguments from command input.")
    parser.add_argument(
        "-t",
        "--task",
        action="store",
        required=True,
        type=str,
        choices=["train", "infer"],
        help='Enter "train" To train models. Enter "infer" to make an inference. ',
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        action="store",
        type=str,
        default="LLM",
        choices=["LLM", "baseline"],
        help="Choose model type to train.",
    )
    parser.add_argument(
        "-c",
        "--csv_name",
        action="store",
        type=str,
        default="data.csv",
        help="Enter the csv file name for training dataset.",
    )
    parser.add_argument(
        "-l",
        "--label_col_name",
        action="store",
        type=str,
        default="spam",
        help="Enter name of the binary classification column in the csv dataset.",
    )
    parser.add_argument(
        "-n",
        "--text_col_name",
        action="store",
        type=str,
        default="text",
        help="Enter name of the text column in the csv dataset.",
    )
    parser.add_argument(
        "-i",
        "--text_input",
        action="store",
        type=str,
        default="This is a sample test",
        help="Enter message to be tested.",
    )
    return parser.parse_args()


if __name__ == "__main__":

    Path("outputs/csv").mkdir(parents=True, exist_ok=True)
    Path("outputs/scores").mkdir(parents=True, exist_ok=True)
    Path("outputs/model").mkdir(parents=True, exist_ok=True)

    model_name = "roberta-base"

    current_file_path = os.path.abspath(__file__)
    project_root_dir = os.path.dirname(os.path.dirname(current_file_path))
    save_directory = os.path.join(project_root_dir, "outputs", "model", "roberta-trained")
    os.makedirs(save_directory, exist_ok=True)

    # load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # save model and tokenizer
    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)

    arg = getArgs()

    DATASET_NAME = arg.csv_name
    LABEL_COL_NAME = arg.label_col_name
    TEXT_COL_NAME = arg.text_col_name
    TEXT_INPUT = arg.text_input

    
    # print(TEXT_INPUT)

    if arg.task == "train":
        if arg.model_type == "baseline":
            print("training baseline models ...")
            train_baselines(
                dataset_name=DATASET_NAME,
                label_col_name=LABEL_COL_NAME,
                text_col_name=TEXT_COL_NAME,
            )
        elif arg.model_type == "LLM":
            print("train LLM model")
            train_llms(
                dataset_name=DATASET_NAME,
                label_col_name=LABEL_COL_NAME,
                text_col_name=TEXT_COL_NAME,
            )
        else:
            print("train LLM model")
            train_llms(
                dataset_name=DATASET_NAME,
                label_col_name=LABEL_COL_NAME,
                text_col_name=TEXT_COL_NAME,
            )

    elif arg.task == "infer":
        if arg.model_type == "baseline":
            pred = make_inference(
                user_input=TEXT_INPUT,
                dataset_name=DATASET_NAME,
                label_col_name=LABEL_COL_NAME,
                text_col_name=TEXT_COL_NAME,
            ).best_baseline()
        elif arg.model_type == "LLM":
            pred = make_inference(
                user_input=TEXT_INPUT,
                dataset_name=DATASET_NAME,
                label_col_name=LABEL_COL_NAME,
                text_col_name=TEXT_COL_NAME,
            ).best_llm()
        
        else:
            pred = make_inference(
                user_input=TEXT_INPUT,
                dataset_name=DATASET_NAME,
                label_col_name=LABEL_COL_NAME,
                text_col_name=TEXT_COL_NAME,
            ).best_llm()

        print(pred)
