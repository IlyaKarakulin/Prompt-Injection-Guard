from pathlib import Path

import yaml
from jsonargparse import CLI

from src.load_dataset import get_dataset
from src.core.classifier import LLMDefenseDetector


def validation(dataset, model_path, model_name):
    detector = LLMDefenseDetector(model_name=model_name, model_path=model_path)

    results = detector.validation(dataset["test"])
    print(results)


def main(args_path: str | Path):
    """Execute the main functionality."""
    if args_path:
        with open(args_path, encoding="utf-8") as file:
            args = yaml.safe_load(file)
    else:
        raise ValueError("args_path is empty or invalid")
    
    train_path = args["train_path"]
    test_path = args["test_path"]
    format = args["format"]
    model_path = args["model_path"]
    model_name = args["model_name"]

    dataset = get_dataset(train_path, test_path, format)
    validation(dataset, model_path, model_name)


if __name__ == "__main__":
    CLI(main, as_positional=False)