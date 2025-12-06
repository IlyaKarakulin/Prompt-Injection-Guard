from pathlib import Path

import yaml
from jsonargparse import CLI

from src.load_dataset import get_dataset
from src.core.classifier import LLMDefenseDetector


def train_and_evaluate(dataset, model_name, output_dir, model_path):
    detector = LLMDefenseDetector(model_name=model_name, output_dir=output_dir, model_path=model_path)
    
    print(" Начало обучения LLM-детектора...")
    
    trainer = detector.train(dataset['train'], dataset['test'])
    
    results = trainer.evaluate()
    print(f"📊 Результаты на тестовой выборке: {results}")
    
    return detector


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
    model_name = args["model_name"]
    output_dir = args["output_dir"]
    model_path = args["model_path"]

    dataset = get_dataset(train_path, test_path, format)
    train_and_evaluate(dataset, model_name, output_dir, model_path)


if __name__ == "__main__":
    CLI(main, as_positional=False)