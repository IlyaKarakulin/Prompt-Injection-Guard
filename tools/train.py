from pathlib import Path

import yaml
from jsonargparse import CLI

from src.load_dataset import get_dataset
from src.core.classifier import LLMDefenseDetector


def train_and_evaluate(
        dataset, 
        model_name: str | None, 
        model_path: str | None,
        output_dir: str,
        lr: float,
        b_size: int,
        w_decay: float,
        epochs: int,
        num_workers: int
    ):
    detector = LLMDefenseDetector(
        model_name=model_name, 
        model_path=model_path,
        output_dir=output_dir
    )
    
    print(" Начало обучения LLM-детектора...")
    
    trainer = detector.train(
        dataset['train'], 
        dataset['test'],
        lr=lr,
        b_size=b_size,
        w_decay=w_decay,
        epochs=epochs,
        num_workers=num_workers,
    )
    
    results = trainer.evaluate()
    print(f"📊 Результаты на валидационной выборке: {results}")
    
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
    lr = args["learning_rate"]
    b_size = args["batch_size"]
    w_decay = args["weight_decay"]
    epochs = args["epochs"]
    num_workers = args["num_workers"]

    dataset = get_dataset(train_path, test_path, format)
    train_and_evaluate(
        dataset=dataset, 
        model_name=model_name,
        model_path=model_path,
        output_dir=output_dir,
        lr=lr,
        b_size=b_size,
        w_decay=w_decay,
        epochs=epochs,
        num_workers=num_workers
    )


if __name__ == "__main__":
    CLI(main, as_positional=False)