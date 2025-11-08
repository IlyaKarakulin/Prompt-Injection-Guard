from pathlib import Path

import yaml
from jsonargparse import CLI

from src.load_dataset import explore_dataset
from src.core.classifier import LLM


def train_and_evaluate(dataset):
    """Обучение и оценка модели"""
    detector = LLMDefenseDetector()
    
    print("🚀 Начало обучения LLM-детектора...")
    
    # Обучаем модель
    trainer = detector.train(dataset['train'], dataset['test'])
    
    # Оцениваем на тестовой выборке
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

    explore_dataset(train_path, test_path, format)


if __name__ == "__main__":
    CLI(main, as_positional=False)

