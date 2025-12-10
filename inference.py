import warnings
warnings.filterwarnings("ignore", 
    message="Some matrices hidden dimension is not a multiple of 64")

from typing import Optional

from src.core.classifier import LLMDefenseDetector


def process_prediction(model_name: Optional[str], model_path: Optional[str], prompt: str, threshold: float):
    detector = LLMDefenseDetector(model_name=model_name, model_path=model_path,)

    try:
        prediction = detector.predict(prompt, threshold)
    except Exception as e:
        print(f"❌ ОШИБКА: {e}")
        print(f"Текст: {prompt[:80]}...")
        print()

    return prediction


def fallback_process_prediction(model_name: Optional[str], model_path: Optional[str], prompt: str, threshold: float = 0.5) -> bool:
    prediction = process_prediction(
        model_name=model_name,
        model_path=model_path,
        prompt=prompt,
        threshold=threshold
    )

    return prediction["is_injection"]