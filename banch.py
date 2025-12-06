"""
deepset/prompt-injections
"""

import argparse
from pathlib import Path
import sys
import csv
import math
from typing import Optional, Any, Dict
import pandas as pd
from tqdm import tqdm

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    _torch_available = True
except Exception:
    _torch_available = False

from tools.predict import process_prediction
_tokenizer = None
_model = None

def _ensure_fallback_model(model_name: str, model_path: Optional[str]):
    global _tokenizer, _model, _torch_available
    if not _torch_available:
        raise RuntimeError("transformers/torch не доступны в окружении. Установи 'transformers' и 'torch' или добавь свою process_prediction.")
    if _tokenizer is None or _model is None:
        src = model_path or model_name
        print(f"[i] Загружаю модель/токенизатор из: {src} (fallback) ...")
        _tokenizer = AutoTokenizer.from_pretrained(src)
        _model = AutoModelForSequenceClassification.from_pretrained(src)
    
        _model.eval()
        if torch.cuda.is_available():
            _model.to("cuda")

def fallback_process_prediction(model_name: str, model_path: Optional[str], prompt: str, threshold: float = 0.5) -> int:
    """
    Вернёт 1 если вредоносно, 0 иначе.
    Поддерживает: модель возвращает logits для classification.
    """
    _ensure_fallback_model(model_name, model_path)

    inputs = _tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.no_grad():
        out = _model(**inputs)
        logits = out.logits 
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

    if probs.size == 1:
    
        p = 1.0 / (1.0 + math.exp(-float(probs[0])))
        return 1 if p >= threshold else 0
    else:
    
        if probs.size >= 2:
            p_mal = float(probs[1])
            return 1 if p_mal >= threshold else 0
        else:
            return int(float(probs[0]) >= threshold)

TEXT_KEYS = ("text", "prompt", "input", "message", "query", "content")
LABEL_KEYS = ("label", "malicious", "is_attack", "attack", "y", "target", "gold")

def extract_text(item: Any) -> Optional[str]:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for k in TEXT_KEYS:
            v = item.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    
        if "messages" in item and isinstance(item["messages"], list):
            for m in reversed(item["messages"]):
                if isinstance(m, dict):
                    for k in ("content", "text"):
                        if k in m and isinstance(m[k], str) and m[k].strip():
                            return m[k].strip()
    
        parts = [v.strip() for v in item.values() if isinstance(v, str) and v.strip()]
        return " ".join(parts) if parts else None
    return None

def extract_label(item: Any) -> Optional[int]:
    if isinstance(item, dict):
        for k in LABEL_KEYS:
            if k in item:
                v = item[k]
                if isinstance(v, bool):
                    return int(v)
                try:
                    return int(v)
                except Exception:
                    if isinstance(v, str):
                        vv = v.lower().strip()
                        if vv in ("malicious", "attack", "1", "true", "yes"):
                            return 1
                        if vv in ("benign", "0", "false", "no"):
                            return 0

    if isinstance(item, (int, float)):
        return int(item)
    return None

def run_eval(model_name: str, model_path: Optional[str], out_csv: str, split: str = "train", limit: Optional[int] = None, threshold: float = 0.5):
    print(f"[i] Загружаю датасет deepset/prompt-injections (split={split}) ...")
    ds = load_dataset("deepset/prompt-injections", split=split) 
    print(f"[i] Загружено записей: {len(ds)}")
    rows = []
    y_true = []
    y_pred = []
    confidences = []

    print('threshold = ', threshold)

    it = ds
    if limit is not None:
        it = it.select(range(min(limit, len(ds))))

    for i, ex in enumerate(tqdm(it, desc="Eval")):
    
        text = extract_text(ex)
        label = extract_label(ex)
        
        try:
            res = process_prediction(model_name, model_path, text)
            confidences.append(res)
            # print("Result: ", res)
        except Exception as e:
            print(f"[!] process_prediction упало на записи {i}: {e}", file=sys.stderr)
            pred = None
        else:
        
            if isinstance(res, bool):
                pred = int(res)
            elif isinstance(res, (int, float)):
                try:
                
                    if isinstance(res, float):
                        pred = 1 if float(res) >= threshold else 0
                    else:
                        pred = int(res)
                except Exception:
                    pred = int(bool(res))
            else:
            
                try:
                    p = float(res)
                    pred = 1 if p >= threshold else 0
                except Exception:
                    pred = int(bool(res))

            
        # print("    Prediction = ", pred)

        rows.append({"idx": i, "text": text, "label": label if label is not None else "", "pred": pred})
        if label is not None and pred is not None:
            y_true.append(int(label))
            y_pred.append(int(pred))


    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[i] Результаты сохранены в {out_csv} ({len(df)} строк)")


    if y_true:
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        print("=== Metrics (на подвыборке с метками) ===")
        print(f"Примеры с метками: {len(y_true)}")
        print(f"Accuracy: {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
        print(f"TP={tp} FP={fp} FN={fn} TN={tn}")

        fpr, tpr, thresholds = roc_curve(y_true, confidences)
        roc_auc = auc(fpr, tpr)

        df = pd.DataFrame({
            "fpr": fpr,
            "tpr": tpr,
            "threshold": thresholds
        })

        df.to_csv("roc_data.csv", index=False, encoding="utf-8")

        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], '--', label='Random baseline')

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.savefig("roc_curve.png", dpi=300, bbox_inches='tight')
        plt.show()
    else:
        print("[i] В датасете нет меток в используемом split (или они нераспознаны). Проверь results.csv.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-name", required=True, help="Имя модели (для логов) / hub id")
    p.add_argument("--model-path", default="", help="Путь к локальным весам (если нужно)")
    p.add_argument("--out", default="results_deepset.csv", help="CSV файл вывода")
    p.add_argument("--split", default="train", help="split датасета: train или test")
    p.add_argument("--limit", type=int, default=None, help="ограничить число примеров")
    p.add_argument("--threshold", type=float, default=0.5, help="порог если process_prediction возвращает вероятность")
    args = p.parse_args()
    mp = args.model_path if args.model_path.strip() else None
    run_eval(args.model_name, mp, args.out, split=args.split, limit=args.limit, threshold=args.threshold)

if __name__ == "__main__":
    main()
