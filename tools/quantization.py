from typing import Optional, Union
from pathlib import Path
import json
import os

import yaml
from jsonargparse import CLI
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
    PreTrainedTokenizer
)


def quantize_and_save_model(
    model_path: str,
    save_path: str,
    quant_bits: int,
    quant_type: str,
    compute_dtype: torch.dtype,
    use_double_quant: bool,
    device_map: str,
    safe_serialization: bool,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Quantize a model and save it to disk or Hugging Face Hub.
    
    Args:
        model_path: Either a PreTrainedModel instance or a path/name to load from.
        save_path: Directory to save quantized model.
        quant_bits: Number of bits for quantization (4 or 8).
        quant_type: Quantization type ("nf4", "fp4", "int8").
        compute_dtype: Data type for computations.
        use_double_quant: Whether to use double quantization for 4-bit.
        device_map: How to map model to devices ("auto", "cpu", "cuda", etc.).
        safe_serialization: Whether to use safetensors for saving.
        
    Returns:
        Tuple of (quantized_model, tokenizer)
        
    Raises:
        ValueError: If quantization parameters are invalid.
        OSError: If model/tokenizer cannot be loaded or saved.
    """
    
    print(f"Starting {quant_bits}-bit quantization...")
    
    if quant_bits not in [4, 8]:
        raise ValueError(f"quant_bits must be 4 or 8, got {quant_bits}")
    
    if quant_bits == 8 and quant_type not in ["int8"]:
        print(f"⚠️  For 8-bit quantization, quant_type should be 'int8', got '{quant_type}'")
        quant_type = "int8"
    
    if quant_bits == 4 and quant_type not in ["nf4", "fp4"]:
        print(f"⚠️  For 4-bit quantization, quant_type should be 'nf4' or 'fp4', got '{quant_type}'")
        quant_type = "nf4"
    
    if quant_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_type,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=use_double_quant,
        )
    else:
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_enable_fp32_cpu_offload=False,
        )
    
    print(f"Quantization config: {quant_bits}-bit {quant_type}")
    
    if isinstance(model_path, str):
        print(f"Loading model from: {model_path}")
        
        # Load model with quantization
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map=device_map
        )
        
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print(f"✅ Model loaded and quantized: {model.__class__.__name__}")
    
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    
    print(f"💾 Saving quantized model to: {save_path}")
    
    # Save model
    model.save_pretrained(
        save_path,
        safe_serialization=safe_serialization
    )
    
    tokenizer.save_pretrained(save_path)
    
    # Save quantization config
    quant_config_dict = bnb_config.to_dict()
    with open(save_path / "quantization_config.json", "w") as f:
        json.dump(quant_config_dict, f, indent=2)
    
    print(f"✅ Quantized model saved to {save_path}")
    print(f"Files created:")
    for file in save_path.glob("*"):
        print(f"   - {file.name} ({file.stat().st_size / 1024 / 1024:.1f} MB)")
    
    return model, tokenizer


def main(args_path: str | Path):
    """Execute the main functionality."""
    if args_path:
        with open(args_path, encoding="utf-8") as file:
            args = yaml.safe_load(file)
    else:
        raise ValueError("args_path is empty or invalid")
    
    output_path = args["output_path"]
    model_path = args["model_path"]
    quant_bits = args["quant_bits"]
    quant_type = args["quant_type"]
    use_double_quant = args["use_double_quant"]
    
    if os.path.exists(model_path):
        quantized_model, quantized_tokenizer = quantize_and_save_model(
            model_path=model_path,
            save_path=output_path,
            quant_bits=quant_bits,
            quant_type=quant_type,
            compute_dtype=torch.float16,
            use_double_quant=use_double_quant,
            safe_serialization=True,
            device_map="auto",
        )
    else:
        print(f"⚠️  Model path {model_path} not found, skipping example")

if __name__ == "__main__":
    CLI(main, as_positional=False)