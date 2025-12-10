from typing import Optional, Dict, Any, Union
import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    BitsAndBytesConfig
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
from datasets import Dataset, DatasetDict


def is_valid_text(example):
    return isinstance(example['prompt'], str)

class LLMDefenseDetector:
    """
    A detector for prompt injection attacks using LLM-based classification.
    
    This class provides functionality for training, validating, and using
    a transformer-based model to detect prompt injection attempts.
    
    Attributes:
        model_name (str): Name or path of the pretrained model.
        tokenizer (AutoTokenizer): Tokenizer for the model.
        model (Optional[AutoModelForSequenceClassification]): The classifier model.
        device (torch.device): Device to run the model on (CPU/CUDA).
        output_dir (Optional[str]): Directory to save trained models.
        predict_threshold (float): Threshold for classification decisions.
        bnb_config (Optional[BitsAndBytesConfig]): Configuration for model quantization.
    """
    
    def __init__(
        self, 
        model_name: str | None = None, 
        output_dir: str | None = None, 
        model_path: str | None = None
    ):
        self.model_name = model_name

        tokenizer_path = model_path if model_path is not None else model_name
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path) if model_path is not None else None
        
        self.output_dir = output_dir

    def load_model(self, model_path: str) -> AutoModelForSequenceClassification:
        return AutoModelForSequenceClassification.from_pretrained(
            model_path,
            device_map="auto"
        )

    def preprocess_data(self, dataset: Union[Dataset, DatasetDict]) -> Union[Dataset, DatasetDict]:
        """
        Preprocess dataset by tokenizing text samples.
        
        Args:
            dataset: Dataset containing 'text' and 'label' fields.
            
        Returns:
            Tokenized dataset with PyTorch format.
        """
        def tokenize_function(examples: Dict[str, list]) -> Dict[str, torch.Tensor]:
            """
            Tokenize a batch of text examples.
            
            Args:
                examples: Dictionary containing 'text' key with list of strings.
                
            Returns:
                Dictionary with tokenized inputs.
            """
            return self.tokenizer(
                examples["prompt"], 
                padding="max_length", 
                truncation=True, 
                max_length=512,
                return_tensors="pt"
            )
        
        dataset = dataset.filter(is_valid_text)

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        tokenized_datasets.set_format("torch")
        
        return tokenized_datasets
    
    def train(
        self, 
        train_dataset: Dataset, 
        eval_dataset: Optional[Dataset],
        lr: float,
        b_size: int,
        epochs: int,
        w_decay: float,
        num_workers: int,
    ) -> Optional[Trainer]:
        """
        Train the model on the provided dataset.
        
        Args:
            train_dataset: Training dataset with 'text' and 'label' fields.
            eval_dataset: Optional validation dataset for evaluation during training.
            
        Returns:
            Trainer object if training successful, None otherwise.
            
        Raises:
            ValueError: If model_name is not specified when model is None.
        """
        tokenized_train = self.preprocess_data(train_dataset)
        tokenized_eval = self.preprocess_data(eval_dataset) if eval_dataset else None
        
        if self.model is None:
            if self.model_name is None:
                raise ValueError("model_name must be specified when model is None")
            
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2,
                ignore_mismatched_sizes=True
            )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=lr,
            per_device_train_batch_size=b_size,
            per_device_eval_batch_size=b_size,
            num_train_epochs=epochs,
            weight_decay=w_decay,
            save_strategy="epoch",
            logging_dir="./logs",
            report_to=None,
            dataloader_num_workers=num_workers,
            dataloader_pin_memory=True,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        
        if self.output_dir:
            trainer.save_model()
            self.tokenizer.save_pretrained(self.output_dir)

        self.model = trainer.model
        return trainer
    
    def validation(self, test_dataset: Dataset) -> Dict[str, float]:
        """
        Validate the model on test dataset.
        
        Args:
            test_dataset: Test dataset for evaluation.
            
        Returns:
            Dictionary containing evaluation metrics.
            
        Note:
            This method loads the model to the appropriate device before validation.
        """
        print("  Loading model for validation...")
        tokenized_test = self.preprocess_data(test_dataset)
        
        # Ensure model is on correct device
        if self.model is None:
            raise ValueError("Model must be loaded before validation")
        
        self.model.to(self.device)
        self.model.eval()

        print("  Running predictions...")
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(len(tokenized_test)):
                inputs = {
                    'input_ids': tokenized_test[i]['input_ids'].unsqueeze(0).to(self.device),
                    'attention_mask': tokenized_test[i]['attention_mask'].unsqueeze(0).to(self.device)
                }
                
                outputs = self.model(**inputs)
                all_logits.append(outputs.logits.cpu().numpy())
                all_labels.append(tokenized_test[i]['labels'].item())
                
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(tokenized_test)} samples")
        
        all_logits = np.vstack(all_logits)
        all_labels = np.array(all_labels)
        
        return self.compute_metrics((all_logits, all_labels))
        
    def compute_metrics(self, eval_pred: tuple) -> dict[str, Any]:
        """
        Compute evaluation metrics from predictions and labels.
        
        Args:
            eval_pred: Tuple containing (predictions, labels).
                predictions: Model outputs (logits).
                labels: Ground truth labels.
                
        Returns:
            Dictionary with accuracy, F1, precision, and recall scores.
        """
        predictions, labels = eval_pred
        
        # Handle case where predictions might already be class indices
        if predictions.ndim == 1:
            pred_classes = predictions
        else:
            pred_classes = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, 
            pred_classes, 
            average='binary',
            zero_division=0
        )
        acc = accuracy_score(labels, pred_classes)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def stable_softmax(self, logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
        """Стабилизированный softmax с защитой от overflow/underflow"""
        logits = logits - logits.max(dim=dim, keepdim=True).values
        exp_logits = torch.exp(logits)
        
        sum_exp = exp_logits.sum(dim=dim, keepdim=True)
        sum_exp = torch.clamp(sum_exp, min=1e-10)
        
        return exp_logits / sum_exp
    
    def predict(self, text: str, predict_threshold: float) -> dict[str, Any]:
        """
        Predict whether a given text contains a prompt injection.
        
        Args:
            text: Input text to classify.
            
        Returns:
            Dictionary containing:
                is_injection: Boolean indicating if injection was detected.
                confidence: Confidence score for the positive class.
                probabilities: Array of probabilities for both classes.
                
        Raises:
            ValueError: If model is not loaded and output_dir is not specified.
        """
        if self.model is None:
            if self.output_dir is None:
                raise ValueError(
                    "Model is not loaded and output_dir is not specified. "
                    "Either load a model or specify output_dir."
                )
            self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        
        self.model.to(self.device)
        self.model.eval()
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=1024
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

            predictions = self.stable_softmax(logits, dim=-1)
            if torch.isnan(predictions).any() or torch.isinf(predictions).any():
                predictions = torch.ones_like(predictions) / predictions.shape[-1]
        
        confidence = predictions[0][1].item()
        
        return {
            'is_injection': confidence > predict_threshold,
            'confidence': confidence,
            'probabilities': predictions[0].cpu().numpy()
        }