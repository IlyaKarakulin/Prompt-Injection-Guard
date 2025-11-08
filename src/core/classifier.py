import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

class LLMDefenseDetector:
    def __init__(self, model_name, output_dir = None, model_path = None):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = self.load_model(model_path) if model_path is not None else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = output_dir

    def load_model(self, model_path):
        return AutoModelForSequenceClassification.from_pretrained(model_path)
    
    def preprocess_data(self, dataset):
        """Подготовка данных для обучения"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"], 
                padding="max_length", 
                truncation=True, 
                max_length=512,
                return_tensors="pt"  # ⬅️ Важно: возвращаем тензоры
            )
        
        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
        
        # Устанавливаем формат для PyTorch
        tokenized_datasets.set_format("torch")
        
        return tokenized_datasets
    
    def train(self, train_dataset, eval_dataset=None):
        """Обучение модели"""
        tokenized_train = self.preprocess_data(train_dataset)
        tokenized_eval = self.preprocess_data(eval_dataset) if eval_dataset else None
        
        if self.model is None:
            if self.model_name is None:
                return None
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, 
                num_labels=2,
                ignore_mismatched_sizes=True
            )
        
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
            report_to=None,
        )
        
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_eval,
            compute_metrics=self.compute_metrics,
        )
        
        trainer.train()
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)

        self.model = trainer.model

        return trainer
    
    def validation(self, test_dataset):
        print("  Загрузка модели...")
        tokenized_test = self.preprocess_data(test_dataset)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        self.model.eval()

        print("  Выполнение предсказаний...")
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
                print(f"iteration: {i}")
        
        all_logits = np.vstack(all_logits)
        all_labels = np.array(all_labels)
        
        return self.compute_metrics((all_logits, all_labels))
        
    def compute_metrics(self, eval_pred):
        """Вычисление метрик"""
        print("📈 Вычисление метрик...")
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        acc = accuracy_score(labels, predictions)
        
        return {
            'accuracy': acc,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def predict(self, text):
        """Предсказание для одного текста"""
        if self.model is None:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.output_dir)
        
        self.model.to(self.device)
        self.model.eval()
        
        inputs = self.tokenizer(
            text, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        
        return {
            'is_injection': predictions[0][1].item() > 0.5,
            'confidence': predictions[0][1].item(),
            'probabilities': predictions[0].cpu().numpy()
        }