# Prompt-Injection-Guard

**Это установка датасета**
* Из root (корневой папки репоса):
```
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
sudo apt-get install git-lfs
git lfs install
git clone https://huggingface.co/datasets/jayavibhav/prompt-injection
```

**Это использование**

* Из root:
```
bash scripts/train.sh - обучение на датасете выше. Всё, что можно менять, меняется в конфиге, кроме некоторых гиперпараметров
bash scripts/validation.sh - валидация на всей тестовой выборке. Аналогично, конфиг ваше всё.
bash scripts/predict.sh - предсказание на конкретных примерах (захардкожено). Конфиг нужен только для имени модели и пути к ней.
```

**Путь к модели**

Достаточно указать путь к директории output_dir (трейновский параметр), т.е. в данном случае:
```
model_path: llm_defense_model (относительно root)
```
Ну или к model.safetensors


* P.S: при запуске обучения могут возникнуть проблемы :)
* Если они возникли, то скажите :)