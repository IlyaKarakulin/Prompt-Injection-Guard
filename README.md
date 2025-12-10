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
bash scripts/banch.sh - валидация на отдельной тестовой выборке. Аналогично, конфиг ваше всё.
bash scripts/quantization.sh - квантование обученной модели с сохранением.
```

**Путь к модели**

* Достаточно указать путь к директории, т.е:
```
model/quantize
```

**Для разработчиков**

* Вас интересует только файл inference.py - готовое решение, которое нужно только интегрировать.

**Демка**

* Вас интересует только файл app_streamlit.py - при запуске выведутся все инструкции