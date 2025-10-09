# mlops-flight-delay
демо проект курса MLOps (7с)

# Команда
Семенова Д.Ю., Иванченко Т.А., Рогава В.К.

## 📌 Описание
Мини-проект для курса по MLOps: прогнозирование оттока клиентов.

## 🚀 Быстрый старт
1. Установите окружение:
   ```bash
   pip install -r requirements.txt

## Структура проекта

```bash
mlops-flight-delay/
│
├── data/                
│   └── raw/             # исходные данные (через DVC)
│
├── models/              # сохранённые модели
├── reports/             # отчёты (метрики, графики)
│
├── src/
│   ├── preprocess.py    # очистка и подготовка данных
│   ├── train.py         # обучение модели
│   ├── evaluate.py      # оценка модели
│   ├── predict.py       # скрипт для инференса
│   ├── api.py           # REST API (FastAPI)
│   └── utils.py         # вспомогательные функции
│
├── tests/               # unit-тесты
│   └── test_dummy.py
│
├── .gitignore
├── requirements.txt
├── dvc.yaml
├── Dockerfile
└── README.md

```
### Предобработка данных:

```bash
python src/preprocess.py
```

### Обучение модели:

```bash
python src/train.py
```

### Запуск API:

```bash
uvicorn src.api:app --reload
```
