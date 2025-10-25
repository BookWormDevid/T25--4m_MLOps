import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score, confusion_matrix, classification_report
import joblib
from pathlib import Path
import json
import os
import mlflow

# Пути
MODEL_PATH = Path("models/model.joblib")
PROCESSED_DATA = Path(r"C:\Users\Koraku\Documents\mlops-flight-delay\data\processed\processed.csv")
REPORT_PATH = Path("reports/eval.json")


def evaluate():
    print("🚀 Запуск оценки модели на тестовой выборке...")

    # Проверяем существование модели и данных
    if not MODEL_PATH.exists():
        print(f"❌ Модель не найдена: {MODEL_PATH}")
        return

    if not PROCESSED_DATA.exists():
        print(f"❌ Данные не найдены: {PROCESSED_DATA}")
        return

    # Загрузка модели и данных
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(PROCESSED_DATA)

    # Подготовка данных
    X = df[["DepHour", "IsWeekend"]]
    y = (df["ArrDelay"] > 15).astype(int)

    # Разделение на train/test (используем те же параметры, что при обучении)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"📊 Тестовая выборка: {X_test.shape[0]} samples")
    print(f"🎯 Распределение классов в тесте:")
    print(y_test.value_counts().sort_index())

    # Предсказания на тестовой выборке
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Вычисление метрик
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    # Подготовка отчета для JSON
    evaluation_report = {
        "test_set_info": {
            "samples_count": len(y_test),
            "test_size": 0.2,
            "random_state": 42
        },
        "metrics": {
            "roc_auc": float(roc_auc),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "f1_score": float(f1_score)
        },
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
            "matrix": cm.tolist()
        },
        "predictions": {
            "total_predictions": len(y_pred),
            "class_0_predictions": int((y_pred == 0).sum()),
            "class_1_predictions": int((y_pred == 1).sum())
        }
    }

    # Сохранение отчета в JSON
    os.makedirs(REPORT_PATH.parent, exist_ok=True)
    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        json.dump(evaluation_report, f, indent=2, ensure_ascii=False)

    # Логирование в MLflow
    try:
        mlflow.set_experiment("flight_delay_evaluation")

        with mlflow.start_run(run_name="model_evaluation"):
            # Логируем параметры оценки
            mlflow.log_params({
                "test_size": 0.2,
                "random_state": 42,
                "evaluation_data": "full_dataset_split"
            })

            # Логируем метрики
            mlflow.log_metrics({
                "roc_auc": roc_auc,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "f1_score": f1_score
            })

            # Логируем confusion matrix как метрики
            mlflow.log_metrics({
                "true_negative": tn,
                "false_positive": fp,
                "false_negative": fn,
                "true_positive": tp
            })

            # Логируем отчет как артефакт
            mlflow.log_artifact(REPORT_PATH)

            print("✅ Метрики записаны в MLflow")

    except Exception as e:
        print(f"⚠️ Ошибка MLflow: {e}")

    # Вывод результатов
    print("\n" + "=" * 50)
    print("📈 РЕЗУЛЬТАТЫ ОЦЕНКИ НА ТЕСТЕ")
    print("=" * 50)
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1_score:.4f}")

    print(f"\n📋 Confusion Matrix:")
    print(f"TN: {tn} | FP: {fp}")
    print(f"FN: {fn} | TP: {tp}")

    print(f"\n💾 Отчет сохранен: {REPORT_PATH}")

    return evaluation_report


if __name__ == "__main__":
    evaluate()
