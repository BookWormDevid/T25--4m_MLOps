import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib
from pathlib import Path
import mlflow
from mlflow.models import infer_signature

# Пути к данным
PROCESSED_DATA = Path("data\processed\processed.csv")
MODEL_PATH = Path("models/model.joblib")


def train():
    # Загрузка данных
    df = pd.read_csv(PROCESSED_DATA)
    print(f"✅ Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")

    # Подготовка признаков и целевой переменной
    X = df[["DepHour", "IsWeekend"]]
    y = (df["ArrDelay"] > 15).astype(int)

    print(f"🎯 Признаки: {list(X.columns)}")
    print(f"📊 Распределение классов:")
    print(y.value_counts().sort_index())

    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"🔄 Разделение данных:")
    print(f"   Train: {X_train.shape[0]} samples")
    print(f"   Test: {X_test.shape[0]} samples")

    # Настройка MLflow
    mlflow.set_experiment("flight_delay")

    with mlflow.start_run():
        # Обучение модели
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X_train, y_train)

        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Вычисление метрик
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        accuracy = accuracy_score(y_test, y_pred)

        # Логирование в MLflow
        mlflow.log_param("model", "RandomForest")
        mlflow.log_param("n_estimators", 50)
        mlflow.log_param("features", "DepHour, IsWeekend")
        mlflow.log_param("target", "ArrDelay > 15")

        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)

        # Сохранение модели
        Path("models").mkdir(exist_ok=True)
        joblib.dump(model, MODEL_PATH)

        # Логирование модели в MLflow (современный способ)
        signature = infer_signature(X_train, model.predict(X_train))
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature
        )

        # Вывод результатов
        print("\n" + "=" * 50)
        print("📈 РЕЗУЛЬТАТЫ:")
        print("=" * 50)
        print(f"ROC AUC: {roc_auc:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))

        print(f"💾 Модель сохранена: {MODEL_PATH}")
        print("✅ MLflow эксперимент записан")

        return roc_auc


if __name__ == "__main__":
    train()
