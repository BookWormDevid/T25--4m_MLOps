import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import joblib
from pathlib import Path

PROCESSED_DATA = Path(r"C:\Users\Koraku\Documents\mlops-flight-delay\data\processed\processed.csv")
MODEL_PATH = Path("models/model.joblib")

def train():
    df = pd.read_csv(PROCESSED_DATA)

    # X и y
    X = df[["DepHour", "IsWeekend"]]
    y = (df["ArrDelay"] > 15).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    score = roc_auc_score(y_test, preds)

    Path("models").mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"✅ Model trained. ROC AUC = {score:.3f}")
    return score

if __name__ == "__main__":
    train()

