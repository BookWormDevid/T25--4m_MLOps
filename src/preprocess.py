import pandas as pd
from pathlib import Path

RAW_DATA = Path("data/raw/flights_sample.csv")
PROCESSED_DATA = Path("/data/processed/processed.csv")


def preprocess():
    print("🔄 Starting data preprocessing...")

    # Загрузка существующих данных
    df = pd.read_csv(RAW_DATA)
    print(f"📊 Loaded data shape: {df.shape}")
    print(f"📝 Columns: {list(df.columns)}")

    # Простая очистка
    initial_rows = len(df)
    df = df.dropna(subset=["DepTime", "ArrDelay"])
    cleaned_rows = len(df)
    print(f"🧹 Removed {initial_rows - cleaned_rows} rows with missing DepTime/ArrDelay")

    # Создание новых признаков
    df["DepHour"] = df["DepTime"] // 100
    df["IsWeekend"] = df["DayOfWeek"].isin([6, 7]).astype(int)

    # Создаем папку для результатов
    PROCESSED_DATA.parent.mkdir(parents=True, exist_ok=True)

    # Сохранение
    df.to_csv(PROCESSED_DATA, index=False)
    print(f"✅ Saved processed data to {PROCESSED_DATA}")
    print(f"📈 Final data shape: {df.shape}")

    return df


if __name__ == "__main__":
    preprocess()