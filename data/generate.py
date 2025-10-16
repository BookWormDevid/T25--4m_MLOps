import pandas as pd
import numpy as np
from pathlib import Path


def generate_flight_data(n_samples=5000):
    """Генерация тестовых данных о авиарейсах"""
    print("🛫 Generating sample flight data...")

    np.random.seed(42)

    # Основные данные
    data = {
        'Year': 2023,
        'Month': np.random.randint(1, 13, n_samples),
        'DayofMonth': np.random.randint(1, 29, n_samples),
        'DayOfWeek': np.random.randint(1, 8, n_samples),
        'DepTime': np.random.randint(0, 2400, n_samples),
        'ArrTime': np.random.randint(0, 2400, n_samples),
        'UniqueCarrier': np.random.choice(['AA', 'DL', 'UA', 'WN', 'NK'], n_samples),
        'FlightNum': np.random.randint(100, 2000, n_samples),
        'TailNum': [f"N{np.random.randint(1000, 9999)}" for _ in range(n_samples)],
        'ActualElapsedTime': np.random.randint(60, 400, n_samples),
        'AirTime': np.random.randint(50, 380, n_samples),
        'ArrDelay': np.random.randint(-30, 180, n_samples),
        'DepDelay': np.random.randint(-20, 160, n_samples),
        'Origin': np.random.choice(['JFK', 'LAX', 'ORD', 'DFW', 'ATL', 'DEN', 'SFO'], n_samples),
        'Dest': np.random.choice(['MIA', 'SEA', 'BOS', 'LAS', 'PHX', 'MCO', 'CLT'], n_samples),
        'Distance': np.random.randint(200, 2500, n_samples),
        'TaxiIn': np.random.randint(5, 20, n_samples),
        'TaxiOut': np.random.randint(10, 30, n_samples),
        'Cancelled': 0,
        'Diverted': 0,
    }

    df = pd.DataFrame(data)

    # Добавляем реалистичные задержки вылета на основе времени
    morning_mask = (df['DepTime'] >= 600) & (df['DepTime'] <= 900)
    evening_mask = (df['DepTime'] >= 1600) & (df['DepTime'] <= 1900)
    df.loc[morning_mask, 'DepDelay'] = np.random.randint(10, 60, morning_mask.sum())
    df.loc[evening_mask, 'DepDelay'] = np.random.randint(15, 75, evening_mask.sum())

    # Добавляем немного отмененных рейсов (2%)
    cancelled_mask = np.random.random(n_samples) < 0.02
    df.loc[cancelled_mask, 'Cancelled'] = 1
    df.loc[cancelled_mask, ['ArrDelay', 'DepDelay']] = np.nan

    # Добавляем NaN значения для реалистичности (5% в DepTime, 3% в ArrDelay)
    dep_time_nan = np.random.random(n_samples) < 0.05
    arr_delay_nan = np.random.random(n_samples) < 0.03
    df.loc[dep_time_nan, 'DepTime'] = np.nan
    df.loc[arr_delay_nan, 'ArrDelay'] = np.nan

    print(f"✅ Generated {len(df)} flight records")
    print(f"📊 Data shape: {df.shape}")
    print(f"📝 Columns: {list(df.columns)}")

    return df


def main():
    # Создаем папку если нет
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    # Генерируем данные
    df = generate_flight_data(5000)

    # Сохраняем
    output_path = Path("data/raw/flights_sample.csv")
    df.to_csv(output_path, index=False)

    print(f"💾 Saved to {output_path}")
    print("🔍 Sample data:")
    print(df[['DepTime', 'ArrDelay', 'DayOfWeek', 'Origin', 'Dest']].head(3))


if __name__ == "__main__":
    main()