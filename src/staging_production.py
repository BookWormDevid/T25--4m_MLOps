import argparse
import mlflow
from mlflow.tracking import MlflowClient

def set_model_stage(version, stage):
    # Указываем правильный путь к локальному хранилищу
    mlflow.set_tracking_uri("file:///C:/Users/Koraku/Documents/mlops-flight-delay/src/mlruns")

    client = MlflowClient()
    model_name = "flight_delay_model"

    # Переводим модель в нужный stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage=stage,
        archive_existing_versions=False
    )

    print(f"✅ Модель '{model_name}' (версия {version}) переведена в stage: {stage}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, required=True, help="Версия модели в MLflow")
    parser.add_argument("--stage", type=str, required=True, choices=["Staging", "Production"], help="Stage назначения")
    args = parser.parse_args()

    set_model_stage(args.version, args.stage)

#  запускать так python staging_production.py --version 1 --stage (Staging или Production)