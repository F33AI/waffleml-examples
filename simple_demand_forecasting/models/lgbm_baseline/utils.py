import joblib
import pandas as pd
from featurization import featurize


def read_train_test(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=",")
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values(by=["store", "item", "date"], ascending=True, inplace=True)
    return df


def export_sklearn(model_dump) -> str:
    output_file = "model.joblib"

    joblib.dump(model_dump, output_file)
    return output_file


def predict_sklearn(test_data_path, model_path, target, features, logger) -> str:
    logger.info("Import model...")
    model, _hyperparameters = joblib.load(model_path)

    logger.info("done. Load & transform data...")
    df = read_train_test(test_data_path)
    df = featurize(df)
    df.drop([target], axis=1, inplace=True)

    logger.info("done. Generate predictions...")
    df["prediction"] = model.predict(df)
    results = df.drop(df.columns.difference(["prediction"]), axis=1)

    logger.info("done. Export predictions to csv file...")
    output_file = "output.csv"
    results.to_csv(output_file)
    logger.info("done")
    return output_file
