def export_sklearn(model_dump) -> str:
    import joblib
    output_file = "model.joblib"

    joblib.dump(model_dump, output_file)
    return output_file


def predict_sklearn(test_data_path, model_path, TARGET, features, logger) -> str:
    from .featurization.transform_data import prepare_test
    import joblib

    logger.info("Import model...")
    model, _hyperparameters, train_stub = joblib.load(model_path)

    logger.info("done. Load & transform data...")
    df = prepare_test(test_data_path, train_stub, features)
    df.drop([TARGET], axis=1, inplace=True)

    logger.info("done. Generate predictions...")
    df["prediction"] = model.predict(df)
    results = df.drop(df.columns.difference(["prediction"]), axis=1)

    logger.info("done. Export predictions to csv file...")
    output_file = "output.csv"
    results.to_csv(output_file)
    logger.info("done")
    return output_file
