from hamilton import base, driver

from . import features, read_data


def prepare_train_dataset(data_path:str):
    config = {
        "data_path":data_path,
        "stage": "train"
    }
    adapter = base.SimplePythonGraphAdapter(base.DictResult())

    dr = driver.Driver(config, read_data, features, adapter=adapter)

    results = dr.execute(["X", "target"])

    return results

def prepare_test_dataset(data_path:str):
    config = {
        "data_path":data_path,
        "stage": "test"
    }
    dr = driver.Driver(config, read_data, features)

    results = dr.execute(["X"])

    return results

def prepare_ground_truth_dataset(data_path:str):
    config = {
        "data_path":data_path,
        "stage": "evaluate"
    }
    adapter = base.SimplePythonGraphAdapter(base.DictResult())
    dr = driver.Driver(config, read_data, features, adapter=adapter)

    results = dr.execute(["input_dataset"])

    return results["input_dataset"] 
