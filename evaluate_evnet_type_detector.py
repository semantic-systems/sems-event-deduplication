from json import JSONDecodeError
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def run_coypu_ee(message):
    url = 'https://event-extraction.skynet.coypu.org'
    json_obj = {'message': message, "key": "32T82GWPSGDJTKFN"}
    x = requests.post(url, json=json_obj)
    try:
        prediction = x.json()
    except JSONDecodeError:
        prediction = {}
    return prediction.get("event type", [np.nan ] *len(message))


def test_coypu_ee_on_crisisfacts():
    df = pd.read_csv(str(Path("./data/crisisfacts_data/test_from_crisisfacts.csv").absolute()))
    df["text"] = df['text'].astype(str)
    all_event_types = []
    batch_size = 512
    num_iteration = int(np.ceil(len(df["text"].values) / batch_size))
    for i in tqdm(range(num_iteration)):
        start_index = batch_size * i
        end_index = batch_size * (i + 1)
        event_types = run_coypu_ee(list(df["text"].values[start_index:end_index]))
        all_event_types.extend(event_types)
        if i % batch_size == 0:
            silvered_df = df.loc[:end_index - 1, :]
            silvered_df.loc[:, ("pred_event_type")] = all_event_types
            silvered_df.to_csv(Path("./data/crisisfacts_data/coypu_ee_test.csv"), index=False)
    return all_event_types


if __name__ == "__main__":
    predictions = test_coypu_ee_on_crisisfacts()
    print(predictions)