# <YOUR_IMPORTS>
import os
import json
import dill
import pandas as pd
from datetime import datetime


def load_model():
    with open("../data/models/cars_pipe_202211161620.pkl", "rb") as file:
        model = dill.load(file)
    return model


def get_files():
    files = []
    for js in os.listdir("../data/test"):
        if js.endswith(".json"):
            files.append(f"../data/test/{js}")
    return files


def predict():
    df_out = pd.DataFrame()
    files = get_files()

    for file in files:
        j = json.loads(open(file).read())
        df = pd.DataFrame.from_dict([j])
        y = load_model().predict(df)
        df_out[df['id'][0]] = y
    df_out.to_csv(f"../data/predictions/pred{datetime.now().strftime('%Y%m%d%H%M')}.csv")


if __name__ == '__main__':
    predict()
