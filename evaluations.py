import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from enum import Enum
from pytorch_lightning import LightningModule, Trainer


# Make simple Enum for code clarity
class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3


# Again create a Dataset but this time, do the split in train test val
class CrisisfactsTestSet(Dataset):
    event_dict = {"001": "Lilac Wildfire 2017",
                  "002": "Cranston Wildfire 2018",
                  "003": "Holy Wildfire 2018",
                  "004": "Hurricane Florence 2018",
                  "005": "2018 Maryland Flood",
                  "006": "Saddleridge Wildfire 2019",
                  "007": "Hurricane Laura 2020",
                  "008": "Hurricane Sally 2020"}

    def __init__(self):
        # load data and shuffle, befor splitting
        self.df = pd.read_csv("./data/test_from_crisisfacts.csv")
        self.train_df = self.df.loc[self.df["event_type"].isin(["Hurricane Florence 2018", "Hurricane Sally 2020"])]
        self.valid_df = self.df.loc[self.df["event_type"].isin(["Hurricane Laura 2020", "Saddleridge Wildfire 2019"])]
        self.test_df = self.df.loc[self.df["event_type"].isin(["2018 Maryland Flood", "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"])]

        self.train = self.train_df["text"]
        self.val = self.valid_df["text"]
        self.test = self.test_df["text"]

        self.train_labels = self.train_df["event_type"]
        self.val_labels = self.valid_df["event_type"]
        self.test_labels = self.test_df["event_type"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def set_fold(self, set_type):
        # Make sure to call this befor using the dataset
        if set_type == DatasetType.TRAIN:
            self.dataset, self.labels = self.train, self.train_labels
        if set_type == DatasetType.TEST:
            self.dataset, self.labels = self.test, self.test_labels
        if set_type == DatasetType.VAL:
            self.dataset, self.labels = self.val, self.val_labels
        return self


class EventDurationPredictionDataset(Dataset):
    event_dict = {"001": "Lilac Wildfire 2017",
                  "002": "Cranston Wildfire 2018",
                  "003": "Holy Wildfire 2018",
                  "004": "Hurricane Florence 2018",
                  "005": "2018 Maryland Flood",
                  "006": "Saddleridge Wildfire 2019",
                  "007": "Hurricane Laura 2020",
                  "008": "Hurricane Sally 2020"}

    def __init__(self):
        # load data and shuffle, befor splitting
        self.df = pd.read_csv("./event_duration_prediction_dataset.csv")
        self.train_df = self.df.loc[self.df["event_type"].isin(["Hurricane Florence 2018", "Hurricane Sally 2020"])]
        self.valid_df = self.df.loc[self.df["event_type"].isin(["Hurricane Laura 2020", "Saddleridge Wildfire 2019"])]
        self.test_df = self.df.loc[self.df["event_type"].isin(["2018 Maryland Flood", "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"])]

        self.train = self.train_df["text"]
        self.val = self.valid_df["text"]
        self.test = self.test_df["text"]

        self.train_labels = self.train_df["regression_label"]
        self.val_labels = self.valid_df["regression_label"]
        self.test_labels = self.test_df["regression_label"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def set_fold(self, set_type):
        # Make sure to call this befor using the dataset
        if set_type == DatasetType.TRAIN:
            self.dataset, self.labels = self.train, self.train_labels
        if set_type == DatasetType.TEST:
            self.dataset, self.labels = self.test, self.test_labels
        if set_type == DatasetType.VAL:
            self.dataset, self.labels = self.val, self.val_labels
        return self



if __name__ == "__main__":
    data = EventDurationPredictionDataset().set_fold(1)