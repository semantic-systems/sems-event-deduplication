import json
from collections import Counter

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from itertools import combinations, chain
import pickle
import random


class StormyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, label_pkl=None, sample_indices_path=None, subset: float = 1.0):
        random.seed(4)
        self.df = pd.read_csv(csv_path)
        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        self.sentence_pairs_indices = list(combinations(range(len(self.df)), 2))
        if label_pkl is not None:
            with open(label_pkl, "rb") as fp:
                self.labels = pickle.load(fp)
        else:
            self.labels = list(map(self.get_label, self.sentence_pairs_indices))

        if sample_indices_path:
            self.sample_indices = self.get_sample_indices(self.labels, sample_indices_path)
            self.labels = self.get_balanced_labels(sample_indices_path)
        self.labels = [self.label2int[label] for label in self.labels]

        self.sentence_pairs_indices = self.get_sentence_pairs_indices(sample_indices_path=sample_indices_path)
        self.end_index = round(subset * len(self.sentence_pairs_indices))
        self.sentence_pairs_indices = self.sentence_pairs_indices[:self.end_index]

        self.get_descriptions()

    def get_sentence_pairs_indices(self, sample_indices_path: str = None):
        if not sample_indices_path:
            return self.sentence_pairs_indices
        elif sample_indices_path and Path(sample_indices_path).exists():
            with open(sample_indices_path, 'r') as f:
                sample_indices = json.load(f)
            sample_indices = list(chain(*list(sample_indices.values())))
            return [self.sentence_pairs_indices[i] for i in sample_indices]
        else:
            raise ValueError

    @staticmethod
    def get_sample_indices(labels, save_path):
        if not Path(save_path).exists():
            c = Counter(labels)
            sample_probabilities = {key: min(c.values())/value for key, value in c.items()}
            print(save_path)
            print(sample_probabilities)
            sample_indices_dict = {key: [] for key, value in c.items()}
            for i, label in enumerate(labels):
                sample = np.random.choice(np.arange(0, 2), p=[1-sample_probabilities[label], sample_probabilities[label]])
                index_selected = sample > 0.5
                if index_selected:
                    sample_indices_dict[label].append(i)
            with open(save_path, 'w') as fp:
                json.dump(sample_indices_dict, fp)

            return sample_indices_dict
        else:
            with open(save_path, 'r') as f:
                sample_indices = json.load(f)
            return sample_indices

    def get_balanced_labels(self, sample_indices_path: str = None):
        with open(sample_indices_path, 'r') as f:
            sample_indices = json.load(f)
        sample_indices = list(chain(*list(sample_indices.values())))
        return [self.labels[i] for i in sample_indices]

    def get_label(self, index_tuple):
        clusters = self.df.new_cluster.values
        unix_times = self.df.normalized_date.values
        cluster_i = clusters[index_tuple[0]]
        cluster_j = clusters[index_tuple[1]]
        if cluster_i != cluster_j:
            return "different_event"
        else:
            unix_time_i = unix_times[index_tuple[0]]
            unix_time_j = unix_times[index_tuple[1]]
            if unix_time_i < unix_time_j:
                label = "earlier"
            elif unix_time_i == unix_time_j:
                label = "same_date"
            else:
                label = "later"
            return label

    def get_descriptions(self):
        print(f"The dataset csv has {len(self.df)} entries with the following columns - {self.df.columns.values}")
        print(f"     Number of clusters - {len(self.df.new_cluster.unique())}")
        print(f"     Number of combinations over sentence pairs - {len(self.sentence_pairs_indices)}")

    def __getitem__(self, idx):
        i, j = self.sentence_pairs_indices[idx]
        return (self.df.title.values[i], self.df.title.values[j]), self.labels[idx]

    def __len__(self):
        return len(self.sentence_pairs_indices)


class CrisisFactsDataset(StormyDataset):
    def __init__(self, csv_path, label_pkl=None, sample_indices_path=None, subset: float = 1.0):
        super().__init__(csv_path, label_pkl, sample_indices_path, subset)
        random.seed(4)
        self.df = pd.read_csv(csv_path)
        self.df.rename({'text': 'title'}, axis=1, inplace=True)
        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        self.sentence_pairs_indices = list(combinations(range(len(self.df)), 2))
        if label_pkl is not None:
            with open(label_pkl, "rb") as fp:
                self.labels = pickle.load(fp)
        else:
            self.labels = list(map(self.get_label, self.sentence_pairs_indices))

        if sample_indices_path:
            self.sample_indices = self.get_sample_indices(self.labels, sample_indices_path)
            self.labels = self.get_balanced_labels(sample_indices_path)
        self.labels = [self.label2int[label] for label in self.labels]

        self.sentence_pairs_indices = self.get_sentence_pairs_indices(sample_indices_path=sample_indices_path)
        self.end_index = round(subset * len(self.sentence_pairs_indices))
        self.sentence_pairs_indices = self.sentence_pairs_indices[:self.end_index]

        self.get_descriptions()

    def get_label(self, index_tuple):
        clusters = self.df.event.values
        unix_times = self.df.unix_timestamp.values
        cluster_i = clusters[index_tuple[0]]
        cluster_j = clusters[index_tuple[1]]
        if cluster_i != cluster_j:
            return "different_event"
        else:
            unix_time_i = unix_times[index_tuple[0]]
            unix_time_j = unix_times[index_tuple[1]]
            if unix_time_i < unix_time_j:
                label = "earlier"
            elif unix_time_i == unix_time_j:
                label = "same_date"
            else:
                label = "later"
            return label

    def get_descriptions(self):
        print(f"The dataset csv has {len(self.df)} entries with the following columns - {self.df.columns.values}")
        print(f"     Number of clusters - {len(self.df.event.unique())}")
        print(f"     Number of combinations over sentence pairs - {len(self.sentence_pairs_indices)}")

    def __getitem__(self, idx):
        i, j = self.sentence_pairs_indices[idx]
        return (self.df.title.values[i], self.df.title.values[j]), self.labels[idx]

    def __len__(self):
        return len(self.sentence_pairs_indices)


if __name__ == "__main__":
    train_csv_path = Path("./data/gdelt_crawled/train_v1.csv")
    valid_csv_path = Path("./data/gdelt_crawled/valid_v1.csv")
    test_csv_path = Path("./data/gdelt_crawled/test_v1.csv")
    test_crisisfacts_csv_path = Path("./data/test_from_crisisfacts.csv")
    # train = StormyDataset(train_csv_path, label_pkl=Path("../data/gdelt_crawled/labels_train.pkl"))
    # valid = StormyDataset(valid_csv_path, label_pkl=Path("../data/gdelt_crawled/labels_valid.pkl"))
    # test = StormyDataset(test_csv_path, label_pkl=Path("../data/gdelt_crawled/labels_test.pkl"))
    test_crisisfacts = CrisisFactsDataset(test_crisisfacts_csv_path, label_pkl=None)
    # print(f"Training data \n {train.get_descriptions()}\n")
    # print(f"Validation data \n {valid.get_descriptions()}\n")
    # print(f"Testing data \n {test.get_descriptions()}\n")
    print(f"Testing data \n {test_crisisfacts.get_descriptions()}\n")