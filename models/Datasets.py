import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from itertools import combinations
import pickle
import random


class StormyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, label_pkl=None, subset: float = 1.0):
        random.seed(4)
        self.df = pd.read_csv(csv_path)
        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        self.sentence_pairs_indices = list(combinations(range(len(self.df)), 2))
        self.end_index = round(subset * len(self.sentence_pairs_indices))
        self.sentence_pairs_indices = self.sentence_pairs_indices[:self.end_index]
        random.shuffle(self.sentence_pairs_indices)
        self.get_descriptions()
        if label_pkl is not None:
            with open(label_pkl, "rb") as fp:
                self.labels = pickle.load(fp)
                self.labels = [self.label2int[label] for label in self.labels]
                random.shuffle(self.labels)
        else:
            self.labels = list(map(self.get_label, self.sentence_pairs_indices))
            self.labels = [self.label2int[label] for label in self.labels]
            random.shuffle(self.labels)

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


class CrisisFactsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, label_pkl=None, subset: float = 1.0):
        random.seed(4)
        self.df = pd.read_csv(csv_path)
        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        self.sentence_pairs_indices = list(combinations(range(len(self.df)), 2))
        self.end_index = round(subset * len(self.sentence_pairs_indices))
        self.sentence_pairs_indices = self.sentence_pairs_indices[:self.end_index]
        random.shuffle(self.sentence_pairs_indices)
        self.get_descriptions()
        if label_pkl is not None:
            with open(label_pkl, "rb") as fp:
                self.labels = pickle.load(fp)
                self.labels = [self.label2int[label] for label in self.labels]
                random.shuffle(self.labels)
        else:
            self.labels = list(map(self.get_label, self.sentence_pairs_indices))
            with open('./data/gdelt_crawled/labels_crisisfacts_test.pkl', 'wb') as file:
                pickle.dump(self.labels, file)
            self.labels = [self.label2int[label] for label in self.labels]
            random.shuffle(self.labels)

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
        print(f"     Number of clusters - {len(self.df.new_cluster.unique())}")
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