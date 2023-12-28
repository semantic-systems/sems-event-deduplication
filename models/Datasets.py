import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from itertools import combinations
import pickle


class StormyDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, label_pkl=None):
        self.df = pd.read_csv(csv_path)
        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        self.sentence_pairs_indices = list(combinations(range(len(self.df)), 2))[:500]
        self.get_descriptions()
        if label_pkl is not None:
            with open(label_pkl, "rb") as fp:
                self.labels = pickle.load(fp)[:500]
                self.labels = [self.label2int[label] for label in self.labels]
        else:
            self.labels = list(map(self.get_label, self.sentence_pairs_indices[:500]))
            self.labels = [self.label2int[label] for label in self.labels]

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
            elif unix_time_i > unix_time_j:
                label = "later"
            else:
                label = "whatisthisshit?"
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
    csv_path = Path("../data/gdelt_crawled/final_df_v1.csv")
    data = StormyDataset(Path("../data/gdelt_crawled/final_df_v1.csv"), label_pkl=Path("../data/gdelt_crawled/labels.pkl"))
    print(data.get_descriptions())