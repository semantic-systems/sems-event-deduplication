import json
from collections import Counter
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from itertools import combinations, chain
import pickle
import random
from sklearn.model_selection import train_test_split


logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


def split_stormy_dataset():
    df = pd.read_csv("./data/stormy_data/final_df_v2.csv")
    unique_clusters = df.new_cluster.value_counts()
    large_clusters = {i: c for i, c in unique_clusters.items() if c > 500}
    medium_clusters = {i: c for i, c in unique_clusters.items() if 100 <= c <= 500}
    small_clusters = {i: c for i, c in unique_clusters.items() if c <= 100}
    train_index_large, test_index_large = train_test_split(list(large_clusters.keys()), test_size=0.2, random_state=420)
    valid_index_large, test_index_large = train_test_split(test_index_large, test_size=0.5, random_state=420)
    train_index_medium, test_index_medium = train_test_split(list(medium_clusters.keys()), test_size=0.3,
                                                             random_state=420)
    valid_index_medium, test_index_medium = train_test_split(test_index_medium, test_size=0.5, random_state=420)
    train_index_small, test_index_small = train_test_split(list(small_clusters.keys()), test_size=0.4, random_state=420)
    valid_index_small, test_index_small = train_test_split(test_index_small, test_size=0.6, random_state=420)
    print(
        f"large_clusters  {len(large_clusters)}] - train {len(train_index_large)} (sum: {sum([v for k, v in large_clusters.items() if k in train_index_large])}) - valid {len(valid_index_large)} (sum: {sum([v for k, v in large_clusters.items() if k in valid_index_large])}) - test {len(test_index_large)} (sum: {sum([v for k, v in large_clusters.items() if k in test_index_large])})")
    print(
        f"medium_clusters  {len(medium_clusters)}] - train {len(train_index_medium)} (sum: {sum([v for k, v in medium_clusters.items() if k in train_index_medium])}) - valid {len(valid_index_medium)} (sum: {sum([v for k, v in medium_clusters.items() if k in valid_index_medium])}) - test {len(test_index_medium)} (sum: {sum([v for k, v in medium_clusters.items() if k in test_index_medium])})")
    print(
        f"small_clusters  {len(small_clusters)}] - train {len(train_index_small)} (sum: {sum([v for k, v in small_clusters.items() if k in train_index_small])}) - valid {len(valid_index_small)} (sum: {sum([v for k, v in small_clusters.items() if k in valid_index_small])})- test {len(test_index_small)} (sum: {sum([v for k, v in small_clusters.items() if k in test_index_small])})")
    df_train = df.loc[df["new_cluster"].isin(train_index_large + train_index_medium + train_index_small)]
    df_valid = df.loc[df["new_cluster"].isin(valid_index_large + valid_index_medium + valid_index_small)]
    df_test = df.loc[df["new_cluster"].isin(test_index_large + test_index_medium + test_index_small)]
    print(f"df_train (len: {len(df_train)}) - df_valid (len: {len(df_valid)}) - df_test (len: {len(df_test)})")
    df_train.to_csv("./data/stormy_data/train_v2.csv", index=False)
    df_valid.to_csv("./data/stormy_data/valid_v2.csv", index=False)
    df_test.to_csv("./data/stormy_data/test_v2.csv", index=False)


def split_crisisfacts_dataset():
    df = pd.read_csv("./data/crisisfacts_data/test_from_crisisfacts.csv")
    train_df = df.loc[df["event_type"].isin(["Hurricane Florence 2018", "Hurricane Sally 2020"])]
    valid_df = df.loc[df["event_type"].isin(["Hurricane Laura 2020", "Saddleridge Wildfire 2019"])]
    test_df = df.loc[df["event_type"].isin(
        ["2018 Maryland Flood", "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"])]
    train_df.to_csv("./data/crisisfacts_data/crisisfacts_train.csv", index=False)
    valid_df.to_csv("./data/crisisfacts_data/crisisfacts_valid.csv", index=False)
    test_df.to_csv("./data/crisisfacts_data/crisisfacts_test.csv", index=False)
    storm_df = df.loc[df["event_type"].isin(["Hurricane Florence 2018", "Hurricane Sally 2020", "Hurricane Laura 2020", "2018 Maryland Flood"])]
    storm_df.to_csv("./data/crisisfacts_data/crisisfacts_storm.csv", index=False)


class StormyDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_path,
                 label_pkl: str,
                 sentence_pairs_indices_pkl: str,
                 sample_indices_path: str,
                 subset: float = 1.0,
                 task: str = "combined",
                 data_type: str = "train",
                 ratio: float = 0.1):
        random.seed(4)
        self.data_type = data_type
        self.task = task
        self.df = pd.read_csv(csv_path)
        self.label2int = self.get_label2int(task)
        self.sentence_pairs_indices_pkl = sentence_pairs_indices_pkl
        self.sentence_pairs_indices = self.get_sentence_pairs_indices(sentence_pairs_indices_pkl)
        self.labels, self.sentence_pairs_indices = self.get_labels(label_pkl,
                                                                   sentence_pairs_indices=self.sentence_pairs_indices,
                                                                   sample_indices_path=sample_indices_path)
        stratified_sample_indices_path = sample_indices_path.replace("sample_indices", "stratified_sample_indices")
        stratified_sample_indices_path = stratified_sample_indices_path.replace("json", "pkl")
        self.sentence_pairs_indices, self.labels = self.stratified_sample(self.sentence_pairs_indices, self.labels,
                                                                          save_path=stratified_sample_indices_path,
                                                                          ratio=ratio)
        self.labels = [self.label2int[label] for label in self.labels]

        self.end_index = round(subset * len(self.sentence_pairs_indices))
        self.sentence_pairs_indices = self.sentence_pairs_indices[:self.end_index]
        self.get_descriptions()

    @staticmethod
    def get_label2int(task):
        if task == "combined":
            label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        elif task == "event_deduplication":
            label2int = {"different_event": 0, "same_event": 1}
        elif task == "event_temporality":
            label2int = {"earlier": 0, "same_date": 1, "later": 2}
        else:
            ValueError(f"{task} not defined! Please choose from 'combined', 'event_deduplication' or 'event_temporality'")
        return label2int

    def get_sentence_pairs_indices(self, sentence_pairs_indices_pkl: str):
        if sentence_pairs_indices_pkl and Path(sentence_pairs_indices_pkl).exists():
            logger.info(f"Found sentence pair indices under {sentence_pairs_indices_pkl}. Loading ({self.task}/{self.data_type}) ...")

            with open(sentence_pairs_indices_pkl, "rb") as fp:
                sentence_pairs_indices = pickle.load(fp)
            return sentence_pairs_indices
        else:
            logger.info(f"Sentence pair indices file ({sentence_pairs_indices_pkl}) not found. Creating sentence pair indices ({self.task}/{self.data_type})...")
            sentence_pairs_indices = list(combinations(range(len(self.df)), 2))
            return sentence_pairs_indices

    def get_labels(self, label_pkl: str, sentence_pairs_indices: list, sample_indices_path: str):
        if Path(label_pkl).exists():
            logger.info(f"Found labels under {label_pkl}. Loading ({self.task}/{self.data_type}) ...")
            with open(label_pkl, "rb") as fp:
                labels = pickle.load(fp)
            assert len(labels) == len(self.sentence_pairs_indices)
        else:
            logger.info(f"Label file ({label_pkl}) not found. Creating labels ({self.task}/{self.data_type})...")
            labels = list(map(self.get_label, self.sentence_pairs_indices))
            if "ignored" in labels:
                valid_label_indices = [i for i, label in enumerate(labels) if label != "ignored"]
                labels = [labels[i] for i in valid_label_indices]
                sentence_pairs_indices = [sentence_pairs_indices[i] for i in valid_label_indices]
            sample_indices = self.get_sample_indices(labels, sample_indices_path)
            labels, sentence_pairs_indices = self.get_balanced_sentence_pairs_and_labels(sample_indices, labels, sentence_pairs_indices)
            logger.info(f"Balanced sentence pairs indices created. Storing locally ({self.sentence_pairs_indices_pkl}).")
            with open(self.sentence_pairs_indices_pkl, 'wb') as file:
                pickle.dump(sentence_pairs_indices, file)
            logger.info(f"Balanced labels created. Storing locally ({label_pkl}).")
            with open(label_pkl, 'wb') as file:
                pickle.dump(labels, file)
        return labels, sentence_pairs_indices

    @staticmethod
    def get_sample_indices(labels, save_path):
        if not Path(save_path).exists():
            logger.info(f"Sample indices ({save_path}) not found. Creating sample indices to balance dataset...")
            c = Counter(labels)
            sample_probabilities = {key: min(c.values())/value for key, value in c.items()}
            logger.info(f"Sample probabilities: {sample_probabilities}")
            sample_indices_dict = {key: [] for key, value in c.items()}
            for i, label in enumerate(labels):
                sample = np.random.choice(np.arange(0, 2), p=[1-sample_probabilities[label], sample_probabilities[label]])
                index_selected = sample > 0.5
                if index_selected:
                    sample_indices_dict[label].append(i)
            logger.info(f"Sample indices created and stored.")
            with open(save_path, 'w') as fp:
                json.dump(sample_indices_dict, fp)

            return sample_indices_dict
        else:
            logger.info(f"Sample indices ({save_path}) found. Loading...")
            with open(save_path, 'r') as f:
                sample_indices = json.load(f)
            return sample_indices

    @staticmethod
    def get_balanced_sentence_pairs_and_labels(sample_indices, labels, sentence_pairs_indices):
        logger.info(f"Creating balanced dataset...")
        sample_indices = list(chain(*list(sample_indices.values())))
        return [labels[i] for i in sample_indices], [sentence_pairs_indices[i] for i in sample_indices]

    @staticmethod
    def stratified_sample(list_data, list_labels, ratio=0.01, random_seed=42, save_path=None, forced=True):
        if not Path(save_path).exists() and forced:
            # Create a dictionary to store indices for each label
            label_indices = {}
            for i, label in enumerate(list_labels):
                if label not in label_indices:
                    label_indices[label] = []
                label_indices[label].append(i)

            # Determine the number of samples to draw for each label
            num_samples_per_label = {label: int(len(indices) * ratio) for label, indices in label_indices.items()}

            # Set random seed for reproducibility
            random.seed(random_seed)

            # Sample from each label category while maintaining balance
            sampled_indices = []
            for label, num_samples in num_samples_per_label.items():
                sampled_indices.extend(random.sample(label_indices[label], num_samples))

            # Sort the sampled indices for consistent order
            sampled_indices.sort()

            with open(save_path, 'wb') as file:
                pickle.dump(sampled_indices, file)
        else:
            with open(save_path, 'rb') as file:
                sampled_indices = pickle.load(file)
        # Create sampled lists based on the sampled indices
        sampled_data = [list_data[i] for i in sampled_indices]
        sampled_labels = [list_labels[i] for i in sampled_indices]
        logger.info(f"Stratified sentence pairs length: {len(sampled_data)}).")
        logger.info(f"Stratified labels length: {len(sampled_labels)}).")

        return sampled_data, sampled_labels

    def get_label(self, index_tuple):
        clusters = self.df.new_cluster.values
        cluster_i = clusters[index_tuple[0]]
        cluster_j = clusters[index_tuple[1]]
        if self.task == "combined":
            if cluster_i != cluster_j:
                return "different_event"
            else:
                unix_times = self.df.seendate.values
                unix_time_i = unix_times[index_tuple[0]][:8]
                unix_time_j = unix_times[index_tuple[1]][:8]
                if unix_time_i < unix_time_j:
                    label = "earlier"
                elif unix_time_i == unix_time_j:
                    label = "same_date"
                else:
                    label = "later"
                return label
        elif self.task == "event_deduplication":
            if cluster_i != cluster_j:
                return "different_event"
            else:
                return "same_event"
        elif self.task == "event_temporality":
            if cluster_i != cluster_j:
                return "ignored"
            else:
                unix_times = self.df.seendate.values
                unix_time_i = unix_times[index_tuple[0]][:8]
                unix_time_j = unix_times[index_tuple[1]][:8]
                if unix_time_i < unix_time_j:
                    label = "earlier"
                elif unix_time_i == unix_time_j:
                    label = "same_date"
                else:
                    label = "later"
                return label

    def get_descriptions(self):
        print(f"The dataset{({self.task}-{self.data_type})} csv has {len(self.df)} entries.")
        print(f"     Number of clusters - {len(self.df.new_cluster.unique())}")
        print(f"     Number of combinations over sentence pairs - {len(self.sentence_pairs_indices)}")

    def __getitem__(self, idx):
        i, j = self.sentence_pairs_indices[idx]
        return (self.df.title.values[i], self.df.title.values[j]), self.labels[idx]

    def __len__(self):
        return len(self.sentence_pairs_indices)


class CrisisFactsDataset(StormyDataset):
    def __init__(self,
                 csv_path,
                 label_pkl: str,
                 sentence_pairs_indices_pkl: str,
                 sample_indices_path: str,
                 subset: float = 1.0,
                 task: str = "combined",
                 data_type: str = "train",
                 ratio: float = 0.01):
        super().__init__(csv_path, label_pkl, sentence_pairs_indices_pkl, sample_indices_path, subset, task, data_type)
        random.seed(4)
        self.df = pd.read_csv(csv_path)
        self.df.rename({'text': 'title'}, axis=1, inplace=True)
        self.label2int = self.get_label2int(task)
        self.sentence_pairs_indices_pkl = sentence_pairs_indices_pkl
        self.sentence_pairs_indices = self.get_sentence_pairs_indices(sentence_pairs_indices_pkl)
        self.labels, self.sentence_pairs_indices = self.get_labels(label_pkl,
                                                                   sentence_pairs_indices=self.sentence_pairs_indices,
                                                                   sample_indices_path=sample_indices_path)
        stratified_sample_indices_path = sample_indices_path.replace("sample_indices", "stratified_sample_indices")
        self.sentence_pairs_indices, self.labels = self.stratified_sample(self.sentence_pairs_indices, self.labels,
                                                                          save_path=stratified_sample_indices_path,
                                                                          ratio=ratio)
        self.labels = [self.label2int[label] for label in self.labels]

        self.end_index = round(subset * len(self.sentence_pairs_indices))
        self.sentence_pairs_indices = self.sentence_pairs_indices[:self.end_index]
        self.get_descriptions()

    def get_label(self, index_tuple):
        clusters = self.df.event.values
        unix_times = self.df.unix_timestamp.values
        cluster_i = clusters[index_tuple[0]]
        cluster_j = clusters[index_tuple[1]]
        if self.task == "combined":
            if cluster_i != cluster_j:
                return "different_event"
            else:
                unix_time_i = self.round_to_day(unix_times[index_tuple[0]])
                unix_time_j = self.round_to_day(unix_times[index_tuple[1]])
                if unix_time_i < unix_time_j:
                    label = "earlier"
                elif unix_time_i == unix_time_j:
                    label = "same_date"
                else:
                    label = "later"
                return label
        elif self.task == "event_deduplication":
            if cluster_i != cluster_j:
                return "different_event"
            else:
                return "same_event"
        elif self.task == "event_temporality":
            if cluster_i != cluster_j:
                return "ignored"
            else:
                unix_time_i = self.round_to_day(unix_times[index_tuple[0]])
                unix_time_j = self.round_to_day(unix_times[index_tuple[1]])
                if unix_time_i < unix_time_j:
                    label = "earlier"
                elif unix_time_i == unix_time_j:
                    label = "same_date"
                else:
                    label = "later"
                return label

    @staticmethod
    def round_to_day(timestamp):
        return timestamp - (timestamp % 86400)

    def get_descriptions(self):
        print(f"The dataset{({self.task}-{self.data_type})} csv has {len(self.df)} entries.")
        print(f"     Number of clusters - {len(self.df.event.unique())}")
        print(f"     Number of combinations over sentence pairs - {len(self.sentence_pairs_indices)}")

    def __getitem__(self, idx):
        i, j = self.sentence_pairs_indices[idx]
        return (self.df.title.values[i], self.df.title.values[j]), self.labels[idx]

    def __len__(self):
        return len(self.sentence_pairs_indices)


if __name__ == "__main__":
    train_csv_path = Path("./data/stormy_data/train_v1.csv")
    valid_csv_path = Path("./data/stormy_data/valid_v1.csv")
    test_csv_path = Path("./data/stormy_data/test_v1.csv")
    test_crisisfacts_csv_path = Path("./data/crisisfacts_data/test_from_crisisfacts.csv")
    # train = StormyDataset(train_csv_path, label_pkl=Path("../data/stormy_data/labels_train.pkl"))
    # valid = StormyDataset(valid_csv_path, label_pkl=Path("../data/stormy_data/labels_valid.pkl"))
    # test = StormyDataset(test_csv_path, label_pkl=Path("../data/stormy_data/labels_test.pkl"))
    test_crisisfacts = CrisisFactsDataset(test_crisisfacts_csv_path, label_pkl=None)
    # print(f"Training data \n {train.get_descriptions()}\n")
    # print(f"Validation data \n {valid.get_descriptions()}\n")
    # print(f"Testing data \n {test.get_descriptions()}\n")
    print(f"Testing data \n {test_crisisfacts.get_descriptions()}\n")
    split_crisisfacts_dataset()