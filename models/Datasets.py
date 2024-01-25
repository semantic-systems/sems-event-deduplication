import json
from collections import Counter
import logging
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from torch.utils.data import Dataset
from itertools import product, chain
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
                 subset: float = 1.0,
                 task: str = "combined",
                 data_type: str = "train",
                 forced: bool = False):
        random.seed(4)
        self.data_type = data_type
        self.task = task
        self.df = pd.read_csv(csv_path)
        self.label2int = self.get_label2int(task)
        if not Path(f"./data/stormy_data/{task}").exists():
            Path(f"./data/stormy_data/{task}").mkdir()
        save_path = str(Path(f"./data/stormy_data/{task}", f"{data_type}.csv").absolute())
        self.sampled_df = self.stratified_sample(save_path=save_path, subset=subset, forced=forced)
        self.labels = [self.label2int[label] for label in self.sampled_df.labels.values]
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

    def stratified_sample(self, save_path=None, subset=0.1, forced=True):
        if not Path(save_path).exists() or forced:
            if 0 < subset < 1:
                self.df = self.df.sample(weights=self.df.groupby("wikidata_link")['seendate'].transform('count'),
                                    n=int(subset * len(self.df)))
            sentence_pairs_indices = list(product(range(len(self.df)), repeat=2))
            logger.info(f"Full data size: {len(sentence_pairs_indices)}).")
            sentence_a = []
            sentence_b = []
            event_a = []
            event_b = []
            time_a = []
            time_b = []
            url_a = []
            url_b = []
            labels = list(map(self.get_label, sentence_pairs_indices))
            for sent_pair_indices in sentence_pairs_indices:
                sentence_a.append(self.df.title.values[sent_pair_indices[0]])
                sentence_b.append(self.df.title.values[sent_pair_indices[1]])
                event_a.append(self.df.wikidata_link.values[sent_pair_indices[0]])
                event_b.append(self.df.wikidata_link.values[sent_pair_indices[1]])
                time_a.append(self.df.seendate.values[sent_pair_indices[0]])
                time_b.append(self.df.seendate.values[sent_pair_indices[1]])
                url_a.append(self.df.url.values[sent_pair_indices[0]])
                url_b.append(self.df.url.values[sent_pair_indices[1]])
            df = pd.DataFrame(
                list(zip(sentence_a, event_a, time_a, labels, sentence_b, event_b, time_b, url_a, url_b)),
                columns=['sentence_a', 'event_a', 'time_a', 'labels', 'sentence_b', 'event_b', 'time_b', 'url_a',
                         'url_b'])
            df = df.loc[df["labels"] != "ignored"]
            print(df.labels.value_counts())

            df.to_csv(save_path)
        else:
            df = pd.read_csv(save_path)

        logger.info(f"Stratified samples length: {len(df)}).")

        return df

    def get_label(self, index_tuple):
        clusters = self.df.wikidata_link.values
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
        print(f"The dataset{({self.task}-{self.data_type})} csv has {len(self.sampled_df)} entries.")
        print(f"     Number of clusters - {len(self.df.wikidata_link.unique())}")

    def __getitem__(self, idx):
        return (self.sampled_df.sentence_a.values[idx], self.sampled_df.sentence_b.values[idx]), self.labels[idx]

    def __len__(self):
        return len(self.sampled_df)


class CrisisFactsDataset(torch.utils.data.Dataset):
    def __init__(self,
                 csv_path,
                 subset: float = 1.0,
                 task: str = "combined",
                 data_type: str = "train",
                 forced: bool = False):
        random.seed(4)
        self.data_type = data_type
        self.task = task
        self.df = pd.read_csv(csv_path)
        self.label2int = self.get_label2int(task)
        if not Path(f"./data/crisisfacts_data/{task}").exists():
            Path(f"./data/crisisfacts_data/{task}").mkdir()
        save_path = str(Path(f"./data/crisisfacts_data/{task}", f"{data_type}.csv").absolute())
        self.sampled_df = self.stratified_sample(save_path=save_path, subset=subset, forced=forced)
        self.labels = [self.label2int[label] for label in self.sampled_df.labels.values]
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
            ValueError(
                f"{task} not defined! Please choose from 'combined', 'event_deduplication' or 'event_temporality'")
        return label2int

    def stratified_sample(self, save_path=None, subset=0.1, forced=True):
        if not Path(save_path).exists() or forced:
            if 0 < subset < 1:
                df = self.df.sample(weights=self.df.groupby("event")['unix_timestamp'].transform('count'),
                                    n=int(subset * len(self.df)))
            else:
                sentence_pairs_indices = list(product(range(len(df)), repeat=2))
                logger.info(f"Full data size: {len(sentence_pairs_indices)}).")

                sentence_a = []
                sentence_b = []
                event_a = []
                event_b = []
                time_a = []
                time_b = []
                url_a = []
                url_b = []
                labels = list(map(self.get_label, sentence_pairs_indices))
                for sent_pair_indices in sentence_pairs_indices:
                    sentence_a.append(self.df.text.values[sent_pair_indices[0]])
                    sentence_b.append(self.df.text.values[sent_pair_indices[1]])
                    event_a.append(self.df.event.values[sent_pair_indices[0]])
                    event_b.append(self.df.event.values[sent_pair_indices[1]])
                    time_a.append(self.df.unix_timestamp.values[sent_pair_indices[0]])
                    time_b.append(self.df.unix_timestamp.values[sent_pair_indices[1]])
                    url_a.append(self.df.source.values[sent_pair_indices[0]].split("\'")[3])
                    url_b.append(self.df.source.values[sent_pair_indices[1]].split("\'")[3])
                df = pd.DataFrame(
                    list(zip(sentence_a, event_a, time_a, labels, sentence_b, event_b, time_b, url_a, url_b)),
                    columns=['sentence_a', 'event_a', 'time_a', 'labels', 'sentence_b', 'event_b', 'time_b', 'url_a',
                             'url_b'])
                df = df.loc[df["labels"] != "ignored"]
                print(df.labels.value_counts())

                df.to_csv(save_path)
        else:
            df = pd.read_csv(save_path)

        logger.info(f"Stratified samples length: {len(df)}).")

        return df

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
        print(f"The dataset{({self.task}-{self.data_type})} csv has {len(self.sampled_df)} entries.")
        print(f"     Number of clusters - {len(self.df.event.unique())}")

    def __getitem__(self, idx):
        return (self.df.sampled_df.sentence_a.values[idx], self.df.sampled_df.sentence_b.values[idx]), self.labels[idx]

    def __len__(self):
        return len(self.df.sampled_df)


if __name__ == "__main__":
    # train_csv_path = Path("./data/stormy_data/train_v1.csv")
    # valid_csv_path = Path("./data/stormy_data/valid_v1.csv")
    test_csv_path = Path("./data/stormy_data/test_v2.csv")
    # train_event_deduplication_storm = StormyDataset(train_csv_path, task="event_deduplication", data_type="train")
    # valid_event_deduplication_storm = StormyDataset(valid_csv_path, task="event_deduplication", data_type="valid")
    test_event_deduplication_storm = StormyDataset(test_csv_path, task="event_deduplication", data_type="test")
    print(f"test_event_deduplication_storm data \n {test_event_deduplication_storm.get_descriptions()}\n")

    split_crisisfacts_dataset()
    train_crisisfacts_csv_path = Path("./data/crisisfacts_data/crisisfacts_train.csv")
    valid_crisisfacts_csv_path = Path("./data/crisisfacts_data/crisisfacts_valid.csv")
    test_crisisfacts_csv_path = Path("./data/crisisfacts_data/crisisfacts_test.csv")
    test_crisisfacts_storm_csv_path = Path("./data/crisisfacts_data/crisisfacts_storm_test.csv")
    train_event_temporality_crisisfacts = CrisisFactsDataset(train_crisisfacts_csv_path, task="event_temporality", data_type="train")
    valid_event_temporality_crisisfacts = CrisisFactsDataset(valid_crisisfacts_csv_path, task="event_temporality", data_type="valid")
    test_event_temporality_crisisfacts = CrisisFactsDataset(test_crisisfacts_csv_path, task="event_temporality", data_type="test")
    test_event_temporality_crisisfacts_storm = CrisisFactsDataset(test_crisisfacts_storm_csv_path, task="event_temporality", data_type="test")

    print(f"test_event_temporality_crisisfacts_storm data \n {test_event_temporality_crisisfacts_storm.get_descriptions()}\n")
    split_crisisfacts_dataset()