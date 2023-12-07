import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import requests
from json import JSONDecodeError
import spacy
from sklearn.cluster import DBSCAN


class EventDeduplicationDataFrame(object):
    def __init__(self, csv_path: str):
        self.raw_df = pd.read_csv(csv_path)
        self.root = Path("./data/gdelt_crawled/")
        self.nlp = spacy.load("en_core_web_md")
        self.nlp.add_pipe("entityLinker", last=True)

    def annotate_event_type(self, df, forced=False):
        if not forced and Path(self.root, "annotated_entity_news_all_events.csv").exists():
            return df
        else:
            df["title"] = df['title'].astype(str)
            all_event_types = []
            batch_size = 512
            num_iteration = int(np.ceil(len(df["title"].values) / batch_size))
            for i in tqdm(range(num_iteration)):
                start_index = batch_size * i
                end_index = batch_size * (i + 1)
                event_types = self.run_coypu_ee(list(df["title"].values[start_index:end_index]))
                all_event_types.extend(event_types)
                if i % batch_size == 0:
                    annotated_df = df.loc[:end_index - 1, :]
                    annotated_df.loc[:, ("pred_event_type")] = all_event_types
                    annotated_df.to_csv(Path(self.root, "annotated_event_news_all_events.csv"), index=False)
            return df
    def annotate_entity(self, df, forced=False):

        if not forced and Path(self.root, "annotated_entity_news_all_events.csv").exists():
            return df
        else:
            df["title"] = df['title'].astype(str)
            df["entities"] = df["title"].map(self.get_entity_from_spacy)
            df.to_csv(Path(self.root, "annotated_entity_news_all_events.csv"), index=False)
            return df

    def get_entity_from_spacy(self, text: str):
        entity_info = {"entity_type": {},
                       "linked_entitiy": {}}
        doc = self.nlp(text)
        for entity in doc.ents:
            if entity.text not in entity_info["entity_type"]:
                entity_info["entity_type"][entity.text] = entity.label_

        all_linked_entities = doc._.linkedEntities
        for entity in all_linked_entities.entities:
            if entity.original_alias not in entity_info["linked_entitiy"]:
                entity_info["linked_entitiy"][entity.original_alias] = entity.url
        return entity_info

    @staticmethod
    def run_coypu_ee(message):
        url = 'https://event-extraction.skynet.coypu.org'
        json_obj = {'message': message, "key": "32T82GWPSGDJTKFN"}
        x = requests.post(url, json=json_obj)
        try:
            prediction = x.json()
        except JSONDecodeError:
            prediction = {}
        return prediction.get("event type", [np.nan] * len(message))

    def create_silver_label(self, df):
        df = df.drop_duplicates(subset='title', keep="last")
        df = self.annotate_event_type(df, forced=False)
        df = self.annotate_entity(df, forced=False)
        df_temporally_clean, df_outliers = self.run_temporal_clustering(df, min_samples=3, eps=1, forced=True)
        df_oos_removed = self.remove_oos_clusters(df_temporally_clean)
        pass

    def run_temporal_clustering(self, df, min_samples=3, eps=1, forced=False):
        if not forced and Path(self.root, "temporally_denoised_df.csv").exists() and Path(self.root, "temporally_noisy_df.csv").exists():
            return df
        else:
            df['start_date'] = pd.to_datetime(df['start_date'])
            df_removed_outliers = []
            df_outliers = []

            for i, cluster in enumerate(df["cluster_15_75.0"].unique()):
                cluster_i = df.loc[df["cluster_15_75.0"] == cluster]
                cluster_i["normalized_date"] = (cluster_i["start_date"] - min(
                    cluster_i["start_date"])) / np.timedelta64(1, 'D')
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np.reshape(cluster_i["normalized_date"].values, (-1, 1)))

                outlier_indices = np.where(clustering.labels_ == -1)[0]
                outlier_dates = cluster_i["start_date"].values[outlier_indices]
                outlier_title = cluster_i["title"].values[outlier_indices]
                outlier_pred_event_type = cluster_i["pred_event_type"].values[outlier_indices]
                outlier_df = cluster_i.iloc[outlier_indices, :]
                df_outliers.append(outlier_df)

                most_populated_cluster = self.most_common(list(clustering.labels_))
                if most_populated_cluster != -1:
                    most_populated_cluster_indices = np.where(clustering.labels_ == most_populated_cluster)[0]
                    most_populated_cluster_dates = cluster_i["start_date"].values[most_populated_cluster_indices]
                    most_populated_cluster_title = cluster_i["title"].values[most_populated_cluster_indices]
                    most_populated_cluster_pred_event_type = cluster_i["pred_event_type"].values[
                        most_populated_cluster_indices]
                    most_popular_cluster_df = cluster_i.iloc[most_populated_cluster_indices, :]
                    df_removed_outliers.append(most_popular_cluster_df)

                    print(
                        f"Cluster {cluster}: \n Outlier dates: {outlier_dates}\n Outlier title: {outlier_title}\n Outlier pred_event_type: {outlier_pred_event_type} \n most popular cluster: {most_populated_cluster}\n most popular cluster dates: {most_populated_cluster_dates}\n most popular cluster title: {most_populated_cluster_title}\n most popular cluster pred_event_type: {most_populated_cluster_pred_event_type}\n\n")
            df_removed_outliers = pd.concat(df_removed_outliers, ignore_index=True)
            df_outliers = pd.concat(df_outliers, ignore_index=True)
            df_removed_outliers.to_csv("temporally_denoised_df.csv", index=False)
            df_outliers.to_csv("temporally_noisy_df.csv", index=False)
            return df_removed_outliers, df_outliers

    @staticmethod
    def most_common(lst):
        return max(set(lst), key=lst.count)

    def remove_oos_clusters(self, df, forced=False):
        if not forced and Path(self.root, "oos_removed_df.csv").exists() and Path(self.root, "oos_df.csv").exists():
            return df
        else:
            df_list = []
            oos_df_list = []
            for cluster_id in df["cluster_15_75.0"].unique():
                unique_clusters = df.loc[df["cluster_15_75.0"] == cluster_id, "pred_event_type"].unique()
                if not (len(unique_clusters) == 1 and unique_clusters[0] == 'oos'):
                    df_list.append(df.loc[df["cluster_15_75.0"] == cluster_id])
                else:
                    oos_df_list.append(df.loc[df["cluster_15_75.0"] == cluster_id])
            oos_removed_df = pd.concat(df_list, ignore_index=True)
            oos_removed_df.to_csv("oos_removed_df.csv")
            oos_df = pd.concat(oos_df_list, ignore_index=True)
            oos_df.to_csv("oos_df.csv")
            return oos_removed_df

    def compare_predicted_event_type_with_gdelt_keyword(self):
        of_interest_predicted_event_type = ["tropical_storm", "flood"]
        pass

    def compare_time(self):
        pass

    def compare_location(self):
        pass

    def compare_entities(self):
        pass

    def merge_cluster(self):
        pass

    def remove_cluster(self):
        pass

    def remove_instance_from_cluster(self):
        pass

    def link_cluster_to_wikidata_events(self):
        ## check for event dates
        ## check for location
        ## check for entity
        pass
    
    def evaluate_on_trec_is(self):
        pass


if __name__ == "__main__":
    spacy.prefer_gpu()
    dataset = EventDeduplicationDataFrame("./data/gdelt_crawled/silvered_news_all_events.csv")
    # dataset.annotate_event_type()
    # dataset.annotate_entity()

