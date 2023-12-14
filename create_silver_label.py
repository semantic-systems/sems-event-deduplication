import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import requests
import time
from sentence_transformers import SentenceTransformer, util
from json import JSONDecodeError
import spacy
from numpy import nan
from sklearn.cluster import DBSCAN
from event_data_processing import NaturalDisasterGdelt


class EventDeduplicationDataFrame(object):
    def __init__(self, csv_path: str = None, aggregate_news: bool = False):
        self.root = Path("./data/gdelt_crawled/")
        self.aggregated_news_all_event_path = Path(self.root, "aggregated_news_all_events.csv")
        if not aggregate_news and csv_path is None and not self.aggregated_news_all_event_path.exists():
            aggregate_news = True
        if aggregate_news:
            self.aggregate_news()
            self.df = pd.read_csv(self.aggregated_news_all_event_path)
        elif csv_path:
            self.df = pd.read_csv(csv_path)
        else:
            self.df = pd.read_csv(self.aggregated_news_all_event_path)

        self.target_df_col = [
            'cluster_20_60', 'cluster_20_70', 'cluster_20_80', 'cluster_20_90',
            'cluster_50_60', 'cluster_50_70', 'cluster_50_80', 'cluster_50_90',
            'cluster_100_60', 'cluster_100_70', 'cluster_100_80', 'cluster_100_90',
            'cluster_5_60', 'cluster_5_70', 'cluster_5_80', 'cluster_5_90',
            'temporal_cluster_20_60', 'temporal_cluster_20_70', 'temporal_cluster_20_80', 'temporal_cluster_20_90',
            'temporal_cluster_50_60', 'temporal_cluster_50_70', 'temporal_cluster_50_80', 'temporal_cluster_50_90',
            'temporal_cluster_100_60', 'temporal_cluster_100_70', 'temporal_cluster_100_80', 'temporal_cluster_100_90',
            'temporal_cluster_5_60', 'temporal_cluster_5_70', 'temporal_cluster_5_80', 'temporal_cluster_5_90',
            'pred_event_type', 'entities']

        self.nlp = self.instantiate_spacy()

    def instantiate_spacy(self):
        if 'entities' not in self.df.columns:
            nlp = spacy.load("en_core_web_md")
            nlp.add_pipe("entityLinker", last=True)
        else:
            nlp = None
        return nlp

    def create_silver_label(self):
        df = self.df
        df['title'] = df['title'].astype(str)
        df['start_date'] = df['start_date'].astype(str)

        print(f"Raw dataset - Number of entires: {len(df)}")
        df = self.remove_stick_in_title(df)
        print(f"Stick replaced dataset - Number of entires: {len(df)}")
        df = df.drop_duplicates(subset='title', keep="last")
        print(f"Dropped duplicates dataset - Number of entires: {len(df)}")
        print(f"Denoising dataset with hierarchical clustering...")
        print(f"")
        print("    Annotating event type with a trained event detector on TREC-IS dataset...")
        df = self.annotate_event_type(df, forced=False)
        print("    Annotating entities and links to wikidata...")
        df = self.annotate_entity(df, forced=False)
        print("    Clustering all news titles with sentence bert...")
        df = self.cluster_titles(df, forced=True)
        print("    Running temporal 1d DBSCAN to remove similar, but temporally far news titles...")
        df, df_outliers = self.run_temporal_clustering(df, min_samples=3, eps=1, forced=True)
        print(f"Temporally valid dataset - Number of entires: {len(df)}")
        df, df_oos = self.remove_oos_clusters(df, forced=True)
        print(f"OOS removed dataset - Number of entires: {len(df)}")

    @staticmethod
    def aggregate_news():
        gdelt_news = NaturalDisasterGdelt()
        gdelt_news.__int__()
        gdelt_news.aggregate_extracted_news()

    def annotate_event_type(self, df, forced=False):
        if not forced and "pred_event_type" in df.columns:
            return df
        else:
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
        if not forced and "entities" in df.columns:
            return df
        else:
            df["title"] = df['title'].astype(str)
            df["entities"] = df["title"].map(self.get_entity_from_spacy)
            df.to_csv(Path(self.root, "annotated_entity_news_all_events.csv"), index=False)
            self.nlp = None  # save memory
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
        return prediction.get("event type", [nan] * len(message))

    @staticmethod
    def remove_stick_in_title(df):
        title_indicese_with_a_stick = [i for i, title in enumerate(df["title"].values) if "|" in title]
        title_wo_stick = []
        for title in tqdm(df.loc[title_indicese_with_a_stick, "title"]):
            title_parts_len = [len(t) for t in title]
            longest_part_index = title_parts_len.index(max(title_parts_len))
            title_wo_stick.append(title[longest_part_index])
        df.loc[title_indicese_with_a_stick, "title"] = title_wo_stick
        return df

    @staticmethod
    def combine_columns(row):
        return str(row['title']) + " (" + row['start_date'] + ")"

    def cluster_titles(self,
                       df,
                       batch_size: int = 512,
                       forced=False):
        model = SentenceTransformer('all-MiniLM-L6-v2')
        df['temporal_title'] = df.apply(self.combine_columns, axis=1)
        cluster_cols = [col for col in self.target_df_col if "cluster" in col and "temporal" not in col]
        temporal_cluster_cols = [col for col in self.target_df_col if "temporal_cluster" in col]
        clustering_params = [col for col in cluster_cols if col not in df.columns]
        temporal_clustering_params = [col for col in temporal_cluster_cols if col not in df.columns]

        # cluster titles
        corpus_embeddings = model.encode(df["title"].values, batch_size=batch_size,
                                         show_progress_bar=True, convert_to_tensor=True)
        for params in tqdm(clustering_params):
            min_community_size = int(params.split("_")[-2])
            threshold = float(params.split("_")[-1])/100
            cluster_col_name = f"cluster_{min_community_size}_{str(threshold * 100)[:2]}"
            start_time = time.time()
            print(f"Start clustering (min_community_size={min_community_size}, threshold={threshold}) ...")
            clusters = util.community_detection(corpus_embeddings,
                                                min_community_size=min_community_size,
                                                threshold=threshold)
            print(f"Clustering (min_community_size={min_community_size}, threshold={threshold}) done after {time.time()-start_time} sec")

            cluster_col = {}
            for i, cluster in enumerate(clusters):
                cluster_i_dict = {sent_id: i for sent_id in cluster}
                cluster_col.update(cluster_i_dict)

            df[cluster_col_name] = df.index.to_series().apply(lambda x: cluster_col.get(x, nan))
            df.to_csv(Path(self.root, "clustered_news_all_events.csv"), index=False)

        # temporal cluster titles
        corpus_embeddings = model.encode(df["temporal_title"].values, batch_size=batch_size,
                                         show_progress_bar=True, convert_to_tensor=True)

        for params in tqdm(temporal_clustering_params):
            min_community_size = int(params.split("_")[-2])
            threshold = float(params.split("_")[-1]) / 100
            cluster_col_name = f"cluster_{min_community_size}_{str(threshold * 100)[:2]}"
            start_time = time.time()
            print(f"Start temporal clustering (min_community_size={min_community_size}, threshold={threshold}) ...")
            clusters = util.community_detection(corpus_embeddings,
                                                min_community_size=min_community_size,
                                                threshold=threshold)
            print(
                f"Temporal clustering (min_community_size={min_community_size}, threshold={threshold}) done after {time.time() - start_time} sec")

            cluster_col = {}
            for i, cluster in enumerate(clusters):
                cluster_i_dict = {sent_id: i for sent_id in cluster}
                cluster_col.update(cluster_i_dict)

            df[cluster_col_name] = df.index.to_series().apply(lambda x: cluster_col.get(x, nan))
            df.to_csv(Path(self.root, "clustered_news_all_events.csv"), index=False)
        return df

    def run_temporal_clustering(self, df, min_samples=3, eps=1, forced=False):
        if not forced and Path(self.root, "temporally_denoised_news.csv").exists() and Path(self.root, "temporally_noisy_news.csv").exists():
            return df, None
        else:
            df['start_date'] = pd.to_datetime(df['start_date'])
            df_removed_outliers = []
            df_outliers = []

            for i, cluster in tqdm(enumerate(df["cluster_100_75"].unique())):
                cluster_i = df.loc[df["cluster_100_75"] == cluster]
                cluster_i["normalized_date"] = (cluster_i["start_date"] - min(
                    cluster_i["start_date"])) / np.timedelta64(1, 'D')
                clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(np.reshape(cluster_i["normalized_date"].values, (-1, 1)))

                outlier_indices = np.where(clustering.labels_ == -1)[0]
                # outlier_dates = cluster_i["start_date"].values[outlier_indices]
                # outlier_title = cluster_i["title"].values[outlier_indices]
                # outlier_pred_event_type = cluster_i["pred_event_type"].values[outlier_indices]
                outlier_df = cluster_i.iloc[outlier_indices, :]
                df_outliers.append(outlier_df)

                most_populated_cluster = self.most_common(list(clustering.labels_))
                if most_populated_cluster != -1:
                    most_populated_cluster_indices = np.where(clustering.labels_ == most_populated_cluster)[0]
                    # most_populated_cluster_dates = cluster_i["start_date"].values[most_populated_cluster_indices]
                    # most_populated_cluster_title = cluster_i["title"].values[most_populated_cluster_indices]
                    # most_populated_cluster_pred_event_type = cluster_i["pred_event_type"].values[
                    #     most_populated_cluster_indices]
                    most_popular_cluster_df = cluster_i.iloc[most_populated_cluster_indices, :]
                    df_removed_outliers.append(most_popular_cluster_df)

                    # print(
                    #     f"Cluster {cluster}: \n Outlier dates: {outlier_dates}\n Outlier title: {outlier_title}\n Outlier pred_event_type: {outlier_pred_event_type} \n most popular cluster: {most_populated_cluster}\n most popular cluster dates: {most_populated_cluster_dates}\n most popular cluster title: {most_populated_cluster_title}\n most popular cluster pred_event_type: {most_populated_cluster_pred_event_type}\n\n")
            df_removed_outliers = pd.concat(df_removed_outliers, ignore_index=True)
            df_outliers = pd.concat(df_outliers, ignore_index=True)
            df_removed_outliers.to_csv(Path(self.root, "temporally_denoised_news.csv"), index=False)
            df_outliers.to_csv(Path(self.root, "temporally_noisy_news.csv"), index=False)
            return df_removed_outliers, df_outliers

    @staticmethod
    def most_common(lst):
        return max(set(lst), key=lst.count)

    def remove_oos_clusters(self, df, forced=False):
        if not forced and Path(self.root, "oos_removed_news.csv").exists() and Path(self.root, "oos_news.csv").exists():
            return df
        else:
            df_list = []
            oos_df_list = []
            for cluster_id in tqdm(df["cluster_100_75"].unique()):
                unique_clusters = df.loc[df["cluster_100_75"] == cluster_id, "pred_event_type"].unique()
                if not (len(unique_clusters) == 1 and unique_clusters[0] == 'oos'):
                    df_list.append(df.loc[df["cluster_100_75"] == cluster_id])
                else:
                    oos_df_list.append(df.loc[df["cluster_100_75"] == cluster_id])
            oos_removed_df = pd.concat(df_list, ignore_index=True)
            oos_removed_df.to_csv(Path(self.root, "oos_removed_news.csv"), index=False)
            oos_df = pd.concat(oos_df_list, ignore_index=True)
            oos_df.to_csv(Path(self.root, "oos_news.csv"), index=False)
            return oos_removed_df, oos_df

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
    dataset = EventDeduplicationDataFrame("./data/gdelt_crawled/clustered_news_all_events.csv")
    dataset.create_silver_label()
    # dataset.annotate_entity()

