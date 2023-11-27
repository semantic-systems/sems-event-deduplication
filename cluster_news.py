import numpy as np
from sentence_transformers import SentenceTransformer, util
import time
import pandas as pd
from pathlib import Path
from numpy import nan
import requests
from json import JSONDecodeError
from tqdm.notebook import tqdm


root = Path("./data/gdelt_crawled/")


class ClusterNews(object):
    def __init__(self, root_dir: str = "./data/gdelt_crawled/"):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.root = root_dir

    def cluster_news_per_disaster_subtype(self,
                                          batch_size: int = 512,
                                          min_community_size: int = 15,
                                          threshold: float = 0.75):
        cluster_col_name = f"cluster_{min_community_size}_{threshold * 100}"
        for event_type in Path(root).iterdir():
            if event_type.is_dir():
                path = Path(event_type, "aggregated_news_all_country.csv")
                try:
                    df = pd.read_csv(path)
                except FileNotFoundError:
                    continue
                df['title'] = df['title'].astype(str)
                corpus_embeddings = self.model.encode(df["title"].values, batch_size=batch_size,
                                                      show_progress_bar=True, convert_to_tensor=True)
                start_time = time.time()
                clusters = util.community_detection(corpus_embeddings,
                                                    min_community_size=min_community_size,
                                                    threshold=threshold)
                print("Clustering done after {:.2f} sec".format(time.time() - start_time))

                cluster_col = {}
                for i, cluster in enumerate(clusters):
                    cluster_i_dict = {sent_id: i for sent_id in cluster}
                    cluster_col.update(cluster_i_dict)
                    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
                    cluster_sentences = list(set([df["title"].values[sentence_id] for sentence_id in cluster[0:]]))

                    for s in cluster_sentences:
                        print(f"  {s}")
                df[cluster_col_name] = df.index.to_series().apply(lambda x: cluster_col.get(x, nan))
                df.to_csv(Path(event_type, f"clustered_news_all_country_{min_community_size}_{threshold * 100}.csv"), index=False)

    def cluster_news_all_events(self,
                                batch_size: int = 512,
                                min_community_size: int = 15,
                                threshold: float = 0.75,
                                drop_duplicates: bool = True):
        aggregated_news_all_event_path = Path(self.root, "aggregated_news_all_events.csv")
        clustered_news_all_event_path = Path(self.root, "clustered_news_all_events.csv")
        cluster_col_name = f"cluster_{min_community_size}_{threshold * 100}"
        if clustered_news_all_event_path.exists():
            df = pd.read_csv(clustered_news_all_event_path)
            if cluster_col_name in df.columns:
                print(f"Clustering with min_community_size ({min_community_size}) "
                      f"and threshold {threshold} already created.")

        elif aggregated_news_all_event_path.exists():
            df = pd.read_csv(aggregated_news_all_event_path)
            df['title'] = df['title'].astype(str)
            if drop_duplicates:
                df.drop_duplicates(subset='title', keep="last")
            corpus_embeddings = self.model.encode(df["title"].values, batch_size=batch_size,
                                                  show_progress_bar=True, convert_to_tensor=True)
            start_time = time.time()
            clusters = util.community_detection(corpus_embeddings,
                                                min_community_size=min_community_size,
                                                threshold=threshold)
            print("Clustering done after {:.2f} sec".format(time.time() - start_time))

            cluster_col = {}
            for i, cluster in enumerate(clusters):
                cluster_i_dict = {sent_id: i for sent_id in cluster}
                cluster_col.update(cluster_i_dict)
                print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
                cluster_sentences = list(set([df["title"].values[sentence_id] for sentence_id in cluster[0:]]))

                for s in cluster_sentences:
                    print(f"  {s}")
            df[cluster_col_name] = df.index.to_series().apply(lambda x: cluster_col.get(x, nan))
            df.to_csv(Path("./data/gdelt_crawled/clustered_news_all_events.csv"), index=False)

    def attach_silver_event_type_to_df(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df["title"] = df['title'].astype(str)
        all_event_types = []
        batch_size = 512
        num_iteration = int(np.ceil(len(df["title"].values)/batch_size))
        for i in tqdm(range(num_iteration)):
            start_index = batch_size*i
            end_index = batch_size*(i+1)
            event_types = self.run_coypu_ee(list(df["title"].values[start_index:end_index]))
            all_event_types.extend(event_types)
            if i % batch_size == 0:
                silvered_df = df.loc[:end_index-1, :]
                silvered_df.loc[:, ("pred_event_type")] = all_event_types
                silvered_df.to_csv(Path(self.root, "silvered_news_all_events.csv"), index=False)
        return all_event_types

    @staticmethod
    def run_coypu_ee(message):
        url = 'https://event-extraction.skynet.coypu.org'
        json_obj = {'message': message, "key": "32T82GWPSGDJTKFN"}
        x = requests.post(url, json=json_obj)
        try:
            prediction = x.json()
        except JSONDecodeError:
            prediction = {}
        return prediction.get("event type", [np.nan]*len(message))


if __name__ == "__main__":
    news_cluster = ClusterNews()
    news_cluster.__init__()
    news_cluster.cluster_news_all_events()
    news_cluster.attach_silver_event_type_to_df("./data/gdelt_crawled/clustered_news_all_events.csv")
