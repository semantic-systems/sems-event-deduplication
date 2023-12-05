import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import requests
from json import JSONDecodeError
import spacy


class EventDeduplicationDataFrame(object):
    def __init__(self, csv_path: str):
        self.raw_df = pd.read_csv(csv_path)
        self.root = Path("./data/gdelt_crawled/")
        self.nlp = spacy.load("en_core_web_md")
        self.nlp.add_pipe("entityLinker", last=True)

    def annotate_event_type(self):
        self.raw_df["title"] = self.raw_df['title'].astype(str)
        all_event_types = []
        batch_size = 512
        num_iteration = int(np.ceil(len(self.raw_df["title"].values) / batch_size))
        for i in tqdm(range(num_iteration)):
            start_index = batch_size * i
            end_index = batch_size * (i + 1)
            event_types = self.run_coypu_ee(list(self.raw_df["title"].values[start_index:end_index]))
            all_event_types.extend(event_types)
            if i % batch_size == 0:
                annotated_df = self.raw_df.loc[:end_index - 1, :]
                annotated_df.loc[:, ("pred_event_type")] = all_event_types
                annotated_df.to_csv(Path(self.root, "annotated_event_news_all_events.csv"), index=False)

    def annotate_entity(self):
        self.raw_df["title"] = self.raw_df['title'].astype(str)
        self.raw_df["entities"] = self.raw_df["title"].map(self.get_entity_from_spacy)
        self.raw_df.to_csv(Path(self.root, "annotated_entity_news_all_events.csv"), index=False)

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

    def create_silver_label(self):
        pass

    @staticmethod
    def remove_clusters_with_wrong_type(df):
        df_list = []
        for cluster_id in df["cluster_15_75.0"].unique():
            tmp_df = df.loc[df["cluster_15_75.0"] == cluster_id]
            if "tropical_storm" in tmp_df["pred_event_type"].values or "flood" in tmp_df["pred_event_type"].values:
                df_list.append(tmp_df)
        new_df = pd.concat(df_list)
        drop_df = new_df.drop(columns=['url_mobile', 'Unnamed: 0', 'seendate', 'socialimage', 'language'])
        drop_df = drop_df.drop_duplicates(subset='title', keep="last")
        drop_df.to_csv("silver_all_v1.csv")
        return drop_df

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

