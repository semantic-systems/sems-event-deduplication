from sentence_transformers import SentenceTransformer, util
import time
import pandas as pd
from pathlib import Path

root = Path("./data/gdelt_crawled/")
model = SentenceTransformer('all-MiniLM-L6-v2')

for event_type in Path(root).iterdir():
    if event_type.is_dir():
        path = Path(event_type, "aggregated_news_all_country.csv")
        df = pd.read_csv(path)

        filtered_df = df
        filtered_df['title'] = filtered_df['title'].astype(str)
        corpus_embeddings = model.encode(filtered_df["title"].values, batch_size=128, show_progress_bar=True, convert_to_tensor=True)
        start_time = time.time()
        clusters = util.community_detection(corpus_embeddings, min_community_size=25, threshold=0.75)
        print("Clustering done after {:.2f} sec".format(time.time() - start_time))

        for i, cluster in enumerate(clusters):
            print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
            cluster_sentences = list(set([filtered_df["title"].values[sentence_id] for sentence_id in cluster[0:]]))
            for s in cluster_sentences:
                print(f"  {s}")