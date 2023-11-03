from sentence_transformers import SentenceTransformer, util
import time
import pandas as pd
from pathlib import Path
from numpy import nan

root = Path("./data/gdelt_crawled/")
model = SentenceTransformer('all-MiniLM-L6-v2')

for event_type in Path(root).iterdir():
    if event_type.is_dir():
        path = Path(event_type, "aggregated_news_all_country.csv")
        try:
            df = pd.read_csv(path)
        except FileNotFoundError:
            continue 
        df['title'] = df['title'].astype(str)
        corpus_embeddings = model.encode(df["title"].values, batch_size=128, show_progress_bar=True, convert_to_tensor=True)
        start_time = time.time()
        clusters = util.community_detection(corpus_embeddings, min_community_size=25, threshold=0.75)
        print("Clustering done after {:.2f} sec".format(time.time() - start_time))

        cluster_col = {}
        for i, cluster in enumerate(clusters):
            cluster_i_dict = {sent_id: i for sent_id in cluster}
            cluster_col.update(cluster_i_dict)
            print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
            cluster_sentences = list(set([df["title"].values[sentence_id] for sentence_id in cluster[0:]]))

            for s in cluster_sentences:
                print(f"  {s}")
        df['cluster_25_75'] = df.index.to_series().apply(lambda x: cluster_col.get(x, nan))
        df.to_csv(Path(event_type, "clustered_news_all_country.csv"), index=False)