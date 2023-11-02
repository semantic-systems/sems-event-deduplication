from sentence_transformers import SentenceTransformer, util
import time
import pandas as pd
from pathlib import Path


df = pd.read_csv("./data/gdelt_crawled/gdelt_crawled/storm/aggregated_news_all_country.csv")
df['start_date'] = pd.to_datetime(df['start_date'], format='%Y-%m-%d')

model = SentenceTransformer('all-MiniLM-L6-v2')
# filtered_df = df.loc[(df['start_date'] >= '2021-01-03') & (df['start_date'] <= '2021-01-17')]
# print(filtered_df)
filtered_df = df
corpus_embeddings = model.encode(filtered_df["title"].values, batch_size=64, show_progress_bar=True, convert_to_tensor=True)
start_time = time.time()
clusters = util.community_detection(corpus_embeddings, min_community_size=25, threshold=0.75)
print("Clustering done after {:.2f} sec".format(time.time() - start_time))

for i, cluster in enumerate(clusters):
    print("\nCluster {}, #{} Elements ".format(i+1, len(cluster)))
    cluster_sentences = list(set([filtered_df["title"].values[sentence_id] for sentence_id in cluster[0:]]))
    for s in cluster_sentences:
        print(f"  {s}")