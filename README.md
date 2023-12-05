# Dataset Creation Pipeline
1. Crawl news title (including events and non-events) from gdelt
2. Aggregate news per event type
3. Cluster news titles 
   - feature: S-BERT embeddings
4. Annotate news instances
    - Event types with coypu event extractor
    - Entities from spacy entity linker
5. Filter news instances
    - by news publication time
    - by event type from coypu ee
    - by entity mentions (location, name entities)
6. Remove/ merge clusters
    - by matching entity mentions 
7. Create silver labels (majority label of each cluster)
8. Evaluate on crisisfact gold standard dataset 

## De-noising Procedures
The crawled gdelt news are noisy, 
which is natural in the regex-based keyword search used in gdelt. 
Examples of noisy instances are followed:
   1. Image captions of a link in the main page mentioning the search keywords
   2. Usage of the search keyword appears in context with pragmatical difference,
      - A flood of doubt arrives at the White House.
      - An Earthquaking speech by president Trump.

### How to detect and remove them
for each title,
1. run a community detection algorithm which computes cosine similarity to each pair of sentences, then creats clusters by thresholding the minimal similarity score to be in one cluster.
2. annotate event type with an event type detector that is trained to detect tropical storm events, which in this benchmark is considered as flooding, hurricane, tornado, tsunami and (tropical) storm. Other events include earthquakes, explosions, wildfire and drought.
3. annotate linked entities with spacy linker,
4. remove clusters with only oos predictions (benchmark bias 1: false negatives of event detector)
5. set clusters with only one type of predicted event as easy cluster, for this the best is to be coupled with a high threshold for creating clusters.
6. compute entropy of the predicted event type distribution for each cluster. 
7. define cluster with 0 entropy and with a whitelisted predicted event type as easy data 
silver_labeled_clusters = {}

for e in event_type:
    for i, c in enumerate(cluster):
        if all(is_event_type(e, c)):
            silver_labeled_clusters[i] = e

9. discard 0 energy with oos
10. mid entropy and without oos as alright data. mid energy with oos move to suspect_cluster_list
11. high energy without oos hard samples


# Task Descriptions
## Event Clustering
### Task Definition
Given a set of news article's titles, 

## Event Duration Prediction

### Task Definition
Event duration prediction is a regression task.
Given a title of a crisis-related news article, 
the model needs to predict a value between 0 and 1, 
describing the temporality of the event instance, 
that this new article's title is referring to.
ß represents the beginning of this event instance and 1 the end. 


### Baseline Models
1. Weak Baseline: fine-tuned s-bert
2. Strong Baseline: fine-tuned s-bert with enriched information from wikidata, constructed event KGs (that grow over event temporal development, indicated by the number of entities in the graph, and complexity of the graph per event instance. Such event KGs contains also event type, entity type, temporal relations and their values); with event graph size estimated news articles volume; and event type predicted by an event type detector trained on TREC-IS.)

**Event duration prediction is a regression task that captures event temporality from a narrative perspective. More specifically, this task investigates how events are addressed in the tone of reporting. It showcases how discourse of a story can be modelled.**