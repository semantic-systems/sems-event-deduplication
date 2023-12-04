# sems-event-deduplication
Event data processing pipeline:
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
6. remove/ merge clusters
7. evaluate on crisisfact golden-labeled dataset 
