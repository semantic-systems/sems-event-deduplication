import pandas as pd
import os
import glob
from pathlib import Path


root = Path("./data/gdelt_crawled/gdelt_crawled/")
for event_type in Path(root).iterdir():
    if event_type.is_dir():
        for country in Path(event_type).iterdir():
            aggregated_news_per_country = []
            for csvs in country.rglob("*.csv"):
                if csvs.name not in ["peaks_timeframe.csv", "aggregated_news.csv"]:
                    start_date = csvs.stem.split("_")[0]
                    end_date = csvs.stem.split("_")[1].split(".")[0]
                    df = pd.read_csv(csvs, index_col=False, header=0)
                    df["start_date"] = start_date
                    df["end_date"] = end_date
                    aggregated_news_per_country.append(df)
            if aggregated_news_per_country:
                aggregated_df_per_country = pd.concat(aggregated_news_per_country, axis=0, ignore_index=True)
                aggregated_path = Path(*csvs.parts[:-1], "aggregated_news.csv")
                aggregated_df_per_country.to_csv(aggregated_path, index=False)
