import pandas as pd
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from sentence_transformers import SentenceTransformer
from transformers import TrainingArguments, RobertaForSequenceClassification, AutoTokenizer, Trainer, RobertaConfig
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn import preprocessing

df = pd.read_csv("event_duration_prediction_dataset.csv")
df["regression_label"] = df["regression_label"].astype(float)
lab_enc = preprocessing.LabelEncoder()
df['regression_label'] = lab_enc.fit_transform(df['regression_label'])

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

df['embeddings'] = df['text'].apply(embedding_model.encode)

train_df = df.loc[df["event_type"].isin(["Hurricane Florence 2018"
						, "Hurricane Sally 2020"
					])]
valid_df = df.loc[df["event_type"].isin(["Hurricane Laura 2020"
						, "Saddleridge Wildfire 2019"
					])]
test_df = df.loc[df["event_type"].isin(
    ["2018 Maryland Flood"
	, "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"
	])]

d = {'train': Dataset.from_dict(
    {'text': train_df['embeddings'].values.tolist()
		, 'label': train_df['regression_label'].values.tolist()
	}),
    'test': Dataset.from_dict(
        {'text': test_df['embeddings'].values.tolist()
		, 'label': test_df['regression_label'].values.tolist()
	}),
    'validation': Dataset.from_dict(
        {'text': valid_df['embeddings'].values.tolist()
		, 'label': valid_df['regression_label'].values.tolist()
	})
}

dataset = DatasetDict(d)

train_dataset = dataset["train"].shuffle(seed=42)
test_dataset = dataset["test"].shuffle(seed=42)
eval_dataset = dataset["validation"].shuffle(seed=42)

LR = LogisticRegression(max_iter=1200000)
LR.fit(train_dataset['text'],train_dataset['label'])

predicted = LR.predict(test_dataset['text'])
print("Logistic Regression Accuracy:",metrics.accuracy_score(test_dataset['label'], predicted))
print("Logistic Regression Precision:",metrics.precision_score(test_dataset['label'], predicted, average='micro'))
print("Logistic Regression Recall:",metrics.recall_score(test_dataset['label'], predicted, average='micro'))
