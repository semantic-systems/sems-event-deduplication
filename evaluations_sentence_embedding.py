import pandas as pd
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from sentence_transformers import SentenceTransformer
from transformers import TrainingArguments, RobertaForSequenceClassification, AutoTokenizer, Trainer, RobertaConfig
from sklearn.metrics import mean_squared_error


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}



df = pd.read_csv("event_duration_prediction_dataset.csv")
df["regression_label"] = df["regression_label"].astype(float)
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

max_length = df["text"].apply(lambda x: len(x)).max()

tokenizer = AutoTokenizer.from_pretrained("roberta-base", ignore_mismatched_sizes=True, model_max_length=max_length)

def tokenize_function(entry):
    return tokenizer(entry, padding="max_length", truncation=True)

#df["tokenize"] = df["text"].apply(tokenize_function)

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
    {'text': train_df['embeddings'].values #.tolist()
		, 'label': train_df['regression_label'].values #.tolist()
	}),
    'test': Dataset.from_dict(
        {'text': test_df['embeddings'].values #.tolist()
		, 'label': test_df['regression_label'].values #.tolist()
	}),
    'validation': Dataset.from_dict(
        {'text': valid_df['embeddings'].values #.tolist()
		, 'label': valid_df['regression_label'].values #.tolist()
	})
}

dataset = DatasetDict(d)

#tokenized_datasets = dataset.map(tokenize_function, batched=True)
#tokenized_datasets = tokenized_datasets.map(embedding_model.encode, batched=True)



train_dataset = dataset["train"].shuffle(seed=42)
test_dataset = dataset["test"].shuffle(seed=42)
eval_dataset = dataset["validation"].shuffle(seed=42)


#train_dataset["embeddings"] = train_dataset["text"].apply(embedding_model.encode)
#test_dataset["embeddings"] = test_dataset["text"].apply(embedding_model.encode)
#eval_dataset["embeddings"] = eval_dataset["text"].apply(embedding_model.encode)



model = RobertaForSequenceClassification.from_pretrained("roberta-base",
                                                            num_labels=1,
                                                            problem_type = 'regression',
                                                            ignore_mismatched_sizes=True)#.to("cuda")


batch_size = 32

training_args = TrainingArguments(
    output_dir="model-training",
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
)

trainer = Trainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

trainer.train()

print('---------------------------------------------------------------------------------------------')

pred = trainer.predict(test_dataset)
print(pred.metrics)
