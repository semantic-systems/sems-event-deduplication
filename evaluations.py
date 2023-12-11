import pandas as pd
import torch
import numpy as np
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from pytorch_lightning.loggers import WandbLogger

import wandb
from torch.utils.data import DataLoader
from enum import Enum
from pytorch_lightning import LightningModule  # , Trainer
import evaluate
from transformers import TrainingArguments, RobertaForSequenceClassification, AutoTokenizer, Trainer
from sklearn.metrics import mean_squared_error

df = pd.read_csv("event_duration_prediction_dataset.csv")
#df["regression_label"] = df["regression_label"].astype(float)
label_enum = {k:j for j, k in enumerate(df['regression_label'].unique())}
number_labels = len(label_enum)
df['regression_label'] = df['regression_label'].apply(lambda x: [1.0 if label_enum[x]==i else 0.0 for i in range(number_labels)])
max_length = df.apply(lambda x: len(x)).max()
train_df = df.loc[df["event_type"].isin(["Hurricane Florence 2018"
						#, "Hurricane Sally 2020"
					])]
valid_df = df.loc[df["event_type"].isin(["Hurricane Laura 2020"
						#, "Saddleridge Wildfire 2019"
					])]
test_df = df.loc[df["event_type"].isin(
    ["2018 Maryland Flood"
	#, "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"
	])]
tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=max_length)


def tokenize_function(entry):
    return tokenizer(entry["text"], padding="max_length", truncation=True)


d = {'train': Dataset.from_dict(
    {'text': train_df['text'].values.tolist(), 'label': train_df['regression_label'].values.tolist()}),
    'test': Dataset.from_dict(
        {'text': test_df['text'].values.tolist(), 'label': test_df['regression_label'].values.tolist()}),
    'validation': Dataset.from_dict(
        {'text': valid_df['text'].values.tolist(), 'label': valid_df['regression_label'].values.tolist()})
}

dataset = DatasetDict(d)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
test_dataset = tokenized_datasets["test"].shuffle(seed=42)
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)


# Make simple Enum for code clarity
class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3


f1_metric = evaluate.load("f1")
recall_metric = evaluate.load("recall")
precision_metric = evaluate.load("precision")
accuracy_metric = evaluate.load("accuracy")



def compute_metrics(eval_pred):
    '''
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=-1)

    results = {}
    results.update(f1_metric.compute(predictions=predictions, references=labels))
    results.update(recall_metric.compute(predictions=predictions, references=labels))
    results.update(precision_metric.compute(predictions=predictions, references=labels))
    results.update(accuracy_metric.compute(predictions=predictions, references=labels))

    return results
    '''
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}


class TrainerTransformer(LightningModule):
    def __init__(self, vocab_size):
        super().__init__()
        self.model = RobertaForSequenceClassification.from_pretrained("roberta-base",
                                                                      num_labels=vocab_size)  # .to("cuda")

    def forward(self, inputs, target):
        return self.model(inputs, target)

    def training_step(self, batch, batch_idx):
        inputs, target = batch
        output = self(inputs, target)
        loss = torch.nn.functional.mse_loss(output, target.view(-1))
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.model.parameters(), lr=0.1)


'''

model = TrainerTransformer(number_labels)
wandb_logger = WandbLogger(log_model="all")
dataloader = DataLoader(train_dataset)
trainer = Trainer(#logger=wandb_logger,
                  max_epochs=3)
trainer.fit(model=model, train_dataloaders=dataloader)
'''

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=number_labels, problem_type = 'regression').to("cuda")


training_args = TrainingArguments(
    output_dir="model-training",
    evaluation_strategy="epoch",
    num_train_epochs=1
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
