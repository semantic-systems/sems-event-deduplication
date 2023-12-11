import pandas as pd
import torch
import numpy as np
#from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from pytorch_lightning.loggers import WandbLogger
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from enum import Enum
from pytorch_lightning import LightningModule  # , Trainer
import evaluate
from transformers import TrainingArguments, RobertaForSequenceClassification, AutoTokenizer, Trainer, RobertaConfig
from sklearn.metrics import mean_squared_error

'''
df = pd.read_csv("event_duration_prediction_dataset.csv")
df["regression_label"] = df["regression_label"].astype(float)
max_length = df.apply(lambda x: len(x)).max()
train_df = df.loc[df["event_type"].isin(["Hurricane Florence 2018"
#						, "Hurricane Sally 2020"
					])]
valid_df = df.loc[df["event_type"].isin(["Hurricane Laura 2020"
#						, "Saddleridge Wildfire 2019"
					])]
test_df = df.loc[df["event_type"].isin(
    ["2018 Maryland Flood"
#	, "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"
	])]
tokenizer = AutoTokenizer.from_pretrained("roberta-base", model_max_length=300)

def tokenize_function(entry):
    return tokenizer(entry["text"], padding="max_length", truncation=True)


d = {'train': Dataset.from_dict(
    {'text': train_df['text'].values #.tolist()
		, 'label': train_df['regression_label'].values #.tolist()
	}),
    'test': Dataset.from_dict(
        {'text': test_df['text'].values #.tolist()
		, 'label': test_df['regression_label'].values #.tolist()
	}),
    'validation': Dataset.from_dict(
        {'text': valid_df['text'].values #.tolist()
		, 'label': valid_df['regression_label'].values #.tolist()
	})
}

dataset = DatasetDict(d)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = tokenized_datasets["train"].shuffle(seed=42)
test_dataset = tokenized_datasets["test"].shuffle(seed=42)
eval_dataset = tokenized_datasets["validation"].shuffle(seed=42)
'''

tokenizer = AutoTokenizer.from_pretrained("roberta-base", ignore_mismatched_sizes=True, model_max_length=300)

def tokenize_function(entry):
    return tokenizer(entry["text"], padding="max_length", truncation=True)

class EventDurationPredictionDataset(Dataset):
    event_dict = {"001": "Lilac Wildfire 2017",
                  "002": "Cranston Wildfire 2018",
                  "003": "Holy Wildfire 2018",
                  "004": "Hurricane Florence 2018",
                  "005": "2018 Maryland Flood",
                  "006": "Saddleridge Wildfire 2019",
                  "007": "Hurricane Laura 2020",
                  "008": "Hurricane Sally 2020"}

    def __init__(self):
        # load data and shuffle, befor splitting
        self.df = pd.read_csv("event_duration_prediction_dataset.csv")
        self.max_lenght = self.df.apply(lambda x: len(x)).max()
        self.train_df = self.df.loc[self.df["event_type"].isin(["Hurricane Florence 2018"
								#, "Hurricane Sally 2020"
								])]
        self.valid_df = self.df.loc[self.df["event_type"].isin(["Hurricane Laura 2020" 
								#, "Saddleridge Wildfire 2019"
								])]
        self.test_df = self.df.loc[self.df["event_type"].isin(["2018 Maryland Flood" 
								#, "Lilac Wildfire 2017", "Cranston Wildfire 2018", "Holy Wildfire 2018"
								])]

        self.train = self.train_df["text"]
        self.val = self.valid_df["text"]
        self.test = self.test_df["text"]

        self.train_labels = self.train_df["regression_label"]
        self.val_labels = self.valid_df["regression_label"]
        self.test_labels = self.test_df["regression_label"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]

    def set_fold(self, set_type):
        # Make sure to call this befor using the dataset
        if set_type == DatasetType.TRAIN:
            self.dataset, self.labels = self.train, self.train_labels
        if set_type == DatasetType.TEST:
            self.dataset, self.labels = self.test, self.test_labels
        if set_type == DatasetType.VAL:
            self.dataset, self.labels = self.val, self.val_labels
        return self

# Make simple Enum for code clarity
class DatasetType(Enum):
    TRAIN = 1
    TEST = 2
    VAL = 3


#tokenized_datasets = dataset.map(tokenize_function, batched=True)

train_dataset = EventDurationPredictionDataset().set_fold(DatasetType.TRAIN)
test_dataset = EventDurationPredictionDataset().set_fold(DatasetType.TEST)
eval_dataset = EventDurationPredictionDataset().set_fold(DatasetType.VAL)


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    rmse = mean_squared_error(labels, predictions, squared=False)
    return {"rmse": rmse}

'''
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




model = TrainerTransformer(number_labels)
wandb_logger = WandbLogger(log_model="all")
dataloader = DataLoader(train_dataset)
trainer = Trainer(#logger=wandb_logger,
                  max_epochs=3)
trainer.fit(model=model, train_dataloaders=dataloader)
'''

config = RobertaConfig(
    vocab_size=800,
    problem_type = 'regression',
    type_vocab_size=1,
    ignore_mismatched_sizes=True
)

class regression_model(nn.Module):
    def __init__(self):
        super(regression_model, self).__init__()
        self.bert = RobertaForSequenceClassification.from_pretrained("roberta-base", config =config, ignore_mismatched_sizes=True).to("cuda")
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, 1)
    
    def forward(self, input_ids, attention_mask, targets, token_type_ids, text, label):
        out, _ = self.bert(input_ids, token_type_ids, attention_mask)
        out = self.dropout(out)
        loss = nn.MSELoss()
        output = loss(out, targets)
        return output

#model = RobertaForSequenceClassification.from_pretrained("roberta-base", 
#	config =config).to("cuda")

model = regression_model()

training_args = TrainingArguments(
    output_dir="model-training",
    evaluation_strategy="epoch",
    num_train_epochs=3
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
