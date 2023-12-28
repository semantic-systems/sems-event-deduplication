from pathlib import Path
import math
import logging

import torch.cuda
from torch import nn
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, InputExample
from torch.utils.data import DataLoader
import pickle
from models.Datasets import StormyDataset
from sklearn.model_selection import train_test_split
from sentence_transformers.evaluation import LabelAccuracyEvaluator


class EventPairwiseTemporalityModel(object):
    def __init__(self,
                 csv_path: Path = Path("../data/gdelt_crawled/final_df_v1.csv"),
                 label_pkl: Path = Path("../data/gdelt_crawled/labels.pkl"),
                 transformer_model: str = 'distilbert-base-uncased'):
        self.prepare_environment()
        word_embedding_model = models.Transformer(transformer_model, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=self.device)
        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}

        self.train_loss = losses.SoftmaxLoss(model=self.model,
                                             sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                             num_labels=len(self.label2int))
        training_data, validation_data, testing_data = self.prepare_data(csv_path, label_pkl=label_pkl)

        training_dataset = SentencesDataset(training_data, self.model)
        validation_dataset = SentencesDataset(validation_data, self.model)
        testing_dataset = SentencesDataset(testing_data, self.model)
        self.training_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=32)
        self.validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=32)
        self.testing_dataloader = DataLoader(testing_dataset, shuffle=True, batch_size=32)
        self.validation_evaluator = LabelAccuracyEvaluator(self.validation_dataloader, name='validation', softmax_model=self.train_loss)
        self.testing_evaluator = LabelAccuracyEvaluator(self.testing_dataloader, name='test', softmax_model=self.train_loss)

    def prepare_data(self, csv_path, label_pkl=None):
        label_pkl = label_pkl if label_pkl is None else Path(label_pkl)
        data = StormyDataset(Path(csv_path), label_pkl=label_pkl)
        labels = data.labels
        titles = data.df["title"].values
        train_examples = [InputExample(texts=[titles[data.sentence_pairs_indices[i][0]], titles[data.sentence_pairs_indices[i][1]]],
                                       label=labels[i]) for i in range(len(data))]
        training_data, testing_data = train_test_split(train_examples, test_size=0.30, random_state=42)
        validation_data, testing_data = train_test_split(testing_data, test_size=0.50, random_state=42)
        return training_data, validation_data, testing_data

    def train(self):
        # Configure the training
        num_epochs = 3

        warmup_steps = math.ceil(len(self.training_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logging.info("Warmup-steps: {}".format(warmup_steps))

        # Train the model
        self.model.fit(train_objectives=[(self.training_dataloader, self.train_loss)],
                       evaluator=self.validation_evaluator,
                       epochs=num_epochs,
                       evaluation_steps=100,
                       warmup_steps=warmup_steps,
                       output_path="./outputs",
                       show_progress_bar=True
                      )

    def test(self):
        self.testing_evaluator(self.model, output_path="./outputs")

    @staticmethod
    def prepare_environment():
        if not Path("./outputs").exists():
            Path("./outputs").mkdir()

    @property
    def device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return "cpu"



if __name__ == "__main__":
    model = EventPairwiseTemporalityModel()
    model.train()
    model.test()
