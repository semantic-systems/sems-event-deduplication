from pathlib import Path
import math
import logging

import numpy as np
import torch.cuda
from torch import nn
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, InputExample
from torch.utils.data import DataLoader, WeightedRandomSampler
from Datasets import StormyDataset
from models.EventPairwiseTemporalityEvaluator import EventPairwiseTemporalityEvaluator
from sentence_transformers.evaluation import LabelAccuracyEvaluator


logger = logging.getLogger(__name__)


class EventPairwiseTemporalityModel(object):
    def __init__(self,
                 exp_name: str = "v1",
                 transformer_model: str = 'distilbert-base-uncased'):
        self.exp_name = exp_name
        self.prepare_environment(exp_name)
        batch_size = 512
        word_embedding_model = models.Transformer(transformer_model, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(), out_features=256, activation_function=nn.Tanh())

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=self.device)
        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}

        self.train_loss = losses.SoftmaxLoss(model=self.model,
                                             sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                             num_labels=len(self.label2int))
        training_data, validation_data, testing_data, sampler = self.prepare_data()

        training_dataset = SentencesDataset(training_data, self.model)
        validation_dataset = SentencesDataset(validation_data, self.model)
        testing_dataset = SentencesDataset(testing_data, self.model)
        self.training_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=batch_size, sampler=sampler)
        self.validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=batch_size)
        self.testing_dataloader = DataLoader(testing_dataset, shuffle=True, batch_size=batch_size)
        self.validation_evaluator = LabelAccuracyEvaluator(self.validation_dataloader, name='validation_'+exp_name, softmax_model=self.train_loss)
        self.testing_evaluator = LabelAccuracyEvaluator(self.testing_dataloader, name='test_'+exp_name, softmax_model=self.train_loss)

    def prepare_data(self):
        train_csv_path = Path("./data/gdelt_crawled/train_v1.csv")
        valid_csv_path = Path("./data/gdelt_crawled/valid_v1.csv")
        test_csv_path = Path("./data/gdelt_crawled/test_v1.csv")
        train = StormyDataset(train_csv_path, label_pkl=Path("./data/gdelt_crawled/labels_train.pkl"))
        valid = StormyDataset(valid_csv_path, label_pkl=Path("./data/gdelt_crawled/labels_valid.pkl"))
        test = StormyDataset(test_csv_path, label_pkl=Path("./data/gdelt_crawled/labels_test.pkl"))
        train_labels = train.labels
        train_titles = train.df["title"].values
        valid_labels = valid.labels
        valid_titles = valid.df["title"].values
        test_labels = test.labels
        test_titles = test.df["title"].values
        train_examples = [InputExample(texts=[train_titles[train.sentence_pairs_indices[i][0]],
                                              train_titles[train.sentence_pairs_indices[i][1]]],
                                       label=train_labels[i]) for i in range(len(train))]
        valid_examples = [InputExample(texts=[valid_titles[valid.sentence_pairs_indices[i][0]],
                                              valid_titles[valid.sentence_pairs_indices[i][1]]],
                                       label=valid_labels[i]) for i in range(len(valid))]
        test_examples = [InputExample(texts=[test_titles[test.sentence_pairs_indices[i][0]],
                                             test_titles[test.sentence_pairs_indices[i][1]]],
                                      label=test_labels[i]) for i in range(len(test))]
        logger.info(f"Train: {len(train_examples)} pairs of sentences")
        logger.info(f"Validation: {len(valid_examples)} pairs of sentences")
        logger.info(f"Test: {len(test_examples)} pairs of sentences")

        class_sample_count = np.array([len(np.where(train_labels == t)[0]) for t in np.unique(train_labels)])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in train_labels])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))

        return train_examples, valid_examples, test_examples, sampler

    def train(self):
        # Configure the training
        num_epochs = 2

        warmup_steps = math.ceil(len(self.training_dataloader) * num_epochs * 0.05)  # 5% of train data for warm-up
        logger.info(f"Warmup-steps: {warmup_steps}")
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info(f"Output path: {str(Path('./outputs', self.exp_name))}")

        # Train the model
        self.model.fit(train_objectives=[(self.training_dataloader, self.train_loss)],
                       evaluator=self.validation_evaluator,
                       epochs=num_epochs,
                       evaluation_steps=1000,
                       warmup_steps=warmup_steps,
                       output_path=str(Path("./outputs", self.exp_name)),
                       show_progress_bar=True,
                       save_best_model=True
                      )

    def test(self):
        logger.info(f"Testing...")
        self.testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name)))

    @staticmethod
    def prepare_environment(exp_name):
        if not Path("./outputs").exists():
            Path("./outputs").mkdir()
        if not Path("./outputs", exp_name).exists():
            Path("./outputs", exp_name).mkdir()

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
