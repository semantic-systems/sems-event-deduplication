from pathlib import Path
import math
import logging

import torch.cuda
from torch import nn
from sentence_transformers import SentenceTransformer, SentencesDataset, losses, models, InputExample
from torch.utils.data import DataLoader
from Datasets import StormyDataset, CrisisFactsDataset
from EventPairwiseTemporalityEvaluator import EventPairwiseTemporalityEvaluator

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


class EventPairwiseTemporalityModel(object):
    def __init__(self,
                 batch_size: int = 512,
                 num_epochs: int = 10,
                 exp_name: str = "v1",
                 transformer_model: str = 'distilbert-base-uncased',
                 subset: float = 1.0,
                 load_pretrained: bool = False,
                 task: str = "combined"):
        self.exp_name = exp_name
        self.num_epochs = num_epochs
        self.task = task  # task in {"combined", "event_deduplication", "event_temporality"}
        self.subset = subset
        self.prepare_environment(exp_name, task)
        self.batch_size = batch_size
        word_embedding_model = models.Transformer(transformer_model, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                   out_features=256, activation_function=nn.Tanh())
        if not load_pretrained:
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=self.device)
        else:
            self.model = SentenceTransformer(str(Path("./outputs", exp_name, task).absolute()))

        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}

        self.train_loss = losses.SoftmaxLoss(model=self.model,
                                             sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                             num_labels=len(self.label2int))
        training_data, validation_data = self.prepare_data(test=False)

        training_dataset = SentencesDataset(training_data, self.model)
        validation_dataset = SentencesDataset(validation_data, self.model)
        self.training_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=self.batch_size)
        self.validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=self.batch_size)
        self.validation_evaluator = EventPairwiseTemporalityEvaluator(self.validation_dataloader, name=f'validation_{exp_name}_{task}', softmax_model=self.train_loss)

    def prepare_data(self, test=False):
        if not test:
            train_csv_path = Path("./data/stormy_data/train_v2.csv")
            valid_csv_path = Path("./data/stormy_data/valid_v2.csv")
            train = StormyDataset(train_csv_path, label_pkl=Path(f"./data/stormy_data/{self.task}/labels_train.pkl"),
                                  sample_indices_path=Path(f"./data/stormy_data/{self.task}/sample_indices_train.json"),
                                  subset=self.subset, task=self.task)
            valid = StormyDataset(valid_csv_path, label_pkl=Path(f"./data/stormy_data/{self.task}/labels_valid.pkl"),
                                  sample_indices_path=Path(f"./data/stormy_data/{self.task}/sample_indices_valid.json"),
                                  subset=self.subset, task=self.task)
            train_labels = train.labels
            train_titles = train.df["title"].values
            valid_labels = valid.labels
            valid_titles = valid.df["title"].values
            train_examples = [InputExample(texts=[train_titles[train.sentence_pairs_indices[i][0]],
                                                  train_titles[train.sentence_pairs_indices[i][1]]],
                                           label=train_labels[i]) for i in range(len(train))]
            valid_examples = [InputExample(texts=[valid_titles[valid.sentence_pairs_indices[i][0]],
                                                  valid_titles[valid.sentence_pairs_indices[i][1]]],
                                           label=valid_labels[i]) for i in range(len(valid))]
            logger.info(f"Train: {len(train_examples)} pairs of sentences")
            logger.info(f"Validation: {len(valid_examples)} pairs of sentences")

            return train_examples, valid_examples
        else:
            test_csv_path = Path(f"./data/stormy_data/test_v2.csv")
            test = StormyDataset(test_csv_path,
                                 label_pkl=Path(f"./data/stormy_data/{self.task}/labels_test.pkl"),
                                 sample_indices_path=Path(f"./data/stormy_data/{self.task}/sample_indices_test.json"),
                                 subset=self.subset, task=self.task)
            test_labels = test.labels
            test_titles = test.df["title"].values
            test_examples = [InputExample(texts=[test_titles[test.sentence_pairs_indices[i][0]],
                                                 test_titles[test.sentence_pairs_indices[i][1]]],
                                          label=test_labels[i]) for i in range(len(test))]
            logger.info(f"Test: {len(test_examples)} pairs of sentences")

            test_csv_path = Path("./data/crisisfacts_data/test_from_crisisfacts.csv")
            test = CrisisFactsDataset(test_csv_path,
                                      label_pkl=Path(f"./data/crisisfacts_data/{self.task}/labels_crisisfacts_test.pkl"),
                                      sample_indices_path=Path(f"./data/crisisfacts_data/{self.task}/sample_indices_test_crisisfacts.json"),
                                      subset=self.subset, task=self.task)
            test_labels = test.labels
            test_titles = test.df["title"].values
            test_examples_crisisfact = [InputExample(texts=[test_titles[test.sentence_pairs_indices[i][0]],
                                                            test_titles[test.sentence_pairs_indices[i][1]]],
                                                     label=test_labels[i]) for i in range(len(test))]
            logger.info(f"Test (crisisfacts): {len(test_examples_crisisfact)} pairs of sentences")
            return test_examples, test_examples_crisisfact

    def train(self):

        warmup_steps = math.ceil(len(self.training_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up
        logger.info(f"Warmup-steps: {warmup_steps}")
        logger.info(f"Number of epochs: {self.num_epochs}")
        logger.info(f"Output path: {str(Path('./outputs', self.exp_name, self.task))}")

        self.validation_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name, self.task)))

        # Train the model
        self.model.fit(train_objectives=[(self.training_dataloader, self.train_loss)],
                       evaluator=self.validation_evaluator,
                       epochs=self.num_epochs,
                       evaluation_steps=10000,
                       warmup_steps=warmup_steps,
                       output_path=str(Path("./outputs", self.exp_name, self.task)),
                       show_progress_bar=True,
                       save_best_model=True
                      )

    def test(self):
        logger.info(f"Testing on curated test set.")
        testing_data, testing_data_crisisfacts = self.prepare_data(test=True)
        testing_dataset = SentencesDataset(testing_data, self.model)
        testing_dataloader = DataLoader(testing_dataset, shuffle=True, batch_size=self.batch_size)
        testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader,
                                                              name=f'test_{self.exp_name}_{self.task}',
                                                              softmax_model=self.train_loss)

        testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name, self.task)))


        logger.info(f"Testing on Crisisfacts test set.")
        testing_dataset = SentencesDataset(testing_data_crisisfacts, self.model)
        testing_dataloader = DataLoader(testing_dataset, shuffle=True, batch_size=self.batch_size)
        testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader,
                                                              name=f'test_crisisfacts_{self.exp_name}_{self.task}',
                                                              softmax_model=self.train_loss)

        testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name, self.task)))

    @staticmethod
    def prepare_environment(exp_name, task):
        if not Path("./outputs").exists():
            Path("./outputs").mkdir()
        if not Path("./outputs", exp_name).exists():
            Path("./outputs", exp_name).mkdir()
        if not Path("./outputs", exp_name, task).exists():
            Path("./outputs", exp_name, task).mkdir()
    @property
    def device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return "cpu"


if __name__ == "__main__":
    model = EventPairwiseTemporalityModel(batch_size=512,
                                          num_epochs=2,
                                          exp_name="test",
                                          transformer_model='distilbert-base-uncased',
                                          subset=0.01,
                                          load_pretrained=True,
                                          task="combined")
    model.train()
    model.test()
