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
                 exp_name: str = "v1",
                 transformer_model: str = 'distilbert-base-uncased',
                 subset: float = 1.0):
        self.exp_name = exp_name
        self.subset = subset
        self.prepare_environment(exp_name)
        self.batch_size = batch_size
        word_embedding_model = models.Transformer(transformer_model, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                   out_features=256, activation_function=nn.Tanh())

        self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=self.device)
        self.label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}

        self.train_loss = losses.SoftmaxLoss(model=self.model,
                                             sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                             num_labels=len(self.label2int))
        training_data, validation_data = self.prepare_data(test=False)

        training_dataset = SentencesDataset(training_data, self.model)
        validation_dataset = SentencesDataset(validation_data, self.model)
        self.training_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=self.batch_size)
        self.validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=self.batch_size)
        self.validation_evaluator = EventPairwiseTemporalityEvaluator(self.validation_dataloader, name='validation_'+exp_name, softmax_model=self.train_loss)

    def prepare_data(self, test=False):
        if not test:
            train_csv_path = Path("./data/gdelt_crawled/train_v2.csv")
            valid_csv_path = Path("./data/gdelt_crawled/valid_v2.csv")
            train = StormyDataset(train_csv_path, label_pkl=Path("./data/gdelt_crawled/labels_train.pkl"),
                                  sample_indices_path=Path("./data/gdelt_crawled/sample_indices_train.json"),
                                  subset=self.subset)
            valid = StormyDataset(valid_csv_path, label_pkl=Path("./data/gdelt_crawled/labels_valid.pkl"),
                                  sample_indices_path=Path("./data/gdelt_crawled/sample_indices_valid.json"),
                                  subset=self.subset)
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
            test_csv_path = Path("./data/gdelt_crawled/test_v2.csv")
            test = StormyDataset(test_csv_path, label_pkl=None, subset=self.subset)
            test_labels = test.labels
            test_titles = test.df["title"].values
            test_examples = [InputExample(texts=[test_titles[test.sentence_pairs_indices[i][0]],
                                                 test_titles[test.sentence_pairs_indices[i][1]]],
                                          label=test_labels[i]) for i in range(len(test))]
            logger.info(f"Test: {len(test_examples)} pairs of sentences")

            test_csv_path = Path("./data/test_from_crisisfacts.csv")
            test = CrisisFactsDataset(test_csv_path, label_pkl=None, subset=self.subset)
            test_labels = test.labels
            test_titles = test.df["title"].values
            test_examples_crisisfact = [InputExample(texts=[test_titles[test.sentence_pairs_indices[i][0]],
                                                            test_titles[test.sentence_pairs_indices[i][1]]],
                                                     label=test_labels[i]) for i in range(len(test))]
            logger.info(f"Test (crisisfacts): {len(test_examples)} pairs of sentences")
            return test_examples, test_examples_crisisfact

    def train(self):
        # Configure the training
        num_epochs = 20

        warmup_steps = math.ceil(len(self.training_dataloader) * num_epochs * 0.1)  # 10% of train data for warm-up
        logger.info(f"Warmup-steps: {warmup_steps}")
        logger.info(f"Number of epochs: {num_epochs}")
        logger.info(f"Output path: {str(Path('./outputs', self.exp_name))}")

        self.validation_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name)))

        # Train the model
        self.model.fit(train_objectives=[(self.training_dataloader, self.train_loss)],
                       evaluator=self.validation_evaluator,
                       epochs=num_epochs,
                       evaluation_steps=10000,
                       warmup_steps=warmup_steps,
                       output_path=str(Path("./outputs", self.exp_name)),
                       show_progress_bar=True,
                       save_best_model=True
                      )

    def test(self):
        logger.info(f"Testing on curated test set.")
        testing_data, testing_data_crisisfacts = self.prepare_data(test=True)
        testing_dataset = SentencesDataset(testing_data, self.model)
        testing_dataloader = DataLoader(testing_dataset, shuffle=True, batch_size=self.batch_size)
        testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader, name='test_' + self.exp_name,
                                                                   softmax_model=self.train_loss)

        testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name)))


        logger.info(f"Testing on Crisisfacts test set.")
        testing_dataset = SentencesDataset(testing_data_crisisfacts, self.model)
        testing_dataloader = DataLoader(testing_dataset, shuffle=True, batch_size=self.batch_size)
        testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader, name='test_crisisfacts_' + self.exp_name,
                                                              softmax_model=self.train_loss)

        testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name)))

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
    model = EventPairwiseTemporalityModel(batch_size=512, exp_name="v3", transformer_model='distilbert-base-uncased', subset=1)
    model.train()
    model.test()
