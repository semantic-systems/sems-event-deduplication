from pathlib import Path
import math
import logging

import torch.cuda
from torch import nn
from sentence_transformers import SentencesDataset, losses, models, InputExample, SentenceTransformer
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
        self.task = task
        self.subset = subset
        self.prepare_environment(exp_name, task)
        self.batch_size = batch_size
        self.label2int = self.get_label2int(task)

        word_embedding_model = models.Transformer(transformer_model, max_seq_length=256)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        dense_model = models.Dense(in_features=pooling_model.get_sentence_embedding_dimension(),
                                   out_features=256, activation_function=nn.Tanh())
        if not load_pretrained:
            self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model], device=self.device)
        else:
            if Path("./outputs", exp_name, task, "pytorch_model.bin").exists():
                self.model = SentenceTransformer(str(Path("./outputs", exp_name, task).absolute()), device=self.device)
            else:
                self.model = SentenceTransformer(modules=[word_embedding_model, pooling_model, dense_model],
                                                 device=self.device)
        self.train_loss = losses.SoftmaxLoss(model=self.model,
                                             sentence_embedding_dimension=self.model.get_sentence_embedding_dimension(),
                                             num_labels=len(self.label2int))

    @staticmethod
    def get_label2int(task):
        if task == "combined":
            label2int = {"different_event": 0, "earlier": 1, "same_date": 2, "later": 3}
        elif task == "event_deduplication":
            label2int = {"different_event": 0, "same_event": 1}
        elif task == "event_temporality":
            label2int = {"earlier": 0, "same_date": 1, "later": 2}
        else:
            ValueError(
                f"{task} not defined! Please choose from 'combined', 'event_deduplication' or 'event_temporality'")
        return label2int

    def prepare_data(self, data_type="train"):
        if data_type != "test":
            train_csv_path = Path("./data/stormy_data/train_v2.csv")
            valid_csv_path = Path("./data/stormy_data/valid_v2.csv")
            train = StormyDataset(train_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            valid = StormyDataset(valid_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            train_examples = [InputExample(texts=[train.sampled_df.sentence_a[i], train.sampled_df.sentence_b[i]],
                                           label=train.sampled_df.labels[i]) for i in range(len(train))]
            valid_examples = [InputExample(texts=[valid.sampled_df.sentence_a[i], valid.sampled_df.sentence_b[i]],
                                           label=valid.sampled_df.labels[i]) for i in range(len(valid))]
            return train_examples, valid_examples, None
        else:
            test_csv_path = Path("./data/stormy_data/test_v2.csv")
            test = StormyDataset(test_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            test_examples = [InputExample(texts=[test.sampled_df.sentence_a[i], test.sampled_df.sentence_b[i]],
                                           label=test.sampled_df.labels[i]) for i in range(len(test))]
            logger.info(f"Test (stormy - test): {len(test_examples)} pairs of sentences")

            test_csv_path = Path("./data/crisisfacts_data/crisisfacts_test.csv")
            test = CrisisFactsDataset(test_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            test_examples_crisisfact = [InputExample(texts=[test.sampled_df.sentence_a[i], test.sampled_df.sentence_b[i]],
                                          label=test.sampled_df.labels[i]) for i in range(len(test))]
            logger.info(f"Test (crisisfacts - test): {len(test_examples_crisisfact)} pairs of sentences")

            test_csv_path = Path("./data/crisisfacts_data/crisisfacts_storm.csv")
            test = CrisisFactsDataset(test_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            test_examples_storm = [InputExample(texts=[test.sampled_df.sentence_a[i], test.sampled_df.sentence_b[i]],
                                          label=test.sampled_df.labels[i]) for i in range(len(test))]
            logger.info(f"Test (crisisfacts - storm): {len(test_examples)} pairs of sentences")
            return test_examples, test_examples_crisisfact, test_examples_storm

    def prepare_task_validation_data(self, data_type="train"):
        if data_type != "test":
            train_csv_path = Path("./data/crisisfacts_data/crisisfacts_train.csv")
            valid_csv_path = Path("./data/crisisfacts_data/crisisfacts_valid.csv")
            train = CrisisFactsDataset(train_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            valid = CrisisFactsDataset(valid_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            train_examples_crisisfact = [InputExample(texts=[train.sampled_df.sentence_a[i], train.sampled_df.sentence_b[i]],
                                                     label=train.sampled_df.labels[i]) for i in range(len(train))]
            valid_examples_crisisfact = [InputExample(texts=[train.sampled_df.sentence_a[i], train.sampled_df.sentence_b[i]],
                                                      label=train.sampled_df.labels[i]) for i in range(len(valid))]
            return train_examples_crisisfact, valid_examples_crisisfact, None
        else:
            test_csv_path = Path("./data/stormy_data/test_v2.csv")
            test = StormyDataset(test_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            test_examples = [InputExample(texts=[test.sampled_df.sentence_a[i], test.sampled_df.sentence_b[i]],
                                          label=test.sampled_df.labels[i]) for i in range(len(test))]
            logger.info(f"Test (stormy - test): {len(test_examples)} pairs of sentences")

            test_csv_path = Path("./data/crisisfacts_data/crisisfacts_test.csv")
            test = CrisisFactsDataset(test_csv_path, task=self.task, data_type=data_type, subset=self.subset)
            test_examples_crisisfact = [InputExample(texts=[test.sampled_df.sentence_a[i], test.sampled_df.sentence_b[i]],
                                                     label=test.sampled_df.labels[i]) for i in range(len(test))]
            logger.info(f"Test (crisisfacts - test): {len(test_examples_crisisfact)} pairs of sentences")
            return test_examples, test_examples_crisisfact, None

    def train(self, task_validation: False):
        logger.info(f"Preparing training and validation dataset.")
        if task_validation:
            training_data, validation_data, _ = self.prepare_task_validation_data(data_type="train")
        else:
            training_data, validation_data, _ = self.prepare_data(data_type="train")

        training_dataset = SentencesDataset(training_data, self.model)
        validation_dataset = SentencesDataset(validation_data, self.model)
        training_dataloader = DataLoader(training_dataset, shuffle=True, batch_size=self.batch_size)
        validation_dataloader = DataLoader(validation_dataset, shuffle=True, batch_size=self.batch_size)
        validation_evaluator = EventPairwiseTemporalityEvaluator(validation_dataloader,
                                                                 name=f'validation_{self.exp_name}_{self.task}',
                                                                 softmax_model=self.train_loss)

        warmup_steps = math.ceil(len(training_dataloader) * self.num_epochs * 0.1)  # 10% of train data for warm-up
        logger.info(f"Warmup-steps: {warmup_steps}")
        logger.info(f"Number of epochs: {self.num_epochs}")
        logger.info(f"Output path: {str(Path('./outputs', self.exp_name, self.task))}")

        validation_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name, self.task, "validation")))

        # Train the model
        self.model.fit(train_objectives=[(training_dataloader, self.train_loss)],
                       evaluator=validation_evaluator,
                       epochs=self.num_epochs,
                       evaluation_steps=2000,
                       warmup_steps=warmup_steps,
                       output_path=str(Path("./outputs", self.exp_name, self.task)),
                       show_progress_bar=True,
                       save_best_model=True
                  )

    def test(self, task_validation: False):
        logger.info(f"Testing on stormy test set...")
        if task_validation:
            testing_data, testing_data_crisisfacts_test, test_examples_crisisfact_test_storm = self.prepare_task_validation_data(data_type="test")
        else:
            testing_data, testing_data_crisisfacts_test, test_examples_crisisfact_test_storm = self.prepare_data(data_type="test")
        testing_dataset = SentencesDataset(testing_data, self.model)
        testing_dataloader = DataLoader(testing_dataset, shuffle=False, batch_size=self.batch_size)
        testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader,
                                                              name=f'test_{self.exp_name}_{self.task}',
                                                              softmax_model=self.train_loss)

        testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name, self.task, "test")))

        logger.info(f"Testing on Crisisfacts test set...")
        testing_dataset = SentencesDataset(testing_data_crisisfacts_test, self.model)
        testing_dataloader = DataLoader(testing_dataset, shuffle=False, batch_size=self.batch_size)
        testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader,
                                                              name=f'test_{self.exp_name}_{self.task}',
                                                              softmax_model=self.train_loss)

        testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name, self.task, "test")))

        if test_examples_crisisfact_test_storm is not None:
            logger.info(f"Testing on Crisisfacts storm set...")
            testing_dataset = SentencesDataset(test_examples_crisisfact_test_storm, self.model)
            testing_dataloader = DataLoader(testing_dataset, shuffle=False, batch_size=self.batch_size)
            testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader,
                                                                  name=f'test_{self.exp_name}_{self.task}_storm',
                                                                  softmax_model=self.train_loss)

            testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name, self.task, "test")))

    @staticmethod
    def prepare_environment(exp_name, task):
        if not Path("./outputs").exists():
            Path("./outputs").mkdir()
        if not Path("./outputs", exp_name).exists():
            Path("./outputs", exp_name).mkdir()
        if not Path("./outputs", exp_name, task).exists():
            Path("./outputs", exp_name, task).mkdir()
        if not Path("./outputs", exp_name, task, "train").exists():
            Path("./outputs", exp_name, task, "train").mkdir()
        if not Path("./outputs", exp_name, task, "validation").exists():
            Path("./outputs", exp_name, task, "validation").mkdir()
        if not Path("./outputs", exp_name, task, "test").exists():
            Path("./outputs", exp_name, task, "test").mkdir()

    @property
    def device(self):
        if torch.cuda.is_available():
            return 'cuda'
        else:
            return "cpu"


if __name__ == "__main__":
    model = EventPairwiseTemporalityModel(batch_size=512,
                                          num_epochs=2,
                                          exp_name="v6",
                                          transformer_model='distilbert-base-uncased',
                                          subset=0.01,
                                          load_pretrained=False,
                                          task="event_deduplication")

    model.train(task_validation=False)
    model.test(task_validation=False)

    model = EventPairwiseTemporalityModel(batch_size=256,
                                          num_epochs=2,
                                          exp_name="v6",
                                          transformer_model='distilbert-base-uncased',
                                          subset=0.0001,
                                          load_pretrained=False,
                                          task="event_deduplication")
    model.train(task_validation=True)
    model.test(task_validation=True)

    model = EventPairwiseTemporalityModel(batch_size=512,
                                          num_epochs=2,
                                          exp_name="v6",
                                          transformer_model='distilbert-base-uncased',
                                          subset=0.01,
                                          load_pretrained=False,
                                          task="event_temporality")
    model.train(task_validation=False)
    model.test(task_validation=False)

    model = EventPairwiseTemporalityModel(batch_size=256,
                                          num_epochs=2,
                                          exp_name="v6",
                                          transformer_model='distilbert-base-uncased',
                                          subset=0.0001,
                                          load_pretrained=False,
                                          task="event_temporality")
    model.train(task_validation=True)
    model.test(task_validation=True)
