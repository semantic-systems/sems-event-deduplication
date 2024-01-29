from pathlib import Path
import math
import logging

import torch.cuda
from torch import nn
from sentence_transformers import SentencesDataset, losses, models, InputExample, SentenceTransformer
from torch.utils.data import DataLoader
from Datasets import StormyDataset, CrisisFactsDataset
from EventPairwiseTemporalityEvaluator import EventPairwiseTemporalityEvaluator
from torch.optim import AdamW

logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


class EventPairwiseTemporalityModel(object):
    def __init__(self,
                 multipliers: list,
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
        self.multipliers = multipliers
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
            train = StormyDataset(train_csv_path, task=self.task, data_type=data_type, forced=False, multiplier=self.multipliers[0]) #50
            valid = StormyDataset(valid_csv_path, task=self.task, data_type="valid", forced=False, multiplier=self.multipliers[1]) #30
            train_examples = [InputExample(texts=[train.sampled_df.sentence_a.values[i], train.sampled_df.sentence_b.values[i]],
                                           label=train.labels[i]) for i in range(len(train))]
            valid_examples = [InputExample(texts=[valid.sampled_df.sentence_a.values[i], valid.sampled_df.sentence_b.values[i]],
                                           label=valid.labels[i]) for i in range(len(valid))]
            logger.info(f"Test (Disc - train): {len(train_examples)} pairs of sentences")
            logger.info(f"Test (Disc - valid): {len(valid_examples)} pairs of sentences\n\n")
            return train_examples, valid_examples
        else:
            test_csv_path = Path("./data/stormy_data/test_v2.csv")
            test = StormyDataset(test_csv_path, task=self.task, data_type=data_type, forced=False, multiplier=self.multipliers[2]) #30
            test_examples = [InputExample(texts=[test.sampled_df.sentence_a.values[i], test.sampled_df.sentence_b.values[i]],
                                           label=test.labels[i]) for i in range(len(test))]
            logger.info(f"Test (Disc - test): {len(test_examples)} pairs of sentences")

            test_csv_path = Path("./data/crisisfacts_data/crisisfacts_test.csv")
            test = CrisisFactsDataset(test_csv_path, task=self.task, data_type=data_type, forced=False, multiplier=self.multipliers[3])  #30
            test_examples_crisisfact = [InputExample(texts=[test.sampled_df.sentence_a.values[i], test.sampled_df.sentence_b.values[i]],
                                          label=test.labels[i]) for i in range(len(test))]
            logger.info(f"Test (Crisisfacts - test): {len(test_examples_crisisfact)} pairs of sentences")

            return test_examples, test_examples_crisisfact

    def prepare_task_validation_data(self, data_type="train"):
        if data_type != "test":
            train_csv_path = Path("./data/crisisfacts_data/crisisfacts_train.csv")
            valid_csv_path = Path("./data/crisisfacts_data/crisisfacts_valid.csv")
            train = CrisisFactsDataset(train_csv_path, task=self.task, data_type=data_type, forced=False, multiplier=self.multipliers[0]) #50
            valid = CrisisFactsDataset(valid_csv_path, task=self.task, data_type="valid", forced=False, multiplier=self.multipliers[1]) #30
            train_examples_crisisfact = [InputExample(texts=[train.sampled_df.sentence_a.values[i], train.sampled_df.sentence_b.values[i]],
                                                     label=train.labels[i]) for i in range(len(train))]
            valid_examples_crisisfact = [InputExample(texts=[train.sampled_df.sentence_a.values[i], train.sampled_df.sentence_b.values[i]],
                                                      label=train.labels[i]) for i in range(len(valid))]
            logger.info(f"Test (Crisisfacts - train): {len(train_examples_crisisfact)} pairs of sentences")
            logger.info(f"Test (Crisisfacts - valid): {len(valid_examples_crisisfact)} pairs of sentences\n\n")

            return train_examples_crisisfact, valid_examples_crisisfact
        else:
            test_csv_path = Path("./data/stormy_data/test_v2.csv")
            test = StormyDataset(test_csv_path, task=self.task, data_type=data_type, forced=False, multiplier=self.multipliers[2]) #30
            test_examples = [InputExample(texts=[test.sampled_df.sentence_a.values[i], test.sampled_df.sentence_b.values[i]],
                                          label=test.labels[i]) for i in range(len(test))]
            logger.info(f"Test (Disc - test): {len(test_examples)} pairs of sentences")

            test_csv_path = Path("./data/crisisfacts_data/crisisfacts_test.csv")
            test = CrisisFactsDataset(test_csv_path, task=self.task, data_type=data_type, forced=False, multiplier=self.multipliers[3]) #30
            test_examples_crisisfact = [InputExample(texts=[test.sampled_df.sentence_a.values[i], test.sampled_df.sentence_b.values[i]],
                                                     label=test.labels[i]) for i in range(len(test))]
            logger.info(f"Test (Crisisfacts - test): {len(test_examples_crisisfact)} pairs of sentences\n\n")
            return test_examples, test_examples_crisisfact

    def train(self, task_validation: False):
        logger.info(f"Preparing training and validation dataset.")
        if task_validation:
            training_data, validation_data = self.prepare_task_validation_data(data_type="train")
        else:
            training_data, validation_data = self.prepare_data(data_type="train")

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
                       evaluation_steps=10000,
                       warmup_steps=warmup_steps,
                       output_path=str(Path("./outputs", self.exp_name, self.task)),
                       show_progress_bar=True,
                       save_best_model=True,
                       optimizer_params={'lr': 2e-05})

    def test(self, task_validation: False):
        logger.info(f"Testing on Disc test set...")
        if task_validation:
            testing_data, testing_data_crisisfacts_test = self.prepare_task_validation_data(data_type="test")
        else:
            testing_data, testing_data_crisisfacts_test = self.prepare_data(data_type="test")
        testing_dataset = SentencesDataset(testing_data, self.model)
        testing_dataloader = DataLoader(testing_dataset, shuffle=False, batch_size=self.batch_size)
        testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader,
                                                              name=f'test_{self.exp_name}_{self.task}_gdelt',
                                                              softmax_model=self.train_loss)

        testing_evaluator(self.model, output_path=str(Path("./outputs", self.exp_name, self.task, "test")))

        logger.info(f"Testing on Crisisfacts test set...")
        testing_dataset = SentencesDataset(testing_data_crisisfacts_test, self.model)
        testing_dataloader = DataLoader(testing_dataset, shuffle=False, batch_size=self.batch_size)
        testing_evaluator = EventPairwiseTemporalityEvaluator(testing_dataloader,
                                                              name=f'test_{self.exp_name}_{self.task}_crisisfacts',
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
    logger.info("\n\nEvent Narrated Time Prediction Crisisfacts\n\n")
    model = EventPairwiseTemporalityModel(multipliers=[50, 30, 30],
                                          batch_size=128,
                                          num_epochs=2,
                                          exp_name="v4",
                                          transformer_model='roberta-base',
                                          load_pretrained=False,
                                          task="event_temporality")
    model.train(task_validation=True)
    model.test(task_validation=True)

    logger.info("\n\nEvent Narrated Time Prediction Disc\n\n")
    model = EventPairwiseTemporalityModel(multipliers=[33, 10, 16],
                                          batch_size=128,
                                          num_epochs=2,
                                          exp_name="v4",
                                          transformer_model='roberta-base',
                                          load_pretrained=False,
                                          task="event_temporality")
    model.train(task_validation=False)
    model.test(task_validation=False)

    logger.info("\n\nEvent Deduplication Crisisfacts\n\n")
    model = EventPairwiseTemporalityModel(multipliers=[33, 10, 16],
                                          batch_size=128,
                                          num_epochs=2,
                                          exp_name="v4",
                                          transformer_model='roberta-base',
                                          load_pretrained=False,
                                          task="event_deduplication")
    model.train(task_validation=True)
    model.test(task_validation=True)


    logger.info("\n\nEvent Deduplication Disc\n\n")
    model = EventPairwiseTemporalityModel(multipliers=[50, 30, 30],
                                          batch_size=128,
                                          num_epochs=2,
                                          exp_name="v4",
                                          transformer_model='roberta-base',
                                          load_pretrained=False,
                                          task="event_deduplication")

    model.train(task_validation=False)
    model.test(task_validation=False)
