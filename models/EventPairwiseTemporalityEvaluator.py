import os
import csv

import numpy as np
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.util import batch_to_device
import logging
import torch
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)


class EventPairwiseTemporalityEvaluator(LabelAccuracyEvaluator):
    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, write_csv: bool = True):
        super().__init__(dataloader, name, softmax_model, write_csv)
        self.csv_file = "evaluation_"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_support",
                                     "micro_precision", "micro_recall",  "micro_f1", "micro_support",
                                     "weighted_precision", "weighted_recall", "weighted_f1", "weighted_support"]

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        model.eval()
        total = 0
        correct = 0
        y_true = []
        y_predict = []

        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("Evaluation on the "+self.name+" dataset"+out_txt)
        self.dataloader.collate_fn = model.smart_batching_collate

        for step, batch in enumerate(self.dataloader):
            features, label_ids = batch
            y_true.append(label_ids)
            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)
            y_predict.append(torch.argmax(prediction, dim=1))
            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        y_true = np.concatenate(y_true)
        y_predict = np.concatenate(y_predict)
        macro_precision, macro_recall, macro_f1, macro_support = precision_recall_fscore_support(y_true, y_predict, average='macro')
        micro_precision, micro_recall, micro_f1, micro_support = precision_recall_fscore_support(y_true, y_predict, average='micro')
        weighted_precision, weighted_recall, weighted_f1, weighted_support = precision_recall_fscore_support(y_true, y_predict, average='weighted')
        accuracy = correct/total

        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        logger.info(f"Macro metrics:")
        logger.info(f"    precision: {macro_precision}")
        logger.info(f"    recall: {macro_recall}")
        logger.info(f"    f1: {macro_f1}")
        logger.info(f"    support: {macro_support}")
        logger.info(f"Micro metrics:")
        logger.info(f"    precision: {micro_precision}")
        logger.info(f"    recall: {micro_recall}")
        logger.info(f"    f1: {micro_f1}")
        logger.info(f"    support: {micro_support}")
        logger.info(f"Weighted metrics:")
        logger.info(f"    precision: {weighted_precision}")
        logger.info(f"    recall: {weighted_recall}")
        logger.info(f"    f1: {weighted_f1}")
        logger.info(f"    support: {weighted_support}")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if "test" in self.name:
                y_predict.dump(os.path.join(output_path, "prediction.pkl"))

            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy, macro_precision, macro_recall, macro_f1, macro_support,
                                     micro_precision, micro_recall,  micro_f1, micro_support,
                                     weighted_precision, weighted_recall, weighted_f1, weighted_support])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy, macro_precision, macro_recall, macro_f1, macro_support,
                                     micro_precision, micro_recall,  micro_f1, micro_support,
                                     weighted_precision, weighted_recall, weighted_f1, weighted_support])

        return accuracy
