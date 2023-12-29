import os
import csv
from sentence_transformers.evaluation import LabelAccuracyEvaluator
from torch.utils.data import DataLoader
from sentence_transformers.util import batch_to_device
import logging
import torch
from sklearn.metrics import precision_recall_fscore_support

logger = logging.getLogger(__name__)


class EventPairwiseTemporalityEvaluator(LabelAccuracyEvaluator):
    def __init__(self, dataloader: DataLoader, name: str = "", softmax_model=None, write_csv: bool = True):
        super().__init__(dataloader, name, softmax_model, write_csv)
        self.csv_file = "evaluation"+name+"_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "f1_micro", "f1_macro"]

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
            y_true.extend(label_ids)

            for idx in range(len(features)):
                features[idx] = batch_to_device(features[idx], model.device)
            label_ids = label_ids.to(model.device)
            with torch.no_grad():
                _, prediction = self.softmax_model(features, labels=None)

            total += prediction.size(0)
            correct += torch.argmax(prediction, dim=1).eq(label_ids).sum().item()
        precision_macro, recall_macro, f1_macro, support_macro = precision_recall_fscore_support(torch.argmax(prediction, dim=1), label_ids, average='macro')
        precision_micro, recall_micro, f1_micro, support_micro = precision_recall_fscore_support(torch.argmax(prediction, dim=1), label_ids, average='micro')
        accuracy = correct/total

        logger.info("Accuracy: {:.4f} ({}/{})\n".format(accuracy, correct, total))
        logger.info(f"Macro precision: {precision_macro}\n")

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow([epoch, steps, accuracy])
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, steps, accuracy])

        return accuracy