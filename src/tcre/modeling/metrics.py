from ignite.metrics import MetricsLambda, Metric
import pandas as pd


def F1(r, p):
    return 2 * (p * r) / (p + r + 1e-20)


def get_f1_metric(precision, recall):
    return MetricsLambda(F1, recall, precision)


class PredictionAggregator(Metric):
    """Aggregate predictions across batches (useful for not repeating prediction steps)"""

    def __init__(self, output_transform=lambda x: {'y_pred': x[0], 'y': x[1], 'id': x[2]}):
        super().__init__(lambda x: x)
        self.predictions = []
        self.output_transform = output_transform

    def reset(self):
        self.predictions = []

    def update(self, output):
        preds = {k: v.cpu().numpy() for k, v in self.output_transform(output).items()}
        self.predictions.append(preds)

    def compute(self):
        return pd.concat([pd.DataFrame(p) for p in self.predictions])
