from ignite.metrics import MetricsLambda


def F1(r, p):
    return 2 * (p * r) / (p + r + 1e-20)


def get_f1_metric(precision, recall):
    return MetricsLambda(F1, recall, precision)