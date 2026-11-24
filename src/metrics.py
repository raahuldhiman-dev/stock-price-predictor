import numpy as np


def rmse(actual, predicted):
    a, p = np.ravel(actual), np.ravel(predicted)
    return float(np.sqrt(np.mean((a - p) ** 2)))


def mae(actual, predicted):
    a, p = np.ravel(actual), np.ravel(predicted)
    return float(np.mean(np.abs(a - p)))


def mape(actual, predicted):
    a, p = np.ravel(actual), np.ravel(predicted)
    mask = a != 0
    return float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100)


def all_metrics(actual, predicted):
    return {"RMSE": rmse(actual, predicted), "MAE": mae(actual, predicted), "MAPE %": mape(actual, predicted)}
