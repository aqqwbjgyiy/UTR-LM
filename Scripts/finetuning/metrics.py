import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def performances(y_true, y_pred):
    """计算各项评估指标"""
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    pearson = pearsonr(y_true, y_pred)[0]
    spearman = spearmanr(y_true, y_pred)[0]
    
    print(f'MSE = {mse:.4f} | MAE = {mae:.4f} | R2 = {r2:.4f} | Pearson = {pearson:.4f} | Spearman = {spearman:.4f}')
    return [mse, mae, r2, pearson, spearman]