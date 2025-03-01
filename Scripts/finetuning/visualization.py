import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_results(args, model_best, ep_best, fold_idx, train_pred, val_pred, test_pred):
    """绘制训练结果图表"""
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # 绘制训练、验证、测试数据的分布图
    plot_distribution(axes[0], train_pred, val_pred, test_pred, args)
    
    # 绘制预测值与真实值的对比图
    plot_predictions(axes[1], train_pred, val_pred, test_pred, args)
    
    # 保存图表
    plt.savefig(os.path.join(config.figure_dir, f'{args.prefix}_fold{fold_idx}.tif'))
    plt.close()

def save_predictions(args, fold_idx, train_pred, val_pred, test_pred):
    """保存预测结果到CSV文件"""
    # 保存训练集预测结果
    if train_pred is not None:
        train_pred.to_csv(os.path.join(config.output_dir, 
                         f'{args.prefix}_fold{fold_idx}_train_predictions.csv'), 
                         index=False)
    
    # 保存验证集预测结果
    if val_pred is not None:
        val_pred.to_csv(os.path.join(config.output_dir, 
                       f'{args.prefix}_fold{fold_idx}_val_predictions.csv'), 
                       index=False)
    
    # 保存测试集预测结果
    if test_pred is not None:
        test_pred.to_csv(os.path.join(config.output_dir, 
                        f'{args.prefix}_fold{fold_idx}_test_predictions.csv'), 
                        index=False)

def generate_and_save_results(args, model_best, ep_best, fold_idx, train_loader=None, val_loader=None):
    """生成并保存训练结果"""
    # 生成预测结果
    if train_loader is not None:
        train_pred = generate_predictions(model_best, train_loader)
    else:
        train_pred = None
        
    if val_loader is not None:
        val_pred = generate_predictions(model_best, val_loader)
    else:
        val_pred = None
        
    if hasattr(args, 'test_loader') and args.test_loader is not None:
        test_pred = generate_predictions(model_best, args.test_loader)
    else:
        test_pred = None
    
    # 保存预测结果
    save_predictions(args, fold_idx, train_pred, val_pred, test_pred)
    
    # 绘制结果图表
    plot_results(args, model_best, ep_best, fold_idx, train_pred, val_pred, test_pred)