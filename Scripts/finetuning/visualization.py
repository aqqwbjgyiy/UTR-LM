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
    plt.savefig(f'/scratch/users/yanyichu/UTR-LM/Sample/figures/{args.prefix}_fold{fold_idx}.tif')
    plt.close()

def generate_and_save_results(args, model_best, ep_best, fold_idx):
    """生成并保存训练结果"""
    # 生成预测结果
    train_pred = generate_predictions(model_best, train_loader)
    val_pred = generate_predictions(model_best, val_loader)
    test_pred = generate_predictions(model_best, test_loader)
    
    # 保存预测结果
    save_predictions(args, fold_idx, train_pred, val_pred, test_pred)
    
    # 绘制结果图表
    plot_results(args, model_best, ep_best, fold_idx, train_pred, val_pred, test_pred)