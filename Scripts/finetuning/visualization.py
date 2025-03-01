import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import config

def generate_predictions(model, dataloader):
    """生成模型预测结果"""
    model.eval()
    y_pred_list = []
    y_true_list = []
    strs_list = []
    
    with torch.no_grad():
        for batch in dataloader:
            # Check if batch is a list/tuple of batches or a single batch
            if isinstance(batch, (list, tuple)) and len(batch) == 1:
                batch = batch[0]  # Extract the actual batch
                
            # Handle different batch formats
            if isinstance(batch, (list, tuple)):
                if len(batch) == 6:
                    labels, strs, masked_strs, toks, masked_toks, _ = batch
                else:
                    # If we have a different number of elements, try to extract what we need
                    try:
                        # Assuming the first element is labels and we can find tokens
                        labels = batch[0]
                        strs = batch[1] if len(batch) > 1 else [""] * len(labels)
                        toks = None
                        for item in batch:
                            if isinstance(item, torch.Tensor) and item.dim() > 1:
                                toks = item
                                break
                        if toks is None:
                            print("Warning: Could not find token tensor in batch")
                            continue
                    except Exception as e:
                        print(f"Error processing batch: {e}")
                        continue
            else:
                # If batch is not a tuple/list, it might be a dictionary or another structure
                print(f"Unexpected batch type: {type(batch)}")
                continue
                
            strs_list.extend(strs)
            toks = toks.to(next(model.parameters()).device)
            labels = torch.FloatTensor(labels).reshape(-1, 1).to(next(model.parameters()).device)
            
            # Create a mock args object with layers attribute if needed
            class MockArgs:
                def __init__(self):
                    self.layers = 6  # Default value, should match your model's configuration
            
            # Get the actual args from the model if possible
            if hasattr(model, 'args'):
                args = model.args
            else:
                args = MockArgs()
            
            # Get the layers parameter
            if hasattr(args, 'layers') and args.layers is not None:
                layers = args.layers
            else:
                # Default to layer 6 if not specified
                layers = 6
            
            # Call the model with the required arguments
            outputs = model(toks, args=args, layers=layers, return_representation=True, return_contacts=True)
            
            y_pred_list.extend(outputs.reshape(-1).cpu().detach().tolist())
            y_true_list.extend(labels.cpu().reshape(-1).tolist())
    
    # 创建结果DataFrame
    results_df = pd.DataFrame({
        'sequence': strs_list,
        'true_value': y_true_list,
        'predicted': y_pred_list
    })
    
    return results_df

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