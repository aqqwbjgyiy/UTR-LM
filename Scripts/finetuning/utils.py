import os
import torch
import torch.distributed as dist

def setup_device(args):
    """设置训练设备"""
    device_ids = list(map(int, args.device_ids.split(',')))
    dist.init_process_group(backend='nccl')
    device = torch.device('cuda:{}'.format(device_ids[args.local_rank]))
    torch.cuda.set_device(device)
    return device_ids, device

def get_model_info(modelfile):
    """从模型文件名解析模型信息"""
    model_info = modelfile.split('/')[-1].split('_')
    layers, heads, embed_dim, batch_toks = 6, 16, 128, 4096
    for item in model_info:
        if 'layers' in item:
            layers = int(item[0])
        elif 'heads' in item:
            heads = int(item[:-5])
        elif 'embedsize' in item:
            embed_dim = int(item[:-9])
        elif 'batchToks' in item:
            batch_toks = int(item[:-9])
    return layers, heads, embed_dim, batch_toks

def prepare_data(args, data):
    """数据预处理"""
    train_obj_col = args.label_type
    if args.log2:
        train_obj_col = f'{args.label_type}_log2'
    return train_obj_col

def save_best_model(model, args, fold_idx, epoch, loss_train, loss_val, r2):
    """保存最佳模型"""
    save_path = os.path.join(config.model_dir, f'{args.prefix}_fold{fold_idx}_epoch{epoch}_lr{args.lr}.pt')
    torch.save(model.state_dict(), save_path)