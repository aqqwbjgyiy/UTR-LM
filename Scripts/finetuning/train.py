import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
import random
import numpy as np
from sklearn.model_selection import KFold
from copy import deepcopy
from tqdm import trange
from esm import Alphabet
from config import config
import os

from models import CNN_linear
from data_utils import load_data, generate_dataset_dataloader, generate_trainbatch_loader
from utils import setup_device, get_model_info, prepare_data  # 添加 prepare_data 导入
from metrics import performances
from visualization import plot_results
import logging

def setup_logging(args):
    """设置日志记录"""
    log_dir = os.path.join(config.output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
        filename=os.path.join(log_dir, f'{args.prefix}.log')
    )

def setup_seed(seed):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def train_step(data, batches_loader, model, device, criterion, optimizer, args, epoch, train_obj_col):
    """训练步骤"""
    try:
        model.train()
        y_pred_list, y_true_list, loss_list = [], [], []
        
        for i, batch in enumerate(batches_loader):
            try:
                batch = np.array(torch.LongTensor(batch))
                e_data = data.iloc[batch]
                
                dataset = FastaBatchedDataset(e_data.loc[:, train_obj_col], e_data.utr, mask_prob=0.0)
                dataloader = torch.utils.data.DataLoader(dataset,
                                                     collate_fn=alphabet.get_batch_converter(),
                                                     batch_size=len(batch),
                                                     shuffle=False)
                
                for (labels, strs, masked_strs, toks, masked_toks, _) in dataloader:
                    try:
                        toks = toks.to(device)
                        labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
                        
                        outputs = model(toks, return_representation=True, return_contacts=True)
                        loss = criterion(outputs, labels)
                        
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        loss_list.append(loss.cpu().detach())
                        y_true_list.extend(labels.cpu().reshape(-1).tolist())
                        y_pred = outputs.reshape(-1).cpu().detach().tolist()
                        y_pred_list.extend(y_pred)
                        
                    except RuntimeError as e:
                        logging.error(f"CUDA error in training batch: {str(e)}")
                        continue
                        
            except Exception as e:
                logging.error(f"Error processing batch {i}: {str(e)}")
                continue
                
        loss_epoch = float(torch.Tensor(loss_list).mean()) if loss_list else float('inf')
        logging.info(f'Train: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f}')
        
        metrics = performances(y_true_list, y_pred_list)
        return metrics, loss_epoch
        
    except Exception as e:
        logging.error(f"Training step failed at epoch {epoch}: {str(e)}")
        raise e

def eval_step(dataloader, model, device, criterion, args, epoch, data=None):
    """评估步骤"""
    model.eval()
    y_pred_list, y_true_list, loss_list = [], [], []
    strs_list = []
    
    with torch.no_grad():
        for (labels, strs, masked_strs, toks, masked_toks, _) in dataloader:
            strs_list.extend(strs)
            toks = toks.to(device)
            labels = torch.FloatTensor(labels).to(device).reshape(-1, 1)
            
            outputs = model(toks, return_representation=True, return_contacts=True)
            y_true_list.extend(labels.cpu().reshape(-1).tolist())
            y_pred = outputs.reshape(-1).cpu().detach().tolist()
            y_pred_list.extend(y_pred)
        
        if args.scaler:
            scaler = preprocessing.StandardScaler()
            scaler.fit(np.array(y_true_list).reshape(-1,1))
            y_pred_list = scaler.inverse_transform(np.array(y_pred_list).reshape(-1,1)).reshape(-1)
        if args.log2:
            y_pred_list = list(map(lambda x:math.pow(2,x), y_pred_list))
            
        loss_epoch = criterion(torch.Tensor(y_pred_list).reshape(-1,1), 
                             torch.Tensor(y_true_list).reshape(-1,1))
        
        print(f'Test: Epoch-{epoch}/{args.epochs} | Loss = {loss_epoch:.4f} | ', end='')
        metrics = performances(y_true_list, y_pred_list)
        
        e_pred = pd.DataFrame([strs_list, y_true_list, y_pred_list], 
                            index=['utr', 'y_true', 'y_pred']).T
        
        if data is not None:
            data_pred = pd.merge(e_pred, data, on=['utr'])
        else:
            data_pred = e_pred
            
    return metrics, loss_epoch, data_pred

def train_model(args):
    """模型训练主函数"""
    try:
        # 设置日志
        setup_logging(args)
        logging.info(f"Starting training with args: {args}")
        
        # 设置随机种子
        setup_seed(args.seed)
        
        # 设置设备
        device_ids, device = setup_device(args)
        
        # 获取模型信息
        layers, heads, embed_dim, batch_toks = get_model_info(args.modelfile)
        inp_len = 50  # Move this to args or config if needed
        
        # 初始化 alphabet
        alphabet = Alphabet.from_architecture("ESM-1b")
        
        # 加载数据
        original_train_data, data, e_test = load_data(args, args.seed)
        
        # 创建K折交叉验证
        kf = KFold(n_splits=args.folds, shuffle=False)
        
        # 训练循环
        loss_train_repeat_dict, loss_val_repeat_dict, loss_test_repeat_dict = dict(), dict(), dict()
        metrics_repeat_dict = dict()
        best_epoch_list = []
        
        for i, (train_index, val_index) in enumerate(kf.split(data)):
            print(f'====Begin Train Fold = {i}====')
            
            # 准备训练集和验证集
            e_train = data.iloc[train_index, :]
            e_val = data.iloc[val_index, :]
            
            # 数据预处理
            train_obj_col = prepare_data(args, e_train)
            
            # 准备数据加载器
            train_dataset, train_batches, train_batches_sampler, train_batches_loader = \
                generate_trainbatch_loader(e_train, train_obj_col, batch_toks, alphabet)
            val_dataset, val_dataloader = generate_dataset_dataloader(e_val, args.label_type, batch_toks, alphabet)
            
            # 创建模型
            model = create_model(args, device, device_ids, layers, heads, embed_dim, alphabet, inp_len)
            
            # 训练循环 - 添加 train_batches_sampler 和 e_train 参数
            best_model, metrics = train_fold(args, model, train_batches_loader, val_dataloader, 
                                        device, i, train_obj_col, train_batches_sampler, e_train)
            
            # 保存结果
            if args.test1fold:
                break
            best_epoch_list.append(metrics['best_epoch'])
        
        return metrics_repeat_dict
    except Exception as e:
            logging.error(f"Training failed with error: {str(e)}")
            raise e

def create_model(args, device, device_ids, layers, heads, embed_dim, alphabet, inp_len):
    """创建并初始化模型"""
    # Check if model files exist
    if args.load_wholemodel and not os.path.exists(args.finetune_modelfile):
        raise FileNotFoundError(
            f"Finetune model file not found: {args.finetune_modelfile}"
        )
    if not args.load_wholemodel and not os.path.exists(args.modelfile):
        raise FileNotFoundError(
            f"Model file not found: {args.modelfile}"
        )

    # Create model
    model = CNN_linear(args, embed_dim, inp_len, layers, heads, alphabet).to(device)
    
    # Load model weights
    storage_id = int(device_ids[args.local_rank])
    try:
        if args.load_wholemodel:
            state_dict = torch.load(args.finetune_modelfile, 
                                  map_location=lambda storage, loc: storage.cuda(storage_id))
        else:
            state_dict = torch.load(args.modelfile, 
                                  map_location=lambda storage, loc: storage.cuda(storage_id))
        
        # Remove 'module.' prefix if present
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        
        # Filter out parameters that do not match
        model_state = model.esm2.state_dict()
        filtered_state_dict = {}
        
        for name, param in state_dict.items():
            if name in model_state:
                if param.size() == model_state[name].size():
                    filtered_state_dict[name] = param
                else:
                    logging.warning(f"跳过参数 {name} (大小不匹配): "
                                  f"检查点大小 {param.size()} vs 模型大小 {model_state[name].size()}")
        
        # Load weights that match current model's architecture
        if args.load_wholemodel:
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            model.esm2.load_state_dict(filtered_state_dict, strict=False)
            
    except Exception as e:
        logging.error(f"加载模型权重失败: {str(e)}")
        raise

    # Set training mode
    if not args.finetune:
        for name, value in model.named_parameters():
            if 'esm2' in name:
                value.requires_grad = False
                
    if args.magic:
        for name, value in model.named_parameters():
            if 'magic_output' not in name:
                value.requires_grad = False
                
    return model

def train_fold(args, model, train_loader, val_loader, device, fold_idx, train_obj_col, train_sampler, train_data):
    """训练单个fold"""
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.lr,
                               momentum=0.9,
                               weight_decay=1e-4)
                               
    criterion = torch.nn.HuberLoss() if args.huber_loss else torch.nn.MSELoss()
    
    loss_best, ep_best, r2_best = np.inf, -1, -1
    loss_train_list, loss_val_list = [], []
    
    for epoch in trange(args.init_epochs+1, args.init_epochs + args.epochs + 2):
        train_sampler.set_epoch(epoch)
        
        metrics_train, loss_train = train_step(train_data, train_loader, model, 
                                             device, criterion, optimizer, 
                                             args, epoch, train_obj_col)
        loss_train_list.append(loss_train)
        
        if epoch == args.init_epochs+1:
            model_best = deepcopy(model)
            
        if args.local_rank == 0:
            metrics_val, loss_val, _ = eval_step(val_loader, model, device, 
                                               criterion, args, epoch)
            loss_val_list.append(loss_val)
            
            if args.epochs >= args.patience and metrics_val[2] > r2_best:
                save_best_model(model, args, fold_idx, epoch, loss_train, 
                              loss_val, metrics_val[2])
                r2_best, ep_best = metrics_val[2], epoch
                model_best = deepcopy(model)
                
            if epoch % args.log_interval == 0:
                generate_and_save_results(args, model_best, ep_best, fold_idx)
                
    return model_best, {'best_epoch': ep_best, 'best_r2': r2_best}