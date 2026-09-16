import time
import argparse
import os
import logging
import datetime
import torch
import numpy as np
from scipy.io import loadmat, savemat
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm

import loaders3
import networkASTRNet


def main():
    start_time = time.time()
    # ======================= 1. 解析输入参数 =========================================================================
    parser = argparse.ArgumentParser(description='ASTR Model Training')
    # --- 常规参数 ---
    parser.add_argument('--save', type=int, default=True, help='Save checkpoint for each epoch or not')
    parser.add_argument('--workers', default=2, type=int, help='Number of data loading workers')
    parser.add_argument('--batch_size', default=24, type=int, help='Batch size')
    parser.add_argument('--device', default='cuda:3', type=str, help='Device for training (e.g., cuda:0 or cpu)')
    parser.add_argument('--arch', default='EEGSourceLocalizationModel', type=str, help='Network architecture class name')
    parser.add_argument('--dat', default='SpikeEEGBuild', type=str, help='Data loader class name')
    parser.add_argument('--train', default='train.mat', type=str, help='Train dataset name or directory')
    parser.add_argument('--test', default='test.mat', type=str, help='Validation dataset name or directory')
    parser.add_argument('--model_id', default='2026-8.19-sub36-weitiao', type=str,
                        help='Output model ID; defaults to weitiao_<subject>_ES')
    parser.add_argument('--epoch', default=300, type=int, help='Maximum number of fine-tuning epochs')
    parser.add_argument('--subject', default='sub36', type=str,
                        help='Lead-field subject ID, e.g. sub02 or 2')
    parser.add_argument('--info', default='', type=str,
                        help='Optional run note; defaults to the resolved subject ID')
    # --- 第二阶段个体化微调参数 ---
    parser.add_argument('--finetune_path', type=str,
                        default='model_result/2026-8.06-jichu_the_model/model_best.pth.tar',
                        help='Best standard-head checkpoint used to initialize fine-tuning')
    parser.add_argument('--resume', type=str, default='',
                        help='Checkpoint to resume training from, e.g. model_result/.../epoch_30. '
                             'Leave empty to start a fresh fine-tuning run from --finetune_path.')
    parser.add_argument('--lr_spatial', type=float, default=3e-4, help='Learning rate for the spatial module during fine-tuning.')
    #parser.add_argument('--lr_temporal', type=float, default=3e-5, help='Learning rate for the temporal module during fine-tuning.')
    parser.add_argument('--lr_temporal', type=float, default=3e-5, help='Learning rate for the temporal module during fine-tuning.')
    parser.add_argument('--early_stopping_patience', default=15, type=int,
                        help='Stop after this many epochs without a validation-loss improvement')
    parser.add_argument('--early_stopping_min_delta', default=1e-3, type=float,
                        help='Minimum absolute validation-loss decrease counted as an improvement')
    parser.add_argument('--early_stopping_min_epochs', default=15, type=int,
                        help='Minimum number of completed epochs before early stopping')
    args = parser.parse_args()

    if args.early_stopping_patience < 1:
        parser.error('--early_stopping_patience must be at least 1')
    if args.early_stopping_min_delta < 0:
        parser.error('--early_stopping_min_delta must be non-negative')
    if args.early_stopping_min_epochs < 1:
        parser.error('--early_stopping_min_epochs must be at least 1')

    # Accept either "sub02" or the numeric shorthand "2". Prefer the actual
    # zero-padded directory when it exists, while retaining support for sub1.
    subject_arg = args.subject.strip()
    if subject_arg.isdigit():
        subject_number = int(subject_arg)
        padded_subject = f'sub{subject_number:02d}'
        unpadded_subject = f'sub{subject_number}'
        padded_dir = os.path.join('anatomy', 'new_fwd', padded_subject)
        args.subject = padded_subject if os.path.isdir(padded_dir) else unpadded_subject
    elif subject_arg.lower().startswith('sub'):
        args.subject = subject_arg.lower()
    else:
        parser.error('--subject must be a number such as 2 or an ID such as sub02')

    args.fwd = os.path.join('new_fwd', args.subject, 'fwd_75x994.mat')
    if not args.model_id:
        args.model_id = f'weitiao_{args.subject}_ES'
    if not args.info:
        args.info = args.subject
    # ======================= 2. 准备环境和日志 ========================================================================
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print(f"Using device: {device}")

    data_root = 'source/simulation/'
    result_root = f'model_result/{args.model_id}_the_model'
    if not os.path.exists(result_root):
        os.makedirs(result_root)

    # 配置日志记录器
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(result_root, f'outputs_{args.arch}.log'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info(f"============================= {datetime.datetime.now()} ====================================")
    logger.info(f"Training data is {args.train}, and validation data is {args.test}")
    for v, val in args.__dict__.items():
        if v not in ['workers']: # 记录所有重要参数
            logger.info(f'{v} is {val}')

    # ================================== 3. 加载数据 =================================================================
    print("Loading forward matrix...")
    fwd_path = os.path.join('anatomy', args.fwd)
    if not os.path.isfile(fwd_path):
        raise FileNotFoundError(f"Subject-specific lead field not found: {fwd_path}")
    print(f"Using subject-specific lead field: {fwd_path}")
    logger.info(f"Subject-specific lead field: {fwd_path}")
    fwd = loadmat(fwd_path)['fwd']
    
    print("Loading datasets...")
    train_data = loaders3.__dict__[args.dat](os.path.join(data_root, args.train), fwd=fwd, args_params={'dataset_len': 60000})
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)

    test_data = loaders3.__dict__[args.dat](os.path.join(data_root, args.test), fwd=fwd, args_params={'dataset_len': 6000})
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)

    # ================================== 4. 创建模型和优化器 (核心逻辑) ===============================================
    net = networkASTRNet.__dict__[args.arch]().to(device)
    
    # Build optimizer/scheduler first. Their states will be restored as well when --resume is used.
    print("=> Setting up optimizer with differential learning rates:")
    print(f"  - Spatial Module LR: {args.lr_spatial}")
    print(f"  - Temporal Module LR: {args.lr_temporal}")
    logger.info(f"Differential LR: Spatial={args.lr_spatial}, Temporal={args.lr_temporal}")

    optimizer = torch.optim.AdamW([
        {'params': net.spatial_module.parameters(), 'lr': args.lr_spatial},
        {'params': net.temporal_module.parameters(), 'lr': args.lr_temporal}
    ], weight_decay=0.01)

    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.2,
        patience=5,
        threshold=1e-3,
        threshold_mode='rel',
    )

    # Default state for a fresh fine-tuning run.
    args.start_epoch = 0
    best_result = np.inf
    early_stopping_best = np.inf
    early_stopping_counter = 0
    train_loss, test_loss = [], []

    if args.resume:
        # ======================== RESUME MODE ========================
        print("\n" + "="*30 + " RESUME MODE " + "="*30)
        logger.info("RESUME MODE ACTIVATED")

        if not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found at: {args.resume}")

        print(f"=> Resuming checkpoint from '{args.resume}'")
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        except TypeError:  # Compatibility with PyTorch versions before 2.6
            checkpoint = torch.load(args.resume, map_location=device)

        if 'arch' in checkpoint and checkpoint['arch'] != args.arch:
            raise ValueError(
                f"Checkpoint architecture is '{checkpoint['arch']}', but --arch is '{args.arch}'."
            )

        net.load_state_dict(checkpoint['state_dict'], strict=True)

        if 'optimizer' not in checkpoint:
            raise KeyError("Resume checkpoint has no 'optimizer' state; cannot perform an exact resume.")
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'lr_scheduler' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        else:
            print("=> Warning: no lr_scheduler state in checkpoint; scheduler will restart from fresh state.")
            logger.warning("No lr_scheduler state in resume checkpoint; scheduler restarted.")

        args.start_epoch = int(checkpoint.get('epoch', 0))
        best_result = float(checkpoint.get('best_result', np.inf))
        early_stopping_best = float(checkpoint.get('early_stopping_best', best_result))
        early_stopping_counter = int(checkpoint.get('early_stopping_counter', 0))

        # New checkpoints store loss history directly. For checkpoints generated by
        # the old script, fall back to train_test_error.mat in the output directory.
        if 'train_loss' in checkpoint and 'test_loss' in checkpoint:
            train_loss = list(checkpoint['train_loss'])
            test_loss = list(checkpoint['test_loss'])
        else:
            loss_history_path = os.path.join(result_root, 'train_test_error.mat')
            if os.path.isfile(loss_history_path):
                loss_history = loadmat(loss_history_path)
                train_loss = np.asarray(loss_history.get('train_loss', [])).reshape(-1).tolist()
                test_loss = np.asarray(loss_history.get('test_loss', [])).reshape(-1).tolist()
                # Keep history aligned with the resumed epoch if a longer MAT file exists.
                train_loss = train_loss[:args.start_epoch]
                test_loss = test_loss[:args.start_epoch]

        print(f"=> Resume successful. Last completed epoch: {args.start_epoch}")
        print(f"=> Best validation loss so far: {best_result:.8f}")
        print(f"=> Early stopping counter: {early_stopping_counter}/{args.early_stopping_patience}")
        print(f"=> Restored Spatial LR: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"=> Restored Temporal LR: {optimizer.param_groups[1]['lr']:.2e}")
        logger.info(
            f"Resumed from epoch {args.start_epoch}; best_result={best_result}; "
            f"early_stopping_counter={early_stopping_counter}; "
            f"spatial_lr={optimizer.param_groups[0]['lr']}; "
            f"temporal_lr={optimizer.param_groups[1]['lr']}"
        )
    else:
        # ====================== FRESH FINETUNE MODE ======================
        print("\n" + "="*30 + " FINETUNING MODE " + "="*30)
        logger.info("FINETUNING MODE ACTIVATED")

        if not os.path.isfile(args.finetune_path):
            raise FileNotFoundError(f"Finetuning model not found at: {args.finetune_path}")

        print(f"=> Loading pretrained weights from '{args.finetune_path}'")
        try:
            checkpoint = torch.load(args.finetune_path, map_location=device, weights_only=False)
        except TypeError:  # Compatibility with PyTorch versions before 2.6
            checkpoint = torch.load(args.finetune_path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'], strict=True)
        print("=> Pretrained weights loaded successfully.")

    criterion = torch.nn.MSELoss(reduction='sum')
    
    print(f'\nNumber of trainable parameters: {net.count_parameters()}')
    print(f'Preparation time: {time.time() - start_time:.2f} seconds\n')

    # =============================== 5. 训练循环 ======================================================================
    for epoch in range(args.start_epoch + 1, args.epoch + 1):
        epoch_start_time = time.time()

        # 在一个epoch上训练
        train_lss_all = train(train_loader, net, criterion, optimizer, {'device': device, 'logger': logger})
        
        # 在验证集上评估
        test_lss_all = validate(test_loader, net, criterion, {'device': device})
        
        train_loss.append(train_lss_all)
        test_loss.append(test_lss_all)
        
        # 更新学习率
        lr_scheduler.step(test_lss_all)

        # 打印并记录日志
        spatial_lr = optimizer.param_groups[0]['lr']
        temporal_lr = optimizer.param_groups[1]['lr']
        print_s = (f"Epoch {epoch}/{args.epoch} [{time.time() - epoch_start_time:.2f}s] | "
                   f"Train Loss: {train_loss[-1]:.6f} | Validation Loss: {test_loss[-1]:.6f} | "
                   f"Best Validation Loss: {best_result:.6f} | "
                   f"Spatial LR: {spatial_lr:.2e} | Temporal LR: {temporal_lr:.2e}")
        logger.info(print_s)
        print(print_s)

        # 检查是否是最佳模型
        current_validation_loss = float(test_loss[-1])
        is_best = current_validation_loss < best_result
        is_significant_improvement = current_validation_loss < (
            early_stopping_best - args.early_stopping_min_delta
        )
        if is_significant_improvement:
            early_stopping_best = current_validation_loss
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if is_best:
            print("  => New best model found!")
            best_result = current_validation_loss
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'early_stopping_best': early_stopping_best,
                'early_stopping_counter': early_stopping_counter,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'args': args # 保存所有参数以供复现
            }, os.path.join(result_root, 'model_best.pth.tar'))

        early_stopping_message = (
            f"Early stopping counter: {early_stopping_counter}/"
            f"{args.early_stopping_patience} "
            f"(absolute min_delta={args.early_stopping_min_delta}, "
            f"min_epochs={args.early_stopping_min_epochs})"
        )
        logger.info(early_stopping_message)
        print(early_stopping_message)

        # 保存周期性检查点
        if args.save:
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'early_stopping_best': early_stopping_best,
                'early_stopping_counter': early_stopping_counter,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'args': args
            }, os.path.join(result_root, f'epoch_{epoch}'))
        
        # 保存损失历史
        savemat(os.path.join(result_root, 'train_test_error.mat'), {'train_loss': train_loss, 'test_loss': test_loss})

        if (epoch >= args.early_stopping_min_epochs and
                early_stopping_counter >= args.early_stopping_patience):
            stop_message = (
                f"Early stopping triggered at epoch {epoch}. "
                f"Best validation loss: {best_result:.8f}."
            )
            logger.info(stop_message)
            print(stop_message)
            break

    best_checkpoint_path = os.path.join(result_root, 'model_best.pth.tar')
    if os.path.isfile(best_checkpoint_path):
        try:
            best_checkpoint = torch.load(
                best_checkpoint_path, map_location=device, weights_only=False
            )
        except TypeError:
            best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        net.load_state_dict(best_checkpoint['state_dict'], strict=True)
        restore_message = (
            f"Restored best weights from epoch {best_checkpoint['epoch']} "
            f"with validation loss {best_checkpoint['best_result']:.8f}."
        )
        logger.info(restore_message)
        print(restore_message)

    print(f"\nTraining finished. Total time: {(time.time() - start_time) / 3600:.2f} hours.")
    logger.info(f"Training finished. Best validation loss: {best_result}")

# The train and validate functions remain the same as in your provided code.
# I've included them here for completeness.

def train(train_loader, model, criterion, optimizer, args_params):
    device = args_params['device']
    logger = args_params['logger']
    max_norm = 0.5

    model.train()
    total_loss = 0.0
    
    for sample_batch in tqdm(train_loader, desc=f"Training", leave=False):
        data = sample_batch['data'].to(device)
        nmm = sample_batch['nmm'].to(device)

        optimizer.zero_grad()
        
        out = model(data)
        
        loss = criterion(out, nmm)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        optimizer.step()
        
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader.dataset)
    torch.cuda.empty_cache()
    return avg_loss

def validate(val_loader, model, criterion, args_params):
    device = args_params['device']
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for sample_batch in tqdm(val_loader, desc="Validating", leave=False):
            data = sample_batch['data'].to(device)
            nmm = sample_batch['nmm'].to(device)
            
            out = model(data)
            
            loss = criterion(out, nmm)
            total_loss += loss.item()
            
    avg_loss = total_loss / len(val_loader.dataset)
    return avg_loss

if __name__ == '__main__':
    main()