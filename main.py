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
import network33

def main():
    start_time = time.time()
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='ASTR-Net')
    
    # Basic parameters
    parser.add_argument('--save', type=int, default=True, help='Save checkpoint for each epoch or not')
    parser.add_argument('--workers', default=16, type=int, help='Number of data loading workers')
    parser.add_argument('--batch_size', default=16, type=int, help='Batch size')
    parser.add_argument('--device', default='cuda:3', type=str, help='Device for training')
    parser.add_argument('--arch', default='EEGSourceLocalizationModel', type=str, help='Network architecture')
    parser.add_argument('--dat', default='SpikeEEGBuild', type=str, help='Data loader class name')
    parser.add_argument('--train', default='train.mat', type=str, help='Train dataset name')
    parser.add_argument('--test', default='test.mat', type=str, help='Test dataset name')
    parser.add_argument('--lr', default=3e-4, type=float, help='Learning rate for normal training')
    parser.add_argument('--epoch', default=150, type=int, help='Total epochs to run')
    parser.add_argument('--fwd', default='sub1.mat', type=str, help='Forward matrix file name')
    #parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str, help='Forward matrix file name')
    parser.add_argument('--info', default='sub1', type=str, help='Additional information')
    parser.add_argument('--resume', default='', type=str, help='Epoch ID to resume from')

    # Fine-tuning parameters
    parser.add_argument('--finetune', action='store_true', help='Activate fine-tuning mode')
    parser.add_argument('--finetune_path', type=str, default='1', help='Path to pretrained model')
    parser.add_argument('--lr_spatial', type=float, default=3e-4, help='LR for spatial module during fine-tuning')
    parser.add_argument('--lr_temporal', type=float, default=3e-5, help='LR for temporal module during fine-tuning')
    
    args = parser.parse_args()

    # Setup device
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Setup directories
    data_root = 'source/simulation/'
    result_root = f'model_result/{args.model_id}_the_model'
    if not os.path.exists(result_root):
        os.makedirs(result_root)

    # Setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(os.path.join(result_root, f'outputs_{args.arch}.log'))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    
    logger.info(f"Training started at {datetime.datetime.now()}")
    logger.info(f"Training data: {args.train}, Testing data: {args.test}")
    
    # Load forward matrix
    print("Loading forward matrix...")
    fwd = loadmat(f'anatomy/{args.fwd}')['fwd']
    
    # Load datasets
    print("Loading datasets...")
    train_data = loaders3.__dict__[args.dat](
        os.path.join(data_root, args.train), 
        fwd=fwd, 
        args_params={'dataset_len': 44712}
    )
    train_loader = DataLoader(train_data, batch_size=args.batch_size, 
                              num_workers=args.workers, shuffle=True, pin_memory=True)

    test_data = loaders3.__dict__[args.dat](
        os.path.join(data_root, args.test), 
        fwd=fwd, 
        args_params={'dataset_len': 3000}
    )
    test_loader = DataLoader(test_data, batch_size=args.batch_size, 
                             num_workers=args.workers, pin_memory=False)

    # Create model
    net = network33.__dict__[args.arch]().to(device)
    
    # Setup optimizer and load weights based on mode
    if args.finetune:
        print("\n" + "="*30 + " FINETUNING MODE " + "="*30)
        logger.info("FINETUNING MODE ACTIVATED")

        if not os.path.isfile(args.finetune_path):
            raise FileNotFoundError(f"Finetuning model not found: {args.finetune_path}")

        checkpoint = torch.load(args.finetune_path, map_location=device)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        print("Pretrained weights loaded successfully.")

        optimizer = torch.optim.AdamW([
            {'params': net.spatial_module.parameters(), 'lr': args.lr_spatial},
            {'params': net.temporal_module.parameters(), 'lr': args.lr_temporal}
        ], weight_decay=0.01)

        args.start_epoch = 0
        best_result = np.Inf
        train_loss, test_loss = [], []
    
    else:
        print("\n" + "="*30 + " NORMAL TRAINING / RESUME MODE " + "="*30)
        logger.info("NORMAL TRAINING / RESUME MODE ACTIVATED")

        optimizer = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=0.01)

        if args.resume:
            checkpoint_path = os.path.join(result_root, f'epoch_{args.resume}')
            if os.path.isfile(checkpoint_path):
                checkpoint = torch.load(checkpoint_path, map_location=device)
                args.start_epoch = checkpoint['epoch']
                best_result = checkpoint['best_result']
                net.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
                print(f"Resumed from epoch {args.start_epoch}.")
                
                loss_history_path = os.path.join(result_root, 'train_test_error.mat')
                if os.path.isfile(loss_history_path):
                    tte = loadmat(loss_history_path)
                    train_loss = tte['train_loss'][0][:args.start_epoch].tolist()
                    test_loss = tte['test_loss'][0][:args.start_epoch].tolist()
                else:
                    train_loss, test_loss = [], []
            else:
                print(f"Checkpoint '{args.resume}' not found. Starting from scratch.")
                args.start_epoch = 0
                best_result = np.Inf
                train_loss, test_loss = [], []
        else:
            args.start_epoch = 0
            best_result = np.Inf
            train_loss, test_loss = [], []

    # Setup scheduler and loss function
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True)
    criterion = torch.nn.MSELoss(reduction='sum')
    
    print(f'\nNumber of trainable parameters: {net.count_parameters()}')
    print(f'Preparation time: {time.time() - start_time:.2f} seconds\n')

    # Training loop
    for epoch in range(args.start_epoch + 1, args.epoch + 1):
        epoch_start_time = time.time()

        train_lss_all = train(train_loader, net, criterion, optimizer, {'device': device, 'logger': logger})
        test_lss_all = validate(test_loader, net, criterion, {'device': device})
        
        train_loss.append(train_lss_all)
        test_loss.append(test_lss_all)
        
        lr_scheduler.step(test_lss_all)

        current_lr = optimizer.param_groups[0]['lr']
        print_s = (f"Epoch {epoch}/{args.epoch} [{time.time() - epoch_start_time:.2f}s] | "
                   f"Train Loss: {train_loss[-1]:.6f} | Test Loss: {test_loss[-1]:.6f} | "
                   f"Best Test Loss: {best_result:.6f} | LR: {current_lr:.2e}")
        logger.info(print_s)
        print(print_s)

        # Check if best model
        is_best = test_loss[-1] < best_result
        if is_best:
            print("New best model found!")
            best_result = test_loss[-1]
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(result_root, 'model_best.pth.tar'))

        # Save periodic checkpoint
        if args.save:
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'optimizer': optimizer.state_dict(),
                'args': args
            }, os.path.join(result_root, f'epoch_{epoch}'))
        
        # Save loss history
        savemat(os.path.join(result_root, 'train_test_error.mat'), 
                {'train_loss': train_loss, 'test_loss': test_loss})

    print(f"\nTraining finished. Total time: {(time.time() - start_time) / 3600:.2f} hours.")
    logger.info(f"Training finished. Best test loss: {best_result}")

def train(train_loader, model, criterion, optimizer, args_params):
    device = args_params['device']
    max_norm = 0.5

    model.train()
    total_loss = 0.0
    
    for sample_batch in tqdm(train_loader, desc="Training", leave=False):
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
