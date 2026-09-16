import argparse
import os
import os
import time
from scipy.io import loadmat, savemat
import numpy as np
import logging
import datetime
import Adamqwe
from Adamqwe import AdamW
import torch
from torch import optim
from torch.utils.data import DataLoader
from RAdam import RAdam
import networkASTRNet
import loaders3
import math
import os
import torch.nn as nn
import matplotlib.pyplot as plt
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

def main():
    start_time = time.time()
    # parse the input
    parser = argparse.ArgumentParser(description='DeepSIF Model')
    parser.add_argument('--save', type=int, default=True, help='save each epoch or not')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers')
    parser.add_argument('--batch_size', default=24, type=int, help='batch size')
    parser.add_argument('--device', default='cuda:3', type=str, help='device running the code')
    parser.add_argument('--arch', default='EEGSourceLocalizationModel', type=str, help='network achitecture class')
    parser.add_argument('--dat', default='SpikeEEGBuild', type=str, help='data loader')
    parser.add_argument('--train', default='train.mat', type=str, help='train dataset name or directory')
    parser.add_argument('--test', default='test.mat', type=str, help='test dataset name or directory')
    parser.add_argument('--model_id', default='', type=str, help='model id')
    # parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--lr', default=3e-4, type=float, help='learning rate')
    parser.add_argument('--resume', default='0', type=str, help='epoch id to resume')
    parser.add_argument('--epoch', default=300, type=int, help='total number of epoch')
    parser.add_argument('--early_stopping_patience', default=15, type=int,
                        help='stop after this many epochs without a validation-loss improvement')
    parser.add_argument('--early_stopping_min_delta', default=1e-3, type=float,
                        help='minimum validation-loss decrease counted as an improvement')
    parser.add_argument('--early_stopping_min_epochs', default=15, type=int,
                        help='minimum number of completed epochs before early stopping')
    parser.add_argument('--fwd', default='leadfield_75_20k.mat', type=str, help='forward matrix to use')
    parser.add_argument('--rnn_layer', default=2, type=int, help='number of rnn layer')
    parser.add_argument('--info', default='', type=str, help='other information regarding this model')
    

    args = parser.parse_args()
    if args.early_stopping_patience < 1:
        parser.error('--early_stopping_patience must be at least 1')
    if args.early_stopping_min_delta < 0:
        parser.error('--early_stopping_min_delta must be non-negative')
    if args.early_stopping_min_epochs < 1:
        parser.error('--early_stopping_min_epochs must be at least 1')

    # ======================= PREPARE PARAMETERS =====================================================================================================
    use_cuda = torch.cuda.is_available()
    device = torch.device(args.device if use_cuda else "cpu")
    
    data_root = 'source/simulation/'
    result_root = 'model_result/{}_the_model'.format(args.model_id)
    if not os.path.exists(result_root):
        os.makedirs(result_root)
    fwd = loadmat('anatomy/{}'.format(args.fwd))['fwd']

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(result_root + '/outputs_{}.log'.format(args.arch))
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.info("============================= {} ====================================".format(datetime.datetime.now()))
    logger.info("Training data is {}, and testing data is {}".format(args.train, args.test))
    # Save every parameters in args
    for v in args.__dict__:
        if v not in ['workers', 'train', 'test']:
            logger.info('{} is {}'.format(v, args.__dict__[v]))
 
    # ================================== LOAD DATA ===================================================================================================
    train_data = loaders3.__dict__[args.dat](data_root + args.train, fwd=fwd, args_params={'dataset_len': 60000})
    train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=args.workers, shuffle=True, pin_memory=True)
    
    #, args_params = {'dataset_len': 10000}
    test_data = loaders3.__dict__[args.dat](data_root + args.test,  fwd=fwd, args_params={'dataset_len': 6000})
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.workers, pin_memory=False)
  #, shuffle=False
    # ================================== CREATE MODEL ================================================================================================
    #net = xiaorongshiyanM2.__dict__[args.arch]().to(device)
    net = networkASTRNet.__dict__[args.arch]().to(device)

    
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)#之前是0.01
    # optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-6)
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, verbose=True, threshold=0.001)




    criterion = torch.nn.MSELoss(reduction='sum')
    args.start_epoch = 0
    best_result = np.inf
    early_stopping_best = np.inf
    early_stopping_counter = 0
    train_loss = []
    test_loss = []

    # =============================== RESUME =========================================================================================================
    if args.resume:
        print("=> Load checkpoint", args.resume, "from", result_root)
        fn = os.path.join(result_root, 'epoch_' + args.resume)
        if os.path.isfile(fn):
            print("=> Found checkpoint '{}'".format(args.resume))
            # checkpoint = torch.load(fn, map_location=torch.device('cpu'))
            checkpoint = torch.load(
                fn,
                map_location=torch.device('cpu'),
                weights_only=False
            )
            args.start_epoch = checkpoint['epoch']
            best_result = checkpoint['best_result']
            early_stopping_best = checkpoint.get('early_stopping_best', best_result)

            


            net =  networkASTRNet.__dict__[checkpoint['arch']]()
            net.load_state_dict(checkpoint['state_dict'], strict=True)
            net = net.to(device)
            
            optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=0.01)

            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.2,
                patience=5,
                verbose=True,
                threshold=0.001,
            )
            if 'lr_scheduler' in checkpoint:
                lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            early_stopping_counter = checkpoint.get('early_stopping_counter', 0)
            print("=> Loaded checkpoint epoch {}, current results: {}".format(args.resume, best_result))
            
            tte = loadmat(result_root + '/train_test_error.mat')
            train_loss.extend(tte['train_loss'][0][:int(args.resume) + 1])
            test_loss.extend(tte['test_loss'][0][:int(args.resume) + 1])
        else:
            print("WARNING: no checkpoint found at '{}', use random weights".format(args.resume))

    print('Number of parameters:', net.count_parameters())
    print('Prepare time:', time.time() - start_time)

    # =============================== TRAINING =======================================================================================================
    for epoch in range(args.start_epoch + 1, args.epoch):

        # train for one epoch
        train_lss_all = train(train_loader, net, criterion, optimizer, {'device': device, 'logger': logger})
        # evaluate on validation set
        test_lss_all = validate(test_loader, net, criterion, {'device': device})
        train_loss.append(train_lss_all)
        test_loss.append(test_lss_all)

        print_s = 'Epoch {}: Time:{:6.2f}, '.format(epoch, time.time() - start_time) + \
                  'Train Loss:{:06.5f}'.format(train_loss[-1]) + ', Test Loss:{:06.5f}'.format(test_loss[-1])
        logger.info(print_s)
        print(print_s)
        # scheduler.step(epoch)
        lr_scheduler.step(test_lss_all)
        # Track both the absolute best checkpoint and significant improvements
        # used by early stopping. Small numerical changes still update model_best,
        # but only changes larger than min_delta reset the patience counter.
        previous_best = best_result
        is_best = test_loss[-1] < previous_best
        is_significant_improvement = test_loss[-1] < (
            early_stopping_best - args.early_stopping_min_delta
        )
        best_result = min(test_loss[-1], best_result)
        if is_significant_improvement:
            early_stopping_best = test_loss[-1]
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        early_stopping_message = (
            'Early stopping counter: {}/{} (min_delta={}, min_epochs={})'.format(
                early_stopping_counter,
                args.early_stopping_patience,
                args.early_stopping_min_delta,
                args.early_stopping_min_epochs,
            )
        )
        logger.info(early_stopping_message)
        print(early_stopping_message)

        if is_best:
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'lr': args.lr,
                
                'info': args.info,
                'train': args.train,
                'test': args.test,
            
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'early_stopping_best': early_stopping_best,
                'early_stopping_counter': early_stopping_counter},
                result_root + '/model_best.pth.tar')

        if args.save:
            # 保存检查点
            torch.save({
                'epoch': epoch,
                'arch': args.arch,
                'state_dict': net.state_dict(),
                'best_result': best_result,
                'lr': args.lr,
                'info': args.info,
                'train': args.train,
                'test': args.test,
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'early_stopping_best': early_stopping_best,
                'early_stopping_counter': early_stopping_counter},
                result_root + '/epoch_{}'.format(epoch))

            savemat(result_root + '/train_test_error.mat', {'train_loss': train_loss, 'test_loss': test_loss})
            savemat(result_root + '/train_test_loss_epoch{}.mat'.format(epoch),
                    {'train_loss': train_lss_all, 'test_loss': test_lss_all})

        if (epoch >= args.early_stopping_min_epochs and
                early_stopping_counter >= args.early_stopping_patience):
            stop_message = (
                'Early stopping triggered at epoch {}. Best validation loss: '
                '{:.8f}. Best weights are saved in model_best.pth.tar.'.format(
                    epoch, best_result
                )
            )
            logger.info(stop_message)
            print(stop_message)
            break

    best_checkpoint_path = result_root + '/model_best.pth.tar'
    if os.path.isfile(best_checkpoint_path):
        # best_checkpoint = torch.load(best_checkpoint_path, map_location=device)
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location=device,
            weights_only=False
        )
        net.load_state_dict(best_checkpoint['state_dict'], strict=True)
        restore_message = (
            'Restored best weights from epoch {} with validation loss {:.8f}.'.format(
                best_checkpoint['epoch'], best_checkpoint['best_result']
            )
        )
        logger.info(restore_message)
        print(restore_message)





def train(train_loader, model, criterion, optimizer, args_params):
    device = args_params['device']
    logger = args_params['logger']
    
    max_norm = 0.5
    log_grad_interval = 10 
    # switch to train mode
    model.train()
    train_loss = []
    start_time = time.time()
    
    for batch_idx, sample_batch in enumerate(tqdm(train_loader, desc="Training")):
        
        data = sample_batch['data'].to(device)
        nmm = sample_batch['nmm'].to(device)


        optimizer.zero_grad()
        
        model_output = model(data)
        out = model_output

        loss = criterion(out, nmm)
        loss.backward() # 计算梯度

        
        original_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
        
        
        if (batch_idx + 1) % log_grad_interval == 0:
            clipped = original_total_norm.item() > max_norm
            log_message = (
                f"Batch {batch_idx+1}: "
                f"Gradient Norm BEFORE clipping = {original_total_norm.item():.4f}, "
                f"Clipping Occurred = {clipped}"
            )
        

        optimizer.step()
        
        train_loss.append(loss.item())


        if (batch_idx + 1) % 500 == 0:
            print_s = "batch_idx_{}_time_{}_train_loss_{}".format(batch_idx + 1, time.time() - start_time, train_loss[-1])
            logger.info(print_s)

    
    train_loss = np.sum(np.array(train_loss)) / len(train_loader.dataset) 
    torch.cuda.empty_cache()

    return  train_loss # 返回平均损失








def validate(val_loader, model, criterion, args_params):
    # switch to evaluate mode
    device = args_params['device']
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch_idx, sample_batch in enumerate(tqdm(val_loader, desc="Validating")):
            data = sample_batch['data'].to(device)
            nmm = sample_batch['nmm'].to(device)
            model_output = model(data)
            out = model_output
            loss = criterion(out, nmm)
            val_loss.append(loss.item())
    val_loss = np.sum(np.array(val_loss)) / len(val_loader.dataset)
    #val_loss = np.mean(np.array(val_loss))
    return val_loss
# END VALIDATE


if __name__ == '__main__':
   main()

