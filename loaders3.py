from torch.utils.data import Dataset
import numpy as np
from scipy.io import loadmat, savemat
import h5py
from utils import add_white_noise, ispadding
import random
import mne
import os

class SpikeEEGBuild(Dataset):
    def __init__(self, data_root, fwd, transform=None, args_params=None):
        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform
        
        if args_params is None:
            args_params = {}

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
            
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]

    def __getitem__(self, index):
        if not self.data:
            self.data = h5py.File('{}_nmm.h5'.format(self.file_path[:-12]), 'r')['data']
            
        raw_lb = self.dataset_meta['selected_region'][index].astype(int)
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]
        raw_nmm = np.zeros((500, self.fwd.shape[1]))

        for kk in range(raw_lb.shape[0]):
            curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
            current_nmm = self.data[self.dataset_meta['index1'][index][kk]]
            ssig = current_nmm[:, [curr_lb[0]]]
            
            ssig = ssig / np.max(ssig) * self.dataset_meta['scale_ratio'][index][kk][random.randint(0, self.num_scale_ratio - 1)]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)
            
            weight_decay = self.dataset_meta['mag_change'][index][kk]
            weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay

            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())
        csnr = self.dataset_meta['current_snr'][index]
        noisy_eeg = add_white_noise(eeg, csnr).transpose()

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))
        
        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        max_value = np.max(empty_nmm)
        empty_nmm = empty_nmm / max_value

        sample = {
            'data': noisy_eeg.astype('float32'),
            'nmm': empty_nmm.astype('float32'),
            'labels': raw_lb,
            'snr': csnr
        }
        
        if self.transform:
            sample = self.transform(sample)
            
        return sample

    def __len__(self):
        return self.dataset_len

class SpikeEEGBuildEval(Dataset):
    def __init__(self, data_root, fwd, transform=None, args_params=None):
        self.file_path = data_root
        self.fwd = fwd
        self.transform = transform

        self.data = []
        self.dataset_meta = loadmat(self.file_path)
        self.eval_params = {}

        if args_params is None:
            args_params = {}
            
        if 'dataset_len' in args_params:
            self.dataset_len = args_params['dataset_len']
        else:
            self.dataset_len = self.dataset_meta['selected_region'].shape[0]
            
        if 'num_scale_ratio' in args_params:
            self.num_scale_ratio = args_params['num_scale_ratio']
        else:
            self.num_scale_ratio = self.dataset_meta['scale_ratio'].shape[2]

        if 'snr_rsn_ratio' in args_params and args_params['snr_rsn_ratio']:
            self.eval_params['rsn'] = loadmat('anatomy/realistic_noise.mat')
            self.eval_params['snr_rsn_ratio'] = args_params['snr_rsn_ratio']
            
        if 'lfreq' in args_params and args_params['lfreq'] > 0:
            if 'hfreq' in args_params and args_params['hfreq'] > 0:
                self.eval_params['lfreq'] = args_params['lfreq']
                self.eval_params['hfreq'] = args_params['hfreq']
            else:
                print('Warning: need to assign both low-pass and high-pass cut-off frequencies. Ignoring filtering.')

    def __getitem__(self, index):
        if not self.data:
            self.data = h5py.File('{}_nmm.h5'.format(self.file_path[:-12]), 'r')['data']

        raw_lb = self.dataset_meta['selected_region'][index].astype(np.int64)
        lb = raw_lb[np.logical_not(ispadding(raw_lb))]
        raw_nmm = np.zeros((500, self.fwd.shape[1]))
        
        for kk in range(raw_lb.shape[0]):
            curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
            current_nmm = self.data[self.dataset_meta['index1'][index][kk]]
            ssig = current_nmm[:, [curr_lb[0]]]
            
            ssig = ssig / np.max(ssig) * self.dataset_meta['scale_ratio'][index][kk][random.randint(0, self.num_scale_ratio - 1)]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1)

            weight_decay = self.dataset_meta['mag_change'][index][kk]
            weight_decay = weight_decay[np.logical_not(ispadding(weight_decay))]
            current_nmm[:, curr_lb] = ssig.reshape(-1, 1) * weight_decay

            raw_nmm = raw_nmm + current_nmm

        eeg = np.matmul(self.fwd, raw_nmm.transpose())
        csnr = self.dataset_meta['current_snr'][index]

        if 'rsn' in self.eval_params:
            noisy_eeg = add_white_noise(eeg, csnr, {
                'ratio': self.eval_params['snr_rsn_ratio'],
                'rndata': self.eval_params['rsn']['data'],
                'rnpower': self.eval_params['rsn']['npower']
            }).transpose()
        else:
            noisy_eeg = add_white_noise(eeg, csnr).transpose()

        if 'lfreq' in self.eval_params:
            noisy_eeg = mne.filter.filter_data(
                np.tile(noisy_eeg.transpose(), (1, 5)), 
                500, 
                self.eval_params['lfreq'], 
                self.eval_params['hfreq'],
                verbose=False
            ).transpose()
            noisy_eeg = noisy_eeg[1000:1500]

        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=0, keepdims=True)
        noisy_eeg = noisy_eeg - np.mean(noisy_eeg, axis=1, keepdims=True)
        noisy_eeg = noisy_eeg / np.max(np.abs(noisy_eeg))

        empty_nmm = np.zeros_like(raw_nmm)
        empty_nmm[:, lb] = raw_nmm[:, lb]
        empty_nmm = empty_nmm / np.max(empty_nmm)

        sample = {
            'data': noisy_eeg.astype('float32'),
            'nmm': empty_nmm.astype('float32'),
            'label': raw_lb,
            'snr': csnr
        }
        
        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.dataset_len
