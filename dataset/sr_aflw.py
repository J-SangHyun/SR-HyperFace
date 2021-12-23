# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from dataset.aflw import AFLW, AFLWDataset


class SR_AFLW:
    def __init__(self, dataset_path, img_size, sr_model):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.sr_model = sr_model

        self.aflw = AFLW(self.dataset_path, self.img_size)
        self.sr_path = self.dataset_path / 'aflw' / 'sr'
        for img_dir in ['0', '2', '3']:
            (self.sr_path / img_dir).mkdir(parents=True, exist_ok=True)
        self.cached_path = self.dataset_path / 'aflw' / 'sr_aflw.npz'
        self.data = []

        self.load_data()

    def load_data(self):
        if os.path.exists(self.cached_path):
            self.data = np.load(self.cached_path, allow_pickle=True)['data']
            print('Cached SR-AFLW dataset loaded.')
            return

        print('Caching SR-AFLW dataset...')

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for i in tqdm(range(len(self.aflw.data))):
            data = self.aflw.data[i]
            path = self.aflw.cropped_path / data['path']
            img = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if img.shape[0] < self.img_size * 0.5:
                img = img.astype(np.float32)
                img = img * 1.0 / 255
                img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))
                img = torch.from_numpy(img).float().unsqueeze(0)

                img = img.to(device)
                img = self.sr_model(img).data.float().cpu().clamp_(0, 1)
                img = (img.numpy()[0] * 255).round()
                img = np.transpose(img[[2, 1, 0], :, :], (1, 2, 0))

            sr_img_path = self.sr_path / data['path']
            cv2.imwrite(str(sr_img_path), img)

        np.savez(self.cached_path, data=self.aflw.data)
        self.data = self.aflw.data

    def get_dataset(self, n_tests):
        if n_tests == 0:
            return AFLWDataset(self.sr_path, self.data, self.img_size)
        train_dataset = AFLWDataset(self.sr_path, self.data[:-n_tests], self.img_size)
        test_dataset = AFLWDataset(self.sr_path, self.data[-n_tests:], self.img_size)
        return train_dataset, test_dataset
