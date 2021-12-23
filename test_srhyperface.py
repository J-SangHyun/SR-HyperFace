# -*- coding: utf-8 -*-
import os
import cv2
import torch
import numpy as np
from torch.utils.data import DataLoader
from pathlib import Path

from dataset.sr_aflw import SR_AFLW
from HyperFace.model import HyperFace

root = Path('./')
data_root = root / 'dataset'
ckpt_root = root / 'HyperFace' / 'checkpoints' / 'sr'
ckpt_root.mkdir(parents=True, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

sr_aflw = SR_AFLW(data_root, 227, None)
train_dataset, test_dataset = sr_aflw.get_dataset(n_tests=1000)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

model = HyperFace().to(device)

best_path = ckpt_root / 'best.pth'
if os.path.exists(best_path):
    ckpt = torch.load(best_path)
    model.load_state_dict(ckpt['model'])
    last_epoch = ckpt['epoch']
    best_val_loss = ckpt['val_loss']
    print('Best checkpoint is loaded.')
    print(f'Last Epoch: {ckpt["epoch"]} |',
          f'Last Avg Train Loss: {ckpt["train_loss"]} |',
          f'Last Avg Val Loss: {ckpt["val_loss"]}')
else:
    print('No checkpoint is found.')

model.eval()
val_iter, val_loss = 0, 0

conn_list = [[0, 1], [1, 2], [3, 4], [4, 5], [6, 7], [7, 8], [9, 10], [10, 11], [13, 14],
             [14, 15], [13, 15], [17, 18], [18, 19], [12, 20], [16, 20]]

with torch.no_grad():
    for img, y, mask in test_loader:
        x = img.to(device)
        pred = model(x).cpu().numpy()[0] * 227
        img = img.numpy()[0]
        img = np.transpose(img[:, :, :], (1, 2, 0))
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pts = []
        for i in range(21):
            x, y = int(pred[2*i]), int(pred[2*i+1])
            pts.append((x, y))
            if mask[0][2*i] == 1.0:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        for c in conn_list:
            if mask[0][2*c[0]] == 1.0 and mask[0][2*c[1]] == 1.0:
                cv2.line(img, pts[c[0]], pts[c[1]], (0, 0, 255), 1)
        cv2.imshow('img', img)
        cv2.waitKey(0)
