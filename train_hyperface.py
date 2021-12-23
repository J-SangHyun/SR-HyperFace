# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path

from dataset.aflw import AFLW
from HyperFace.model import HyperFace

root = Path('./')
data_root = root / 'dataset'
ckpt_root = root / 'HyperFace' / 'checkpoints' / 'normal'
ckpt_root.mkdir(parents=True, exist_ok=True)
log_dir = ckpt_root / 'log'
log_dir.mkdir(parents=True, exist_ok=True)
writer = SummaryWriter(log_dir)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# hyper-parameters
lr = 1e-2
batch_size = 64
max_epoch = 1000

aflw = AFLW(data_root, 227)
train_dataset, test_dataset = aflw.get_dataset(n_tests=1000)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = HyperFace().to(device)
optimizer = optim.SGD(model.parameters(), lr=lr)

last_path = ckpt_root / 'last.pth'
best_path = ckpt_root / 'best.pth'
best_val_loss = np.inf
last_epoch = 0
if os.path.exists(last_path):
    ckpt = torch.load(last_path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    last_epoch = ckpt['epoch']
    best_val_loss = ckpt['val_loss']
    print('Last checkpoint is loaded.')
    print(f'Last Epoch: {ckpt["epoch"]} |',
          f'Last Avg Train Loss: {ckpt["train_loss"]} |',
          f'Last Avg Val Loss: {ckpt["val_loss"]}')
else:
    print('No checkpoint is found.')

for epoch in range(last_epoch+1, max_epoch+1):
    start_epoch = time.time()
    print(f'-------- EPOCH {epoch} / {max_epoch} --------')

    model.train()
    train_iter, train_loss = 0, 0
    for x, y, mask in train_loader:
        x = x.to(device)
        y = y.to(device)
        mask = mask.to(device)

        pred = model(x)
        loss = model.compute_loss(pred, y, mask)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_iter += 1

    model.eval()
    val_iter, val_loss = 0, 0
    with torch.no_grad():
        for x, y, mask in test_loader:
            x = x.to(device)
            y = y.to(device)
            mask = mask.to(device)

            pred = model(x)
            loss = model.compute_loss(pred, y, mask)

            val_loss += loss.item()
            val_iter += 1

    print(f'EPOCH {epoch}/{max_epoch} |',
          f'Avg train loss: {train_loss / train_iter} |',
          f'Avg val loss: {val_loss / val_iter}')
    print(f'This epoch took {time.time() - start_epoch} seconds')

    writer.add_scalar('train_loss', train_loss / train_iter, epoch)
    writer.add_scalar('val_loss', val_loss / val_iter, epoch)
    writer.flush()

    ckpt = {'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'train_loss': train_loss / train_iter,
            'val_loss': val_loss / val_iter}
    torch.save(ckpt, last_path)

    if val_loss / val_iter < best_val_loss:
        best_val_loss = val_loss / val_iter
        torch.save(ckpt, best_path)
        print('New Best Model!')

writer.close()
