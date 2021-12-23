# -*- coding: utf-8 -*-
import os
import cv2
import random
import sqlite3
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset


class AFLW:
    def __init__(self, dataset_path, img_size):
        self.dataset_path = dataset_path
        self.img_size = img_size

        self.cropped_path = self.dataset_path / 'aflw' / 'cropped'
        self.cropped_path.mkdir(parents=True, exist_ok=True)
        for img_dir in ['0', '2', '3']:
            (self.cropped_path / img_dir).mkdir(parents=True, exist_ok=True)
        self.cached_path = self.dataset_path / 'aflw' / 'aflw.npz'
        self.data = []

        self.load_data()

    def load_data(self):
        if os.path.exists(self.cached_path):
            self.data = np.load(self.cached_path, allow_pickle=True)['data']
            print('Cached AFLW dataset loaded.')
            return

        print('Caching AFLW dataset...')
        fid2data = {}

        sqlite_path = self.dataset_path / 'aflw' / 'aflw.sqlite'
        conn = sqlite3.connect(str(sqlite_path))
        cursor = conn.cursor()

        select_str = 'faces.face_id, imgs.filepath, rect.x, rect.y, rect.w, rect.h'
        from_str = 'faces, faceimages imgs, facerect rect'
        where_str = 'faces.file_id = imgs.file_id and faces.face_id = rect.face_id'
        res = self.execute_sql(cursor, select_str, from_str, where_str)

        for fid, path, x, y, w, h in res:
            landmark = np.zeros(42, dtype=np.float32)
            visibility = np.zeros(21, dtype=np.float32)
            fid2data[fid] = {'path': path, 'rect': (x, y, w, h), 'landmark': landmark, 'visib': visibility}

        select_str = 'faces.face_id, coords.feature_id, coords.x, coords.y'
        from_str = 'faces, featurecoords coords'
        where_str = 'faces.face_id = coords.face_id'
        res = self.execute_sql(cursor, select_str, from_str, where_str)
        cursor.close()

        for fid, feat_id, x, y in res:
            if fid in fid2data:
                fid2data[fid]['landmark'][2*(feat_id-1)], fid2data[fid]['landmark'][2*(feat_id-1)+1] = x, y
                fid2data[fid]['visib'][feat_id-1] = 1

        raw_data = list(fid2data.values())
        data = []
        for rd in tqdm(raw_data):
            x, y, w, h = rd['rect']
            img = cv2.imread(str(self.dataset_path / 'aflw' / 'flickr' / rd['path']), cv2.IMREAD_COLOR)
            img = img[max(0, y):y + h, max(0, x):x + w, :]
            new_h, new_w, _ = img.shape

            for i in range(21):
                lx, ly = int(rd['landmark'][2*i]), int(rd['landmark'][2*i+1])
                lx, ly = 1.0 * (lx - x) / new_w, 1.0 * (ly - y) / new_h
                rd['landmark'][2 * i], rd['landmark'][2 * i + 1] = lx, ly
                if lx < 0 or lx > 1 or ly < 0 or ly > 1:
                    rd['visib'][i] = 0

            new_data = {}
            crop_img_path = self.cropped_path / rd['path']
            new_data['path'] = rd['path']
            new_data['landmark'] = rd['landmark']
            new_data['visib'] = rd['visib']

            cv2.imwrite(str(crop_img_path), img)
            data.append(new_data)
        random.shuffle(data)
        np.savez(self.cached_path, data=data)
        self.data = data

    @staticmethod
    def execute_sql(cursor, select_str, from_str, where_str):
        query = f'SELECT {select_str} FROM {from_str} WHERE {where_str}'
        return [row for row in cursor.execute(query)]

    def get_dataset(self, n_tests):
        if n_tests == 0:
            return AFLWDataset(self.cropped_path, self.data, self.img_size)
        train_dataset = AFLWDataset(self.cropped_path, self.data[:-n_tests], self.img_size)
        test_dataset = AFLWDataset(self.cropped_path, self.data[-n_tests:], self.img_size)
        return train_dataset, test_dataset


class AFLWDataset(Dataset):
    def __init__(self, img_path, data, img_size):
        self.img_path = img_path
        self.data = data
        self.img_size = img_size

    def __getitem__(self, idx):
        path = self.img_path / self.data[idx]['path']
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img.astype(np.float32)
        img = img * 1.0 / 255
        img = np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))

        landmark = self.data[idx]['landmark'].reshape(42)
        visibility = self.data[idx]['visib']
        mask = np.zeros(42, dtype=np.float32)
        for i in range(21):
            mask[2*i] = visibility[i]
            mask[2*i+1] = visibility[i]

        return img, landmark, mask

    def __len__(self):
        return len(self.data)
