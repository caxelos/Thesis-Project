# coding: utf-8

import os
import numpy as np

import torch
import torch.utils.data


class MPIIGazeDataset(torch.utils.data.Dataset):
    def __init__(self, subject_id, dataset_dir):
        path = os.path.join(dataset_dir, '{}.npz'.format(subject_id))
        with np.load(path) as fin:
            self.images = fin['image']
            self.poses = fin['pose']
            self.gazes = fin['gaze']
        self.length = len(self.images)

        self.images = torch.unsqueeze(torch.from_numpy(self.images), 1)
        self.poses = torch.from_numpy(self.poses)
        self.gazes = torch.from_numpy(self.gazes)

    def __getitem__(self, index):
        return self.images[index], self.poses[index], self.gazes[index]

    def __len__(self):
        return self.length

    def __repr__(self):
        return self.__class__.__name__


def get_loader(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu):
    assert os.path.exists(dataset_dir)
    assert test_subject_id in range(15)

    subject_ids = ['p{:02}'.format(index) for index in range(15)]
    test_subject_id = subject_ids[test_subject_id]
    print("test:",test_subject_id)

    train_dataset = torch.utils.data.ConcatDataset([
        MPIIGazeDataset(subject_id, dataset_dir) for subject_id in subject_ids
        if subject_id != test_subject_id
    ])
    test_dataset = MPIIGazeDataset(test_subject_id, dataset_dir)

    assert len(train_dataset) == 42000
    assert len(test_dataset) == 3000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader


# train_loader, test_loader = get_loader(
#        args.dataset=data, args.test_id=0, args.batch_size=32, args.num_workers, True)

def get_loader_multiview_mpiigaze(trainset_dir, testset_dir, batch_size, num_workers, use_gpu):
#def get_loader_multiview_mpiigaze(dataset_dir, test_subject_id, batch_size, num_workers, use_gpu):
    

    assert os.path.exists(trainset_dir)
    assert os.path.exists(testset_dir)
    #assert test_subject_id in range(15)

    trainsubject_ids = ['p{:02}'.format(index) for index in range(50)]
    testsubject_id = ['p{:02}'.format(index) for index in range(15)]

    train_dataset = torch.utils.data.ConcatDataset([
        MPIIGazeDataset(subject_id, trainset_dir) for subject_id in trainsubject_ids
    ])
    train_dataset = torch.utils.data.ConcatDataset([
        MPIIGazeDataset(subject_id, testset_dir) for subject_id in trainsubject_ids
    ])


    assert len(train_dataset) == 64000
    assert len(test_dataset) == 45000

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_gpu,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=use_gpu,
        drop_last=False,
    )
    return train_loader, test_loader
