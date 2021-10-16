from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from data_loading.PathMinerLoader import PathMinerLoader
from data_loading.PathMinerSnapshotLoader import PathMinerSnapshotLoader
from data_loading.UtilityEntities import PathContexts, path_contexts_from_index

import ipdb

class PathMinerDataset(Dataset):
    """
    Transforms data, loaded from AstMiner's output, to format suitable for model training.
    """
    def __init__(self, path_contexts: Union[PathContexts, Tuple[List, List]], labels: np.ndarray, mode='nn', should_pad: bool = True):
        self._mode = mode
        self._size = labels.size
        self._labels = torch.LongTensor(labels)
        # For training PbNN
        if mode == 'nn':
            self._contexts = path_contexts
            self._should_pad = should_pad
            if should_pad:
                self._pad_length = 500
        # For training PbRF/JCaliskan
        else:
            self._tokens, self._paths = path_contexts

    @classmethod
    def from_loader(cls, loader: PathMinerLoader, indices: np.ndarray = None, should_pad: bool = True, fold_ind: int = 0, set: str = 'train'):
        """
        Prepares a dataset suitable for PbNN training.
        """
        def dataset_export(indices, label: np.ndarray, method_ids: np.ndarray, depth: int, split: str):
            # method_ids: [('changeId','authorName','methodAfterId')]

            data_dict = {
                'label': label[indices],
                'change_id': method_ids[:, 0][indices],
                'author_name': method_ids[:, 1][indices],
                'method_id': method_ids[:, 2][indices],
            }
            df = pd.DataFrame(data_dict)
            df.to_csv(f'parsed_result_{depth}_{split}.csv')
            
        dataset_export(indices, loader.labels(), loader.method_ids(), fold_ind, set)
        return
        contexts, labels = (path_contexts_from_index(loader.path_contexts(), indices), loader.labels()[indices]) \
            if indices is not None \
            else (loader.path_contexts(), loader.labels())

        return cls(contexts, labels, 'nn', should_pad)

    @classmethod
    def from_rf_loader(cls, loader: PathMinerSnapshotLoader, indices: np.ndarray = None):
        """
        Prepares a dataset suitable for PbRF/JCaliskan training.
        """
        tokens = loader._tokens_by_author if indices is None else loader._tokens_by_author[indices]
        paths = loader._paths_by_author if indices is None else loader._paths_by_author[indices]
        labels = loader.labels() if indices is None else loader.labels()[indices]
        return cls((tokens, paths), labels, 'rf')

    def __len__(self):
        return self._size

    def __getitem__(self, index):
        """
        For PbNN pads tensors with zeros and picks random path-contexts if their number exceeds padding length.
        """
        if self._mode == 'nn':
            cur_len = len(self._contexts.starts[index])
            starts = torch.LongTensor(self._contexts.starts[index])
            paths = torch.LongTensor(self._contexts.paths[index])
            ends = torch.LongTensor(self._contexts.ends[index])
            if self._should_pad:
                if cur_len < self._pad_length:
                    return {
                        'starts': F.pad(starts, [0, self._pad_length - cur_len], mode='constant', value=0),
                        'paths': F.pad(paths, [0, self._pad_length - cur_len], mode='constant', value=0),
                        'ends': F.pad(ends, [0, self._pad_length - cur_len], mode='constant', value=0),
                        'labels': self._labels[index]
                    }
                else:
                    inds = np.random.permutation(cur_len)[:self._pad_length]
                    return {
                        'starts': starts[inds],
                        'paths': paths[inds],
                        'ends': ends[inds],
                        'labels': self._labels[index]
                    }
            else:
                return {
                    'starts': starts,
                    'paths': paths,
                    'ends': ends,
                    'labels': self._labels[index]
                }
        else:
            return {
                'starts': self._tokens[index],
                'paths': self._paths[index],
                'ends': self._paths[index],
                'labels': self._labels[index]
            }

    def labels(self) -> np.ndarray:
        return self._labels.numpy()
