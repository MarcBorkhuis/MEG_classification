import os
import random
import logging
import h5py
from scipy.signal import resample
from scipy.stats import zscore
import numpy as np
import torch
from torch.utils.data import Dataset, random_split, Subset

from utils import string_to_int

LOG = logging.getLogger(__name__)

class EpochedDataset(Dataset):
    """Base class for M/EEG data, handles non-windowed properties."""
    def __init__(self, **kwargs):
        # FIX: Explicitly pop required arguments from kwargs.
        self.sfreq = kwargs.pop('sfreq')
        self.scaling = kwargs.pop('scaling')
        self.lso = kwargs.pop('lso')
        self.random_state = kwargs.pop('random_state')
        self.split_sizes = kwargs.pop('split_sizes')
        
        if self.scaling == "zscore":
            self.scaler = lambda x: zscore(x, axis=-1).astype(np.float32) if np.std(x) > 1e-8 else x.astype(np.float32)
        elif self.scaling == "minmax":
            self.scaler = lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-8)).astype(np.float32)
        else:
            raise ValueError(f"Invalid scaling option: {self.scaling}")
            
        self.groups = None
        self._reset_seed()

    def _reset_seed(self):
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def split_data(self):
        train_size, valid_size, test_size = self.split_sizes
        assert np.isclose(sum((train_size, valid_size, test_size)), 1.0), "Sum of data ratios must be equal to 1"
        
        generator = torch.Generator().manual_seed(self.random_state)
        n_total = len(self)
        if n_total == 0:
            LOG.warning("Dataset is empty, returning empty splits.")
            return [], [], []
            
        n_train = int(n_total * train_size)
        n_valid = int(n_total * valid_size)
        n_test = n_total - n_train - n_valid
        
        if n_train + n_valid + n_test != n_total:
            n_train = n_total - n_valid - n_test

        if self.lso and self.groups is not None and len(torch.unique(self.groups)) > 1:
            LOG.info("Performing Leave-Subject-Out (LSO) split.")
            return self._leave_subjects_out_split((train_size, valid_size, test_size), generator)
        else:
            LOG.info("Performing within-subject (random) split.")
            return random_split(range(n_total), [n_train, n_valid, n_test], generator)

    def _leave_subjects_out_split(self, sizes, generator):
        unique_subjects = torch.unique(self.groups).tolist()
        n_subjects = len(unique_subjects)
        
        subject_indices_shuffled = torch.randperm(n_subjects, generator=generator).tolist()
        
        n_train_subs = int(n_subjects * sizes[0])
        n_valid_subs = int(n_subjects * sizes[1])
        
        train_subs = [unique_subjects[i] for i in subject_indices_shuffled[:n_train_subs]]
        valid_subs = [unique_subjects[i] for i in subject_indices_shuffled[n_train_subs:n_train_subs + n_valid_subs]]
        
        train_indices = [i for i, g in enumerate(self.groups) if g in train_subs]
        valid_indices = [i for i, g in enumerate(self.groups) if g in valid_subs]
        test_indices = [i for i, g in enumerate(self.groups) if g not in train_subs and g not in valid_subs]

        LOG.info(f"LSO split: {len(train_indices)} train, {len(valid_indices)} valid, {len(test_indices)} test indices.")
        return train_indices, valid_indices, test_indices
        
    def __len__(self): raise NotImplementedError
    def __getitem__(self, idx): raise NotImplementedError

class ContinuousDataset(EpochedDataset):
    """Handles continuous data by adding windowing properties."""
    def __init__(self, **kwargs):
        # FIX: Pop arguments for this class before passing kwargs to parent.
        self.window = kwargs.pop('window')
        self.overlap = kwargs.pop('overlap')
        super().__init__(**kwargs)
        assert 0 <= self.overlap < 1, "Overlap must be between 0 and 1."

class CustomMEGDataset(ContinuousDataset):
    """Custom PyTorch Dataset for loading and preprocessing MEG data from H5 files."""
    def __init__(self, **kwargs):
        # FIX: Pop arguments for this class before passing kwargs to parent.
        self.data_root = kwargs.pop('data_root')
        self.scenario = kwargs.pop('scenario')
        self.mode = kwargs.pop('mode')
        self.task_to_label_map = kwargs.pop('task_to_label_map')
        self.original_sfreq = kwargs.pop('orig_sfreq', 2034)
        
        super().__init__(**kwargs)
        
        self._subject_str_to_int_map = {}
        self._next_subject_int_id = 0
        self.file_metadata = []
        self.total_windows = 0

        self.data_paths_to_load = self._get_data_paths()
        self._index_files()
        self._prepare_groups_for_splitting()

    def _index_files(self):
        LOG.info(f"Scanning and indexing files for scenario='{self.scenario}', mode='{self.mode}'...")
        window_samples = int(self.window * self.sfreq)
        step_samples = int(window_samples * (1 - self.overlap))
        if not self.data_paths_to_load:
            LOG.warning(f"No valid data paths found for '{self.mode}'. Dataset will be empty.")
            return
        for data_dir in self.data_paths_to_load:
            for filename in os.listdir(data_dir):
                if not filename.endswith(".h5"): continue
                filepath = os.path.join(data_dir, filename)
                task_type, subject_id = self._extract_info_from_filename(filename)
                if not (task_type and subject_id) or task_type not in self.task_to_label_map: continue
                try:
                    with h5py.File(filepath, 'r') as f:
                        h5_dataset_name = list(f.keys())[0]
                        total_timepoints = f[h5_dataset_name].shape[-1]
                except Exception as e:
                    LOG.error(f"Could not read metadata from {filepath}: {e}")
                    continue
                resampled_timepoints = int(total_timepoints * self.sfreq / self.original_sfreq)
                if resampled_timepoints < window_samples: continue
                num_windows = (resampled_timepoints - window_samples) // step_samples + 1
                if num_windows <= 0: continue
                label = self.task_to_label_map[task_type]
                group_id = self._get_or_create_subject_int_id(subject_id)
                self.file_metadata.append({"path": filepath, "label": label, "group": group_id, "start_index": self.total_windows, "n_windows": num_windows})
                self.total_windows += num_windows
        LOG.info(f"Indexing complete. Found {self.total_windows} total windows in {len(self.file_metadata)} files.")
        LOG.info(f"Discovered {self.n_subjects} unique subjects.")
    
    def _prepare_groups_for_splitting(self):
        if self.total_windows == 0:
            self.groups = torch.empty(0, dtype=torch.long)
            return
        LOG.debug("Preparing group information for data splitting...")
        groups_list = np.zeros(self.total_windows, dtype=int)
        for info in self.file_metadata:
            start, end = info['start_index'], info['start_index'] + info['n_windows']
            groups_list[start:end] = info['group']
        self.groups = torch.tensor(groups_list, dtype=torch.long)
    
    def __len__(self):
        return self.total_windows
    
    def __getitem__(self, idx):
        if not (0 <= idx < self.total_windows):
            raise IndexError(f"Index {idx} is out of bounds for dataset with length {self.total_windows}.")
        file_info = self._find_file_for_idx(idx)
        continuous_data = self._load_h5_file_data(file_info['path'])
        downsampled_data = self._downsample(continuous_data)
        window_idx_in_file = idx - file_info['start_index']
        window_samples = int(self.window * self.sfreq)
        step_samples = int(window_samples * (1 - self.overlap))
        start = window_idx_in_file * step_samples
        end = start + window_samples
        trial = downsampled_data[:, start:end]
        scaled_trial = np.array([self.scaler(channel) for channel in trial])
        window_tensor = torch.from_numpy(scaled_trial[np.newaxis, :, :])
        label_tensor = torch.tensor(file_info['label'], dtype=torch.long)
        return window_tensor, label_tensor
    
    def _find_file_for_idx(self, idx):
        for info in self.file_metadata:
            if info['start_index'] <= idx < info['start_index'] + info['n_windows']:
                return info
        raise RuntimeError(f"Metadata not found for index {idx}.")
    
    def _get_data_paths(self):
        paths = []
        base_path = os.path.join(self.data_root, self.scenario.capitalize())
        if self.scenario == "intra":
            paths.append(os.path.join(base_path, self.mode))
        elif self.scenario == "cross":
            if self.mode == "train": paths.append(os.path.join(base_path, "train"))
            else: paths.append(os.path.join(base_path, self.mode))
        valid_paths = [p for p in paths if os.path.isdir(p)]
        if not valid_paths: LOG.warning(f"No data directories found for path: {base_path} and mode: {self.mode}")
        return valid_paths
    
    def _extract_info_from_filename(self, filename):
        base = os.path.splitext(filename)[0]
        parts = base.split('_')
        if len(parts) >= 3 and parts[-2].isdigit():
            return "_".join(parts[:-2]), parts[-2]
        return None, None
    
    def _load_h5_file_data(self, filepath):
        with h5py.File(filepath, 'r') as f:
            return f[list(f.keys())[0]][()].astype(np.float32)
    
    def _downsample(self, data):
        if self.sfreq >= self.original_sfreq: return data
        n_samples = int(data.shape[-1] * self.sfreq / self.original_sfreq)
        return resample(data, n_samples, axis=-1)
    
    def _get_or_create_subject_int_id(self, subject_str_id):
        if subject_str_id not in self._subject_str_to_int_map:
            new_id = self._next_subject_int_id
            self._subject_str_to_int_map[subject_str_id] = new_id
            self._next_subject_int_id += 1
        return self._subject_str_to_int_map[subject_str_id]
    
    @property
    def n_subjects(self):
        return self._next_subject_int_id
