import os
import random  # Retained for potential use by parent class's random_state features
import logging
import h5py
from scipy.signal import resample
import numpy as np
import torch
# Subset is used by parent's split_data
from torch.utils.data import DataLoader, TensorDataset, Subset

# Correct import for ContinuousDataset from meegnet
try:
    from meegnet.dataloaders import ContinuousDataset
except ImportError:
    # This is a critical dependency. If not found, the module cannot function.
    logging.critical(
        "MEEGNet library not found or ContinuousDataset import failed. "
        "Please ensure MEEGNet is correctly installed: 'pip install meegnet'."
    )
    raise

LOG = logging.getLogger(__name__)
# Basic logging configuration if the user of this module hasn't configured logging yet.
if not LOG.hasHandlers():
    logging.basicConfig(level=logging.WARNING,
                        format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')


class CustomMEGDataset(ContinuousDataset):
    """
    Custom PyTorch Dataset for loading and preprocessing MEG data from H5 files
    for the INFOMDLR project. It segments continuous data into windows, applies
    scaling, and extracts labels (task types) and groups (subject identifiers)
    from filenames.

    Inherits from meegnet.dataloaders.ContinuousDataset to leverage its
    scaling setup and data splitting functionalities (like Leave-Subject-Out).
    """

    def __init__(self, data_root, scenario, mode, task_to_label_map,
                 sfreq=2034, orig_sfreq=2034, scaling="minmax", window=2, overlap=0.5, offset=0,
                 random_state=42, target_subject_id=None, lso_for_split=False, **kwargs):
        """
        Initializes the CustomMEGDataset.

        Parameters:
        ----------
        data_root : str
            Absolute or relative path to the root data directory which contains
            'Intra' and 'Cross' subfolders.
        scenario : str
            The classification scenario, either "intra" or "cross".
        mode : str
            Dataset mode: "train" or "test" for the "intra" scenario;
            "train", "test1", "test2", or "test3" for the "cross" scenario.
        task_to_label_map : dict
            A dictionary mapping task type strings (e.g., 'rest', 'task_motor')
            to integer labels (e.g., 0, 1).
        sfreq : float, optional
            Preferred sampling frequency of the MEG data in Hz. Default is 2034.
        orig_sfreq : float, optional
            Reference sampling frequency of the MEG data in Hz, used for
            downsampling if the provided `sfreq` is different.
        scaling : str, optional
            The type of scaling to apply to the data: "minmax" (scales to [0,1]),
            "zscore" (standard normalization), or None. Default is "minmax".
            This is handled by the parent ContinuousDataset's scaler.
        window : float, optional
            Duration of each window/segment in seconds. Default is 2.
        overlap : float, optional
            Fraction of overlap between consecutive windows (0.0 to <1.0).
            Default is 0.5 (50% overlap).
        offset : float, optional
            Time offset in seconds from the beginning of each recording before
            starting the windowing process. Default is 0.
        random_state : int, optional
            Seed for random number generation, used by parent class for
            reproducible data splitting. Default is 42.
        target_subject_id : str, optional
            For the "intra" scenario, specifies the string identifier of the
            subject to load. If None, data for all subjects found in the
            'Intra' paths will be loaded (with a warning, as 'Intra' is
            expected to be for a single subject). Default is None.
        lso_for_split : bool, optional
            Sets the default Leave-Subject-Out (LSO) behavior for the
            `split_data` method inherited from the parent class.
            Default is False. Should typically be True for cross-subject validation.
        kwargs : dict
            Additional keyword arguments passed to the parent
            `meegnet.dataloaders.ContinuousDataset` constructor.
        """
        self.data_root = data_root
        self.scenario = scenario.lower()
        self.mode = mode.lower()
        self.task_to_label_map = task_to_label_map
        self.target_subject_id_str = target_subject_id
        # Original sampling frequency, used for downsampling if needed
        self.orginal_sfreq = orig_sfreq

        # Initialize parent ContinuousDataset.
        # This sets up self.sfreq, self.scaling (and self.scaler), self.window,
        # self.overlap, self.offset, self.random_state, and self.lso.
        super().__init__(
            sfreq=sfreq,
            scaling=scaling,
            window=window,
            overlap=overlap,
            offset=offset,
            random_state=random_state,
            lso=lso_for_split,
            split_sizes=(0.9, 0.1, 0),  # Test data is already provided
            **kwargs
        )

        # Store string names of target labels for reference (e.g., for plotting confusion matrices)
        self.string_target_labels = sorted(
            task_to_label_map.keys(), key=lambda k: task_to_label_map[k])

        # Determine paths to the actual data files
        self.data_paths_to_load = self._get_data_paths()

        # Mappings for subject identifiers
        self._subject_str_to_int_map = {}
        self._int_to_subject_str_map = {}
        self._next_subject_int_id = 0

        # Initialize data-holding attributes, to be populated by load_all_data
        self.data = torch.empty(0)
        self.targets = torch.empty(0, dtype=torch.long)
        # Integer subject IDs for splitting
        self.groups = torch.empty(0, dtype=torch.long)

        # Load and preprocess data
        self.load_all_data()

    def _get_data_paths(self):
        """
        Determines and validates the list of directory paths to load data from,
        based on the specified scenario and mode.
        """
        paths = []
        # Construct path based on scenario and mode
        if self.scenario == "intra":
            if self.mode == "train":
                paths.append(os.path.join(self.data_root, "Intra", "train"))
            elif self.mode == "test":
                paths.append(os.path.join(self.data_root, "Intra", "test"))
            else:
                # Critical configuration error
                raise ValueError(
                    f"Invalid mode '{self.mode}' for 'intra' scenario. Expected 'train' or 'test'.")
        elif self.scenario == "cross":
            if self.mode == "train":
                paths.append(os.path.join(self.data_root, "Cross", "train"))
            elif self.mode in ["test1", "test2", "test3"]:
                paths.append(os.path.join(self.data_root, "Cross", self.mode))
            else:
                # Critical configuration error
                raise ValueError(
                    f"Invalid mode '{self.mode}' for 'cross' scenario. Expected 'train', 'test1', 'test2', or 'test3'.")
        else:
            # Critical configuration error
            raise ValueError(
                f"Invalid scenario: {self.scenario}. Expected 'intra' or 'cross'.")

        # Validate that determined paths actually exist
        valid_paths = []
        for p in paths:
            if not os.path.isdir(p):
                LOG.warning(
                    f"Data directory not found: {p}. This path will be skipped.")
            else:
                valid_paths.append(p)
        return valid_paths

    def _split_window(self, sub_data_np):
        """
        Segments continuous raw data into windows and applies scaling.
        This method uses parameters (sfreq, window, overlap, offset, scaler)
        initialized in the parent ContinuousDataset.

        Input:
        ------
        sub_data_np : numpy.ndarray
            Raw continuous data with shape (channels, timepoints).

        Output:
        -------
        torch.Tensor
            Windowed data with shape (num_windows, 1, channels, window_len),
            or an empty tensor if no valid windows can be created.
            The added dimension of size 1 can be used as an input "channel" for
            some neural network architectures (e.g., 2D CNNs treating MEG channels
            and time as spatial dimensions).
        """
        # Basic input validation for shape
        if sub_data_np.ndim == 1:
            LOG.warning(
                f"Input to _split_window is 1D (shape: {sub_data_np.shape}). Assuming single channel data.")
            sub_data_np = sub_data_np.reshape(1, -1)
        elif sub_data_np.ndim > 2:
            LOG.warning(
                f"Input to _split_window has >2 dimensions (shape: {sub_data_np.shape}). Expected (channels, timepoints). Taking first 2D slice.")
            # Assuming the first slice is the relevant (channels, timepoints) data
            sub_data_np = sub_data_np[0]

        if sub_data_np.ndim != 2:
            LOG.error(
                f"Input data for windowing must be 2D (channels, timepoints), but got shape {sub_data_np.shape}. Cannot proceed.")
            # Return empty tensor if shape is unrecoverable
            return torch.empty(0)

        num_channels, total_timepoints = sub_data_np.shape

        # Calculate window parameters in samples
        window_samples = int(self.window * self.sfreq)
        # Step size based on overlap
        step_samples = int(window_samples * (1 - self.overlap))
        offset_samples = int(self.offset * self.sfreq)

        # Validate calculated window parameters
        if window_samples <= 0 or step_samples <= 0:
            LOG.error(f"Window length ({window_samples} samples) or step size ({step_samples} samples) is non-positive. "
                      f"Check sfreq ({self.sfreq}), window ({self.window}), and overlap ({self.overlap}) parameters.")
            return torch.empty(0)

        # Check if the recording is long enough for at least one window
        if total_timepoints < window_samples + offset_samples:
            LOG.warning(f"Total timepoints ({total_timepoints}) in recording is too short for the specified "
                        f"window_samples ({window_samples}) + offset_samples ({offset_samples}). No windows will be generated from this file.")
            return torch.empty(0)

        windows = []
        # Iterate through the data to extract windows
        for i in range(offset_samples, total_timepoints - window_samples + 1, step_samples):
            # Extract a window: (channels, window_samples)
            trial = sub_data_np[:, i: i + window_samples]

            # Skip trial if it contains NaN values
            if np.isnan(trial).any():
                # Debug level, can be noisy
                LOG.debug("NaNs found in a trial segment; skipping this trial.")
                continue

            # Apply scaling (self.scaler is initialized by parent ContinuousDataset, e.g., zscore or minmax)
            # The scaler typically operates on the last axis (time).
            # Shape: (channels, window_samples)
            scaled_trial = np.array([self.scaler(channel_data)
                                    for channel_data in trial])

            # Add a new axis to represent an input "channel" for the window itself,
            # making the shape (1, channels, window_samples).
            windows.append(scaled_trial[np.newaxis, :, :])

        if not windows:
            # This can happen if all trials had NaNs or other issues.
            return torch.empty(0)

        # Stack all collected windows and convert to a PyTorch tensor
        # Shape: (num_windows, 1, channels, window_samples)
        return torch.from_numpy(np.array(windows)).float()

    def _get_h5_dataset_name(self, filepath):
        """
        Constructs the H5 dataset name from the filepath, following the project's convention.
        Example: ".../data/Intra/train/rest_105923_1.h5" -> H5 dataset key "rest_105923".
        """
        filename_without_dir = os.path.basename(filepath)
        # Remove .h5 extension and the last part (chunk number, e.g., "_1")
        parts = os.path.splitext(filename_without_dir)[0].split('_')
        if len(parts) > 1:  # Ensure there's at least "name_chunk" structure
            # Join all parts except the last one (chunk number)
            dataset_name = "_".join(parts[:-1])
            return dataset_name
        else:
            # Fallback if filename format is unexpected (e.g., no underscore or chunk number)
            LOG.warning(f"Could not reliably determine H5 dataset name from filename structure: {filename_without_dir}. "
                        f"Using base name '{parts[0]}' as dataset key.")
            # Return the part before any potential extension/chunk
            return parts[0]

    def _extract_info_from_filename(self, filename):
        """
        Extracts task type (string) and subject identifier (string) from the H5 filename.
        Expected filename format: "taskType_subjectIdentifier_chunkNumber.h5".
        Handles task types that may themselves contain underscores (e.g., "task_story_math").
        """
        base_filename = os.path.splitext(filename)[0]  # Remove .h5 extension
        parts = base_filename.split('_')

        # Filename must have at least 3 parts: task, subjectID, chunkNumber
        if len(parts) < 3:
            LOG.warning(f"Filename '{filename}' has fewer than 3 parts separated by underscores. "
                        f"Expected 'taskType_subjectIdentifier_chunkNumber.h5' format. Attempting fallback.")
            # Fallback for "task_subject" if chunk number is missing and subject is numeric
            if len(parts) == 2 and parts[1].isdigit():
                LOG.debug(f"Assuming 'task_subject' format for {filename}.")
                return parts[0], parts[1]
            LOG.error(f"Cannot parse filename: {filename}")
            return None, None

        # Assume the last part is the chunk number (not used for labeling/grouping)
        # Assume the second to last part is the subject identifier if it's numeric
        subject_id_str = parts[-2]

        if subject_id_str.isdigit():
            # If subject ID is found, the remaining parts at the beginning form the task type
            task_type_str = "_".join(parts[:-2])
            if not task_type_str:  # Should not happen if len(parts) >= 3
                LOG.error(
                    f"Extracted empty task type from filename: {filename}")
                return None, None
            return task_type_str, subject_id_str
        else:
            # If the second-to-last part isn't a simple number, try a more robust search
            # for subject ID (e.g. if task name has variable underscores)
            LOG.debug(f"Potential subject ID part '{subject_id_str}' in '{filename}' is not purely numeric. "
                      f"Attempting more robust subject ID extraction.")
            subject_id_candidate = None
            task_type_parts_candidate = []
            # Iterate backwards from the part before the chunk number
            for i in range(len(parts) - 2, -1, -1):
                # Heuristic: subject ID is numeric, length >=5
                if parts[i].isdigit() and len(parts[i]) >= 5:
                    subject_id_candidate = parts[i]
                    # Parts before subject ID form the task
                    task_type_parts_candidate = parts[:i]
                    break

            if subject_id_candidate:
                subject_id_str = subject_id_candidate
                task_type_str = "_".join(task_type_parts_candidate)
                if task_type_str:
                    return task_type_str, subject_id_str

            LOG.error(
                f"Failed to robustly extract numeric subject ID from filename: {filename}. Parts: {parts}")
            return None, None

    def _load_h5_file_data(self, filepath):
        """Loads the data matrix from a single H5 file."""
        try:
            with h5py.File(filepath, 'r') as f:
                h5_dataset_name = self._get_h5_dataset_name(filepath)
                if h5_dataset_name not in f:
                    keys_in_file = list(f.keys())
                    if len(keys_in_file) == 1:
                        # If only one dataset exists, assume it's the correct one
                        h5_dataset_name = keys_in_file[0]
                        LOG.debug(
                            f"Derived H5 dataset name not found. Using single available key '{h5_dataset_name}' for H5 file {filepath}")
                    else:
                        LOG.error(
                            f"Dataset key '{h5_dataset_name}' not found in H5 file: {filepath}. Available keys: {keys_in_file}")
                        return None
                # Load data and ensure it's float32 for PyTorch
                return f[h5_dataset_name][()].astype(np.float32)
        except Exception as e:
            # Log error with filepath for easier debugging
            LOG.error(f"Error loading H5 file '{filepath}': {e}")
            return None

    def _downsample(self, data):
        if self.sfreq == self.orginal_sfreq:
            # No downsampling needed, or target frequency is not specified
            return data
        if self.sfreq > self.orginal_sfreq:
            LOG.warning(f"Target sampling frequency ({self.sfreq} Hz) is higher than original ({self.orginal_sfreq} Hz). "
                        "No upsampling implemented. Returning original data.")
            return data
        n_samples = int(data.shape[-1] * self.orginal_sfreq / self.sfreq)
        # data shape: (channels, time)
        return resample(data, n_samples, axis=-1)

    def _get_or_create_subject_int_id(self, subject_str_id):
        """Maps a string subject identifier to a unique integer ID for grouping."""
        if subject_str_id not in self._subject_str_to_int_map:
            new_id = self._next_subject_int_id
            self._subject_str_to_int_map[subject_str_id] = new_id
            # Store reverse mapping
            self._int_to_subject_str_map[new_id] = subject_str_id
            self._next_subject_int_id += 1
        return self._subject_str_to_int_map[subject_str_id]

    def load_all_data(self):
        """
        Main method to load all H5 files from the determined paths,
        extract information, preprocess data into windows, and populate
        the dataset's `data`, `targets`, and `groups` attributes.
        """
        all_windows_list, all_targets_list, all_groups_list = [], [], []

        if not self.data_paths_to_load:
            LOG.warning(
                f"No valid data paths found for scenario='{self.scenario}', mode='{self.mode}'. Dataset will be empty.")

        for data_dir in self.data_paths_to_load:
            # Keep this for clarity on which dir is processed
            LOG.info(f"Processing data directory: {data_dir}")
            try:
                # List all .h5 files in the current directory
                filenames = [f for f in os.listdir(
                    data_dir) if f.endswith(".h5")]
            except FileNotFoundError:
                LOG.error(
                    f"Data directory '{data_dir}' not found during os.listdir. Skipping.")
                continue

            if not filenames:
                LOG.warning(f"No .h5 files found in directory: {data_dir}")
                continue

            for filename in filenames:
                filepath = os.path.join(data_dir, filename)
                task_type_str, subject_id_str = self._extract_info_from_filename(
                    filename)

                # Validate parsing results
                if not (task_type_str and subject_id_str):
                    LOG.warning(
                        f"Could not parse info from filename '{filepath}'. Skipping.")
                    continue

                # Apply subject filter for 'intra' scenario
                if self.scenario == "intra" and self.target_subject_id_str and subject_id_str != self.target_subject_id_str:
                    continue

                # Check if task type is recognized
                if task_type_str not in self.task_to_label_map:
                    LOG.warning(
                        f"Unrecognized task type '{task_type_str}' in filename '{filepath}'. Skipping.")
                    continue

                target_label_int = self.task_to_label_map[task_type_str]
                subject_id_int = self._get_or_create_subject_int_id(
                    subject_id_str)

                # Load raw data from H5 file
                continuous_data_np = self._load_h5_file_data(filepath)
                if continuous_data_np is None:
                    continue  # Error already logged by _load_h5_file_data

                # Optional downsampling to reduce training time
                continuous_data_np = self._downsample(continuous_data_np)
                # Segment into windows and apply scaling
                windowed_data_torch = self._split_window(continuous_data_np)

                if windowed_data_torch is None or len(windowed_data_torch) == 0:
                    LOG.warning(
                        f"No valid windows generated from file '{filepath}'. Skipping.")
                    continue

                # Append processed data and corresponding labels/groups
                all_windows_list.append(windowed_data_torch)
                num_windows_from_file = len(windowed_data_torch)
                all_targets_list.extend(
                    [target_label_int] * num_windows_from_file)
                all_groups_list.extend(
                    [subject_id_int] * num_windows_from_file)

        # Consolidate all loaded data
        if all_windows_list:
            self.data = torch.cat(all_windows_list, dim=0)
            self.targets = torch.tensor(all_targets_list, dtype=torch.long)
            self.groups = torch.tensor(all_groups_list, dtype=torch.long)
        else:
            # If no data was loaded (e.g., all files failed or paths were empty)
            LOG.warning(
                f"No data successfully loaded for scenario='{self.scenario}', mode='{self.mode}'. Dataset is empty.")
            # Initialize with correct (empty) shapes to prevent downstream errors
            num_window_samples = int(
                self.sfreq * self.window) if self.sfreq and self.window > 0 else 1
            num_channels = 248  # Default number of channels as per project description
            self.data = torch.empty(0, 1, num_channels, num_window_samples)
            self.targets = torch.empty(0, dtype=torch.long)
            self.groups = torch.empty(0, dtype=torch.long)

        # Update subject list and count for compatibility with parent's split_data
        self.subject_list = sorted(list(self._subject_str_to_int_map.values()))
        self.n_subjects = len(self.subject_list)

        # Sanity check for 'intra' scenario if no specific subject was targeted
        if self.scenario == "intra" and self.target_subject_id_str is None and self.n_subjects > 1:
            LOG.warning(f"Intra-subject scenario: Loaded data for {self.n_subjects} subjects "
                        f"({list(self._subject_str_to_int_map.keys())}) because 'target_subject_id' was not specified. "
                        "The project description implies the 'Intra' folder is for a single subject.")

        LOG.info(f"Data loading complete for scenario='{self.scenario}', mode='{self.mode}'. "
                 f"Total windows: {len(self.data)}, Unique subjects: {self.n_subjects}. "
                 f"Subject map: {self._subject_str_to_int_map}")
        if len(self.data) > 0:
            LOG.info(f"Final data tensor shape: {self.data.shape}")

    def __len__(self):
        """Returns the total number of samples (windows) in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a single sample (a window of MEG data and its corresponding target label)
        from the dataset, as required by PyTorch DataLoader.
        """
        # Basic type check for index, though DataLoader usually handles this.
        if not isinstance(idx, int):
            if isinstance(idx, (list, torch.Tensor)) and len(idx) == 1:
                idx = idx[0].item() if isinstance(
                    idx, torch.Tensor) else idx[0]  # Convert to int
            else:
                # This case should ideally not be hit with standard DataLoader usage.
                raise TypeError(
                    f"Index must be int or a single-element list/tensor, not {type(idx)}")

        # Boundary check for the index
        if not (0 <= idx < len(self.data)):
            LOG.error(
                f"Index {idx} out of bounds for dataset of length {len(self.data)}")
            raise IndexError("Dataset index out of range.")

        sample_data = self.data[idx]
        sample_target = self.targets[idx]
        # To return group ID as well (e.g., for custom batching or analysis):
        # sample_group = self.groups[idx]
        # return sample_data, sample_target, sample_group
        return sample_data, sample_target
