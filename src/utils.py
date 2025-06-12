import numpy as np
import re
from itertools import combinations
import torch

def string_to_int(string_array, mapping=None):
    """
    Converts an array of strings to an array of integers.

    Args:
        string_array (list or np.ndarray): The array of strings to convert.
        mapping (dict, optional): An existing mapping from string to int. 
                                  If None, a new one is created.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The array of integers.
            - dict: The mapping used for the conversion.
    """
    if mapping is None:
        unique_strings = sorted(list(set(string_array)))
        mapping = {s: i for i, s in enumerate(unique_strings)}
    
    int_array = np.array([mapping[s] for s in string_array])
    return int_array, mapping

def strip_string(s):
    """Removes leading/trailing whitespace and special characters from a string."""
    return re.sub(r'[^A-Za-z0-9_]+', '', s)

def stratified_sampling(data, targets, n_samples_per_class, subject_id, groups):
    """
    Performs stratified sampling for a given subject.

    Args:
        data (torch.Tensor): The full dataset.
        targets (torch.Tensor): The labels for the full dataset.
        n_samples_per_class (int): The number of samples to draw from each class.
        subject_id (int): The integer ID of the subject to sample from.
        groups (torch.Tensor): The group/subject ID for each sample.

    Returns:
        np.ndarray: The indices of the sampled data points.
    """
    subject_indices = np.where(groups == subject_id)[0]
    subject_targets = targets[subject_indices]
    
    sampled_indices = []
    unique_classes = torch.unique(subject_targets)
    
    for class_label in unique_classes:
        class_indices_for_subject = np.where(subject_targets == class_label)[0]
        
        # Ensure we don't sample more than available
        n_to_sample = min(n_samples_per_class, len(class_indices_for_subject))
        
        # Randomly choose indices
        chosen_local_indices = np.random.choice(
            class_indices_for_subject, n_to_sample, replace=False
        )
        
        # Map local indices back to global indices
        sampled_indices.extend(subject_indices[chosen_local_indices])
        
    return np.array(sampled_indices)

def random_pairs_of_minibatches(minibatches):
    """
    Generates random pairs of minibatches for algorithms like Mixup.
    """
    # Create all unique pairs of environments
    pairs = list(combinations(range(len(minibatches)), 2))
    
    # If we have an odd number of environments, we might need to pair one with itself
    # or handle it differently, but for now, we shuffle and return unique pairs.
    np.random.shuffle(pairs)

    # Return list of (minibatch1, minibatch2)
    return [(minibatches[i], minibatches[j]) for i, j in pairs]
