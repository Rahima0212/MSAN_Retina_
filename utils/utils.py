import torch
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, classes, title='Confusion Matrix', save_path=None):
    """
    Plot confusion matrix using seaborn's heatmap.
    
    Args:
        y_true (array-like): True labels
        y_pred (array-like): Predicted labels
        classes (list): List of class names
        title (str): Title of the plot
        save_path (str): Path to save the plot. If None, the plot will be displayed
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, cmap='Blues', fmt='.2f',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def count_param(model):
    """
    Counts the total number of trainable parameters in a PyTorch model.

    Args:
        model (nn.Module): The PyTorch model instance.

    Returns:
        int: The total count of parameters.
    """
    param_count = 0
    # Iterate over all parameters in the model
    for param in model.parameters():
        # param.view(-1) flattens the tensor, and .size()[0] gets its size (number of elements)
        param_count += param.view(-1).size()[0]
    return param_count

def load_state_dict(model_dir, is_multi_gpu):
    """
    Loads a model's state dictionary from a file and handles multi-GPU prefixing.

    Args:
        model_dir (str): Path to the saved model checkpoint file.
        is_multi_gpu (bool): True if the saved model was wrapped in 
                             nn.DataParallel or nn.DistributedDataParallel (i.e., keys 
                             start with 'module.').

    Returns:
        OrderedDict: The cleaned state dictionary ready to be loaded into a model.
    """
    # Load the state dictionary, mapping it to CPU storage (safe for any environment)
    # Assumes the checkpoint is saved as a dictionary with a 'state_dict' key
    checkpoint = torch.load(model_dir, map_location=lambda storage, loc: storage)
    state_dict = checkpoint.get('state_dict', checkpoint) # Handle case where state_dict is the root
    
    if is_multi_gpu:
        # If saved from DataParallel, keys will be prefixed with 'module.'
        # We need to remove this prefix to load into a single-GPU model
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith('module.'):
                name = k[7:]  # remove `module.`
            else:
                name = k
            new_state_dict[name] = v
        return new_state_dict
    else:
        # If not multi-GPU, return the state dictionary as is
        return state_dict