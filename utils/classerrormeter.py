import numpy as np
import torch
import numbers
from torchnet.meter import meter
from sklearn import metrics as mt
import matplotlib.pyplot as plt

class aucmeter(meter.Meter):
    """
    The AUCMeter measures the area under the receiver-operating characteristic
    (ROC) curve for binary classification problems. The area under the curve (AUC)
    can be interpreted as the probability that, given a randomly selected positive
    example and a randomly selected negative example, the positive example is
    assigned a higher score by the classification model than the negative example.

    The AUCMeter is designed to operate on one-dimensional Tensors `output`
    and `target`, where (1) the `output` contains model output scores that ought to
    be higher when the model is more convinced that the example should be positively
    labeled, and smaller when the model believes the example should be negatively
    labeled (for instance, the output of a signoid function); and (2) the `target`
    contains only values 0 (for negative examples) and 1 (for positive examples).
    """

    def __init__(self):
        super(aucmeter, self).__init__()
        self.reset()

    def draw_roc(self, fpr, tpr, auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def reset(self):
        # Initialize scores and targets as empty numpy arrays to be stacked later
        # When first initialized, they are 0-dimensional numpy arrays created from empty torch tensors.
        self.scores = torch.DoubleTensor(torch.DoubleStorage()).numpy()
        self.targets = torch.LongTensor(torch.LongStorage()).numpy()

    def add(self, output, target):
        # Conversion from torch tensors to numpy arrays (assuming this happens outside the commented block)
        if torch.is_tensor(output):
            # Ensure output is a numpy array for append/stacking
            output = output.cpu().detach().squeeze().numpy()
        if torch.is_tensor(target):
            # Ensure target is a numpy array for append/stacking
            target = target.cpu().detach().squeeze().numpy()
        elif isinstance(target, numbers.Number):
            target = np.asarray([target])
            
        # Ensure target is binary (0 or 1)
        assert np.all(np.add(np.equal(target, 1), np.equal(target, 0))), \
             'targets should be binary (0, 1)'
        
        # Ensure outputs and targets are at least 1D (or 2D if multi-label)
        if np.ndim(output) == 0:
            output = np.expand_dims(output, axis=0)
        if np.ndim(target) == 0:
            target = np.expand_dims(target, axis=0)
            
        # Handle 1D vs 2D (batch size x num_labels)
        if np.ndim(output) == 1:
            # If 1D, make it (BatchSize, 1) for consistent stacking
            output = np.expand_dims(output, axis=1)
            target = np.expand_dims(target, axis=1)

        if self.scores.shape[0] == 0:
            self.scores = output
            self.targets = target
        else:
            # Concatenate current batch data to accumulated data
            self.scores = np.append(self.scores, output, axis=0)
            self.targets = np.append(self.targets, target, axis=0)

    def value(self):
        # case when number of elements added are 0
        if self.scores.shape[0] == 0:
            return 0.5, 0.5, 0.5, 0.5, 0.5, np.array([0, 1]), np.array([0, 1])

        # Initialize lists for multi-label metrics
        accuracy, precision, recall, f1_score, area = [], [], [], [], []
        
        # Determine predictions based on a 0.5 threshold
        prediction = self.scores > 0.5
        prediction = prediction.astype('uint8')
        
        # Calculate metrics for each label (column)
        for i in range(self.targets.shape[1]):
            # Accuracy (Balanced Accuracy for binary classification)
            accuracy1 = mt.balanced_accuracy_score(self.targets[:, i], prediction[:, i])
            accuracy.append(accuracy1)
            
            # Precision, Recall, F1-score
            # We take the metrics for the positive class (index 1)
            precision1, recall1, f1_score1, supports1 = mt.precision_recall_fscore_support(
                self.targets[:, i], prediction[:, i], average=None, labels=[0, 1])
            
            # AUC
            area.append(mt.roc_auc_score(self.targets[:, i], self.scores[:, i]))
            
            # Append the positive class (index 1) metrics
            precision.append(precision1[1])
            recall.append(recall1[1])
            f1_score.append(f1_score1[1])

        # --- Calculate Global ROC Curve (Assuming flattening for a single global curve) ---
        
        # Flatten scores and targets to 1D arrays for the global ROC calculation
        # Note: This is usually only meaningful for a single-label problem or
        # when treating all (example, label) pairs equally.
        flat_scores = self.scores.reshape(-1)
        flat_targets = self.targets.reshape(-1)
        
        # Sort scores and get sorting indices
        scores, sortind = torch.sort(torch.from_numpy(
            flat_scores), dim=0, descending=True)
        scores = scores.numpy()
        sortind = sortind.numpy()

        # Create the ROC curve using sorted indices
        # tpr and fpr arrays are size N+1 where N is the total number of flattened samples
        tpr_count = np.zeros(shape=(scores.size + 1), dtype=np.float64)
        fpr_count = np.zeros(shape=(scores.size + 1), dtype=np.float64)

        for i in range(1, scores.size + 1):
            target_val = flat_targets[sortind[i - 1]]
            tpr_count[i] = tpr_count[i - 1] + (1 if target_val == 1 else 0)
            fpr_count[i] = fpr_count[i - 1] + (1 if target_val == 0 else 0)

        # Normalize TPR and FPR
        P = flat_targets.sum() * 1.0 # Total positive samples
        N = (flat_targets == 0).sum() * 1.0 # Total negative samples
        
        # Handle cases with zero positives/negatives to avoid division by zero
        tpr = tpr_count / P if P > 0 else np.zeros_like(tpr_count)
        fpr = fpr_count / N if N > 0 else np.zeros_like(fpr_count)
        
        # --- Final Averaging and Return ---

        # Average all per-label metrics
        mean_area = np.mean(area)
        mean_accuracy = np.mean(accuracy)
        mean_precision = np.mean(precision)
        mean_recall = np.mean(recall)
        mean_f1_score = np.mean(f1_score)

        # self.draw_roc(fpr, tpr, mean_area) # Commented out ROC drawing

        return mean_accuracy, mean_precision, mean_recall, mean_f1_score, mean_area, tpr, fpr