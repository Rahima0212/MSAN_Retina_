import torch
import torch.nn as nn

def clip_by_tensor(t, t_min, t_max):
    """
    Clips the values of a tensor element-wise between t_min and t_max.
    Equivalent to torch.clamp(t, t_min, t_max).
    """
    return torch.clamp(t, min=t_min, max=t_max)


class FocalLoss(nn.Module):
    """
    Proper multi-class Focal Loss for classification.
    Works when labels are class indices (0..num_classes-1).
    """

    def __init__(self, gamma=2, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, labels):
        """
        logits: [batch_size, num_classes]
        labels: [batch_size]  (class indices)
        """

        # Convert labels to one-hot
        num_classes = logits.size(1)
        labels_one_hot = torch.zeros_like(logits).scatter_(1, labels.unsqueeze(1), 1)

        # Compute probabilities
        probs = torch.softmax(logits, dim=1)

        # Clip probabilities to avoid log(0)
        probs_clipped = torch.clamp(probs, 1e-8, 1.0)

        # Focal loss components
        pt = torch.sum(probs_clipped * labels_one_hot, dim=1)  # probability of the true class
        focal_weight = (1 - pt) ** self.gamma

        loss = -self.alpha * focal_weight * torch.log(pt)

        return loss.mean()
