
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights # <--- ADD THIS IMPORT


class ROI_guided_OCT(nn.Module):
    def __init__(self, num_classes=6): # CHANGED to take num_classes
        super(ROI_guided_OCT, self).__init__()

        # 1. Load the pre-trained ResNet18 model
        base_model = resnet18(weights=ResNet18_Weights.DEFAULT)

        # 2. Copy all layers EXCEPT the first conv and final fc
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4
        self.avgpool = base_model.avgpool

        # 3. Create a NEW final layer for your number of classes
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
        # 4. Create a NEW first conv layer for our 4-channel input
        # We copy the weights from the old 3-channel layer and add a new channel initialized to zero
        old_weights = base_model.conv1.weight
        self.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight[:, :3, :, :] = old_weights # Copy weights for RGB channels
            self.conv1.weight[:, 3, :, :] = 0 # Initialize weights for the mask channel to zero

    def forward(self, oct_img, roi_mask):
        # (The forward pass remains exactly the same)
        x = torch.cat([oct_img, roi_mask], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        return logits, features