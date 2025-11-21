import torch
import torch.nn as nn
from model.MSA_subnet import resnet18_with_position_attention
from model.RGA_subnet import ROI_guided_OCT

class MSAN(nn.Module):
    def __init__(self, num_classes=6): # CHANGED to 6
        super(MSAN, self).__init__()
        
        # 1. Instantiate the two branches
        self.fundus_branch = resnet18_with_position_attention(num_classes=num_classes)
        self.oct_branch = ROI_guided_OCT(num_classes=num_classes)
        
        # 2. Define the fusion classifier
        # The feature vector from each branch has size 512
        fundus_feature_dim = 512
        oct_feature_dim = 512
        
        self.fusion_classifier = nn.Sequential(
            nn.Linear(fundus_feature_dim + oct_feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, fundus_img, oct_img, roi_mask):
        # Get logits and features from each branch
        fundus_logits, fundus_features = self.fundus_branch(fundus_img)
        oct_logits, oct_features = self.oct_branch(oct_img, roi_mask)
        
        # Concatenate the features from both branches
        fused_features = torch.cat((fundus_features, oct_features), dim=1)
        
        # Get the final prediction from the fused features
        fusion_logits = self.fusion_classifier(fused_features)
        
        # Return all three sets of logits for calculating combined loss
        return fusion_logits, fundus_logits, oct_logits