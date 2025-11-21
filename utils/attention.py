import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch import nn

__all__ = ['PAM_Module', 'CAM_Module', 'semanticModule']


# --- Helper Blocks for Semantic Module (U-Net style) ---

class _EncoderBlock(Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


# --- Semantic Module (semanticModule) ---

class semanticModule(Module):
    """ Semantic attention module (Simple U-Net for feature compression/expansion)"""
    def __init__(self, in_dim):
        super(semanticModule, self).__init__()
        self.chanel_in = in_dim

        self.enc1 = _EncoderBlock(in_dim, in_dim*2)
        self.enc2 = _EncoderBlock(in_dim*2, in_dim*4)
        self.dec2 = _DecoderBlock(in_dim * 4, in_dim * 2, in_dim * 2)
        self.dec1 = _DecoderBlock(in_dim * 2, in_dim, in_dim )

    def forward(self,x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec2 = self.dec2( enc2)
        
        # Note: F.upsample is deprecated. Using F.interpolate which is recommended.
        dec1 = self.dec1( F.interpolate(dec2, size=enc1.size()[2:], mode='bilinear', align_corners=True))

        # The original code had enc2.view(-1) which flattens the output for a large batch
        # Assuming intended output is the feature vector and the upsampled map.
        return enc2.view(enc2.size(0), -1), dec1


# --- Position Attention Module (PAM_Module) ---

class PAM_Module(Module):
    """ Position attention module (Non-local block, spatial attention)"""
    # Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # Queries, Keys, Values (typically reduce C by 8 for efficiency)
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        
        # Learnable parameter to scale the output attention map
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)
        
    def forward(self, v, k, q):
        """
        Inputs:
            v : Value input feature maps (B X C X H X W)
            k : Key input feature maps (B X C X H X W)
            q : Query input feature maps (B X C X H X W)
        Returns:
            out : attention value + input feature
        """
        m_batchsize, C, height, width = v.size()
        
        # 1. Project Query and Key features, then reshape to (B, N, C') where N=H*W
        # Query: (B, C/8, H, W) -> (B, N, C/8)
        proj_query = self.query_conv(q).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        # Key: (B, C/8, H, W) -> (B, C/8, N)
        proj_key = self.key_conv(k).view(m_batchsize, -1, width*height)

        # 2. Attention Map: energy = Q * K
        # energy: (B, N, C/8) * (B, C/8, N) -> (B, N, N) (N=H*W)
        energy = torch.bmm(proj_query, proj_key) # batch matmul
        attention = self.softmax(energy) # softmax along the last dim (rows)
        
        # 3. Project Value feature and reshape to (B, C, N)
        # Value: (B, C, H, W) -> (B, C, N)
        proj_value = self.value_conv(v).view(m_batchsize, -1, width*height)

        # 4. Output: out = V * Attention_T
        # out: (B, C, N) * (B, N, N) -> (B, C, N)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # Transpose attention for V*A
        out = out.view(m_batchsize, C, height, width)

        # 5. Residual Connection
        out = self.gamma*out + v
        return out


# --- Channel Attention Module (CAM_Module) ---

class CAM_Module(Module):
    """ Channel attention module (Non-local block, channel attention)"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = Parameter(torch.zeros(1))
        self.softmax = Softmax(dim=-1)
        
    def forward(self,v, k, q):
        """
        Inputs:
            v : Value input feature maps (B X C X H X W)
            k : Key input feature maps (B X C X H X W)
            q : Query input feature maps (B X C X H X W)
        Returns:
            out : attention value + input feature
        """
        m_batchsize, C, height, width = v.size()
        
        # 1. Reshape Query, Key, and Value to (B, C, N) where N=H*W
        # Query: (B, C, N)
        proj_query = q.view(m_batchsize, C, -1)
        # Key: (B, C, N) -> (B, N, C)
        proj_key = k.view(m_batchsize, C, -1).permute(0, 2, 1)
        
        # 2. Attention Map: energy = Q * K_T
        # energy: (B, C, N) * (B, N, C) -> (B, C, C)
        energy = torch.bmm(proj_query, proj_key)
        
        # Apply the proposed channel attention trick: E' = max(E) - E
        # This is a key modification for CAM to ensure stability/focus.
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        
        # 3. Reshape Value to (B, C, N)
        proj_value = v.view(m_batchsize, C, -1)

        # 4. Output: out = Attention * V
        # out: (B, C, C) * (B, C, N) -> (B, C, N)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        # 5. Residual Connection
        out = self.gamma*out + v
        return out

if __name__ == '__main__':
    # Example usage (as in the original snippet)
    x = torch.rand(2,8,4,4)
    # ROI is defined but not used in the attention module test
    # ROI = torch.rand(2,1,4,4) 
    
    # Test PAM
    pam = PAM_Module(x.shape[1])
    pam_out = pam(x,x,x)
    print(f"PAM Output Shape: {pam_out.shape}")
    
    # Test CAM
    cam = CAM_Module(x.shape[1])
    cam_out = cam(x,x,x)
    print(f"CAM Output Shape: {cam_out.shape}")
    
    # Test semanticModule
    sem = semanticModule(x.shape[1])
    sem_feat, sem_map = sem(x)
    print(f"Semantic Feature Shape: {sem_feat.shape}")
    print(f"Semantic Map Shape: {sem_map.shape}")
    
    # Check if outputs are identical (they shouldn't be, they implement different attention mechanisms)
    # The original check: print(pam_out==cam_out) is misleading as they are different modules.
    # print(pam_out==cam_out) 
    print("PAM_Module and CAM_Module implement different attention mechanisms and are not expected to be equal.")