from torch import nn
import torch
from TripletLoss import TripletLoss as TripletLoss1
from TripletLoss2 import TripletLoss as TripletLoss2
from center_loss import CenterLoss


class STCLoss(nn.Module):
    """
    STC Loss = weight0 * CrossEntropyLoss + weight1 * TripletLoss + weight2 * CenterLoss
    weight0, weight1, weight2 are 1.0 0.01 0.01 by default
    """

    def __init__(self, weights=(1.0, 0.01, 0.01), triplet_margin=5, num_classes=10, feat_dim=2, device="cuda", triplet_loss_type=1):
        device = torch.device(device)
        super(STCLoss, self).__init__()
        TripletLoss = TripletLoss1 if triplet_loss_type == 1 else TripletLoss2
        self.weights = weights
        self.ce_loss = nn.CrossEntropyLoss().to(device)
        self.triplet_loss = TripletLoss(margin=triplet_margin).to(device)
        self.center_loss = CenterLoss(num_classes=num_classes, feat_dim=feat_dim, use_gpu=(device.type == "cuda")).to(device)

    def forward(self, x, f, y):
        ce_loss = self.ce_loss(x, y)
        triplet_loss = self.triplet_loss(f, y)
        center_loss = self.center_loss(f, y)
        loss = self.weights[0] * ce_loss + self.weights[1] * triplet_loss + self.weights[2] * center_loss
        return loss, ce_loss, triplet_loss, center_loss
