import torch
import torch.nn.functional as F

class ContrastiveLossWithMargin(torch.nn.Module):
    def __init__(self, margin=2.0):
        super(ContrastiveLossWithMargin, self).__init__()
        self.margin = margin

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        loss_contrastive = torch.mean(torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive, torch.mean(torch.pow(euclidean_distance,2))

class EuclideanLoss(torch.nn.Module):
    def __init__(self):
        super(EuclideanLoss, self).__init__()

    def forward(self, output1, output2):
        euclidean_distance = F.pairwise_distance(output1, output2, keepdim = True)
        euclidean_loss = torch.mean(torch.pow(euclidean_distance, 2))
        return euclidean_loss