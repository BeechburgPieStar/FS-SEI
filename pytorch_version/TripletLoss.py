import torch
import torch.nn as nn


class TripletLoss(nn.Module):
    def __init__(self, margin=5):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        n = inputs.size(0)

        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()  # a^2 + b^2
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)  # dist * 1 - 2 * input * input.t()
        # dist = dist - 2 * (inputs @ inputs.t())
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        mask = targets.expand(n, n).eq(targets.expand(n, n).t())
        dist_ap, dist_an = [], []
        for i in range(n):
            dist_ap.append(dist[i][mask[i]].max().unsqueeze(0))
            dist_an.append(dist[i][mask[i] == 0].min().unsqueeze(0))
        dist_ap = torch.cat(dist_ap)
        dist_an = torch.cat(dist_an)

        y = torch.ones_like(dist_an)
        loss = self.ranking_loss(dist_an, dist_ap, y)
        return loss