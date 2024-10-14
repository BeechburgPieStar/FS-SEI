import torch
import torch.nn.functional as F

margin_config = 5


def _pairwise_distances(embeddings, squared=False):
    """Compute the 2D matrix of distances between all the embeddings."""
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)
    distances = square_norm.unsqueeze(1) - 2.0 * dot_product + square_norm.unsqueeze(0)
    distances = torch.max(distances, torch.tensor(0.0))
    if not squared:
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def _get_triplet_mask(labels):
    """Return a 3D mask where mask[a, p, n] is True if the triplet (a, p, n) is valid."""
    indices_equal = torch.eye(labels.size(0), device=labels.device).bool()
    indices_not_equal = ~indices_equal
    i_not_equal_j = indices_not_equal.unsqueeze(2)
    i_not_equal_k = indices_not_equal.unsqueeze(1)
    j_not_equal_k = indices_not_equal.unsqueeze(0)
    distinct_indices = i_not_equal_j & i_not_equal_k & j_not_equal_k

    label_equal = (labels.unsqueeze(0) == labels.unsqueeze(1)).bool()
    i_equal_j = label_equal.unsqueeze(2)
    i_equal_k = label_equal.unsqueeze(1)
    valid_labels = i_equal_j & ~i_equal_k

    mask = distinct_indices & valid_labels
    return mask


def batch_all_triplet_loss(labels, embeddings, margin=margin_config, squared=False):
    """Build the triplet loss over a batch of embeddings."""
    # Get the pairwise distance matrix
    pairwise_dist = _pairwise_distances(embeddings, squared=squared)
    # Compute a 3D tensor of size (batch_size, batch_size, batch_size)
    anchor_positive_dist = pairwise_dist.unsqueeze(2)
    anchor_negative_dist = pairwise_dist.unsqueeze(1)
    triplet_loss = anchor_positive_dist - anchor_negative_dist + margin
    # Put to zero the invalid triplets
    mask = _get_triplet_mask(labels)
    triplet_loss = triplet_loss * mask.float()
    # Remove negative losses (i.e. the easy triplets)
    triplet_loss = torch.max(triplet_loss, torch.tensor(0.0))
    # Count number of positive triplets (where triplet_loss > 0)
    valid_triplets = (triplet_loss > 1e-16).float()
    num_positive_triplets = valid_triplets.sum()
    num_valid_triplets = mask.sum()
    fraction_positive_triplets = num_positive_triplets / (num_valid_triplets + 1e-16)
    # Get final mean triplet loss over the positive valid triplets
    triplet_loss = triplet_loss.sum() / (num_positive_triplets + 1e-16)
    return triplet_loss


class TripletLoss(torch.nn.Module):
    def __init__(self, margin=margin_config, squared=False):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared

    def forward(self, embeddings, labels):
        return batch_all_triplet_loss(labels, embeddings, margin=self.margin, squared=self.squared)
