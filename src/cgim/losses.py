import torch
import torch.nn as nn
import torch.nn.functional as nnf


def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    full_batch_size = embeddings.size(0)
    embeddings = nn.functional.normalize(embeddings, dim=1)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    labels = labels.view(-1, 1)
    positive_mask = (labels == labels.T).float()
    self_mask = torch.eye(full_batch_size, device=embeddings.device)
    positive_mask = positive_mask - self_mask
    negative_mask = 1.0 - positive_mask - self_mask
    exp_sim = torch.exp(sim_matrix)
    pos_sum = torch.sum(exp_sim * positive_mask, dim=1)
    total_sum = torch.sum(exp_sim * (1.0 - self_mask), dim=1)
    loss = -torch.log(pos_sum / (total_sum + 1e-9))
    loss = torch.mean(loss)
    return loss


def simclr_loss(embeddings1, embeddings2, temperature=0.07):
    batch_size = embeddings1.size(0)
    embeddings1 = nn.functional.normalize(embeddings1, dim=1)
    embeddings2 = nn.functional.normalize(embeddings2, dim=1)
    embeddings = torch.cat([embeddings1, embeddings2], dim=0)
    sim_matrix = torch.matmul(embeddings, embeddings.T) / temperature
    labels = torch.arange(batch_size, device=embeddings.device)
    labels = torch.cat([labels + batch_size, labels])
    mask = torch.eye(2 * batch_size, device=embeddings.device).bool()
    sim_matrix = sim_matrix.to(torch.float32)
    sim_matrix = sim_matrix.masked_fill(mask, -1e9)
    loss = nn.CrossEntropyLoss()(sim_matrix, labels)
    return loss


def supervised_dclw_loss(embeddings1, embeddings2, labels, temperature=0.07, sigma=0.5, use_weighting=False):
    batch_size = embeddings1.size(0)
    embeddings1 = nnf.normalize(embeddings1, dim=1)
    embeddings2 = nnf.normalize(embeddings2, dim=1)
    cross_view_distance = torch.mm(embeddings1, embeddings2.t())
    label_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    pos_similarities1 = cross_view_distance * label_mask
    num_positives1 = label_mask.sum(dim=1).clamp(min=1)
    positive_loss1 = -pos_similarities1.sum(dim=1) / (num_positives1 * temperature)
    if use_weighting:
        avg_pos_similarity = pos_similarities1.sum(dim=1) / num_positives1
        weight = 2 - batch_size * nnf.softmax(avg_pos_similarity / sigma, dim=0)
        positive_loss1 = positive_loss1 * weight
    neg_mask1 = 1 - label_mask
    neg_mask1.fill_diagonal_(0)
    neg_similarity1 = torch.cat(
        (torch.mm(embeddings1, embeddings1.t()), cross_view_distance), dim=1
    ) / temperature
    neg_similarity1.masked_fill_(neg_mask1.repeat(1, 2).bool(), float('-inf'))
    negative_loss1 = torch.logsumexp(neg_similarity1, dim=1)
    pos_similarities2 = cross_view_distance.t() * label_mask
    num_positives2 = label_mask.sum(dim=1).clamp(min=1)
    positive_loss2 = -pos_similarities2.sum(dim=1) / (num_positives2 * temperature)
    if use_weighting:
        positive_loss2 = positive_loss2 * weight
    neg_mask2 = 1 - label_mask
    neg_mask2.fill_diagonal_(0)
    neg_similarity2 = torch.cat(
        (torch.mm(embeddings2, embeddings2.t()), cross_view_distance.t()), dim=1
    ) / temperature
    neg_similarity2.masked_fill_(neg_mask2.repeat(1, 2).bool(), float('-inf'))
    negative_loss2 = torch.logsumexp(neg_similarity2, dim=1)
    loss1 = (positive_loss1 + negative_loss1).mean()
    loss2 = (positive_loss2 + negative_loss2).mean()
    loss = (loss1 + loss2) / 2
    return loss


def dclw_loss(embeddings1, embeddings2, temperature=0.07, sigma=0.5):
    """Distance-weighted contrastive loss."""
    batch_size = embeddings1.size(0)
    embeddings1 = nnf.normalize(embeddings1, dim=1)
    embeddings2 = nnf.normalize(embeddings2, dim=1)
    cross_view_distance = torch.mm(embeddings1, embeddings2.t())
    sim_positive = torch.diag(cross_view_distance)
    positive_loss = -sim_positive / temperature
    similarities = (embeddings1 * embeddings2).sum(dim=1)
    weight = 2 - batch_size * nnf.softmax(similarities / sigma, dim=0)
    positive_loss = positive_loss * weight
    neg_similarity1 = torch.cat(
        (torch.mm(embeddings1, embeddings1.t()), cross_view_distance), dim=1
    ) / temperature
    neg_mask1 = torch.eye(batch_size, device=embeddings1.device).repeat(1, 2)
    neg_similarity1.masked_fill_(neg_mask1.bool(), float('-inf'))
    negative_loss1 = torch.logsumexp(neg_similarity1, dim=1)
    loss1 = (positive_loss + negative_loss1).mean()
    neg_similarity2 = torch.cat(
        (torch.mm(embeddings2, embeddings2.t()), torch.mm(embeddings2, embeddings1.t())), dim=1
    ) / temperature
    neg_mask2 = torch.eye(batch_size, device=embeddings2.device).repeat(1, 2)
    neg_similarity2.masked_fill_(neg_mask2.bool(), float('-inf'))
    negative_loss2 = torch.logsumexp(neg_similarity2, dim=1)
    loss2 = (positive_loss + negative_loss2).mean()
    loss = (loss1 + loss2) / 2
    return loss


LOSS_FUNCTIONS = {
    "dclw": dclw_loss,
    "simclr": simclr_loss,
    "supervised_dclw": supervised_dclw_loss,
    "supervised_contrastive": supervised_contrastive_loss,
}

