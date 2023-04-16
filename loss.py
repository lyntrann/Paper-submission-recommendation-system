import torch
from torch import nn, optim
import torch.nn.functional as F

class CELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        
    def forward(self, outputs, targets):
        return self.xent_loss(outputs['logits'], targets)

class SupConLoss(nn.Module):

    def __init__(self, alpha, temp):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.alpha = alpha
        self.temp = temp

    def nt_xent_loss(self, anchor, target, labels):
        with torch.no_grad():
            labels = labels.unsqueeze(-1)
            mask = torch.eq(labels, labels.transpose(0, 1))
            # delete diag elem
            mask = mask ^ torch.diag_embed(torch.diag(mask))
        # compute logits
        anchor_dot_target = torch.einsum('bd,cd->bc', anchor, target) / self.temp
        # delete diag elem
        anchor_dot_target = anchor_dot_target - torch.diag_embed(torch.diag(anchor_dot_target))
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_target, dim=1, keepdim=True)
        logits = anchor_dot_target - logits_max.detach()
        
        exp_logits = torch.exp(logits)
        # mask out positives
        logits = logits * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        # in case that mask.sum(1) is zero
        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        # compute log-likelihood
        pos_logits = (mask * log_prob).sum(dim=1) / mask_sum.detach()
        loss = -1 * pos_logits.mean()
        return loss

    def forward(self, outputs, targets):
        normed_cls_feats = F.normalize(outputs['cls_feats'], dim=-1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['logits'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_cls_feats, normed_cls_feats, targets)
        return ce_loss + cl_loss

class DualLoss(SupConLoss):
    def __init__(self, alpha=0.1, temp=0.1):
        super().__init__(alpha, temp)

    def forward(self, outputs, targets):
        cls_feats = outputs['cls_feats']
        label_feats = outputs['label_feats']
        normed_pos_label_feats = torch.gather(label_feats, dim=1, index=targets.reshape(-1, 1, 1).expand(-1, 1, label_feats.size(-1))).squeeze(1)
        ce_loss = (1 - self.alpha) * self.xent_loss(outputs['logits'], targets)
        cl_loss = self.alpha * self.nt_xent_loss(normed_pos_label_feats, cls_feats, targets)
        return ce_loss, cl_loss
    
class LabelRegLoss(nn.Module):
    def __init__(self, threshold=0.5, is_normalize=True):
        super().__init__()
        self.is_normalize = is_normalize
        self.threshold = threshold
    
    def forward(self, x):
        if self.is_normalize == False:
            x = F.normalize(x, dim=-1)
        sim_matrix = torch.bmm(x, x.transpose(1, 2))
        mask = torch.diag_embed(torch.ones_like(sim_matrix.diagonal(dim1=1, dim2=2)))
        sim_matrix = sim_matrix.masked_fill(mask == 1, float('-inf'))
        sim_matrix, _ = torch.max(sim_matrix, dim=-1)
        sim_matrix = F.relu(sim_matrix-self.threshold)
        loss = sim_matrix.mean()
        return loss