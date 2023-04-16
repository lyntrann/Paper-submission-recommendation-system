import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
from torch import nn, optim
from utils import mean_pooling
from math import sqrt

class StackAttentionLayer(nn.Module):
    def __init__(self, embeds_dim, n_classes):
        super().__init__()
        self.linear_query = nn.Linear(embeds_dim, embeds_dim, bias=False)
        self.linear_key = nn.Linear(embeds_dim, embeds_dim, bias=False)
        self.attn_linear = nn.Linear(n_classes, 1, bias=True)
        self.scale = sqrt(embeds_dim)
        self.layer_norm1 = nn.LayerNorm(embeds_dim, eps=1e-12)
        self.mlp = nn.Sequential(
            nn.Linear(embeds_dim, embeds_dim),
            nn.ReLU(),
            nn.Linear(embeds_dim, embeds_dim),
            nn.ReLU(),
            nn.Linear(embeds_dim, embeds_dim)
        )
        self.layer_norm2 = nn.LayerNorm(embeds_dim, eps=1e-12)

    def forward(self, input_feats, label_feats, attn_mask=None):
        '''
        input_feats size: BxSxD
        label_feats size: CxD
        '''
        residual = input_feats[:, 0, :]
        input_feats = self.linear_query(input_feats)
        label_feats = self.linear_key(label_feats)
        dot_product = torch.div(torch.matmul(input_feats, label_feats.T), self.scale)
        # dot product: BxM
        attn = self.attn_linear(dot_product).squeeze()
        attn = attn.masked_fill_(attn_mask.eq(0), value=float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bc, bcd->bd', attn, input_feats)
        residual = self.layer_norm1(out+residual)
        out = self.layer_norm2(self.mlp(out) + residual)
        return out

class PaperModel(nn.Module):
    def __init__(self, hidden_size, model_name_or_path, num_classes) -> None:
        super(PaperModel, self).__init__()
        if isinstance(model_name_or_path, str):
            self.encoder = AutoModel.from_pretrained(model_name_or_path)
        else:
            self.encoder = model_name_or_path
        self.n_classes = num_classes
        self.temperature =  nn.Parameter(torch.ones([]) * 0.07)
        self.attn = StackAttentionLayer(hidden_size, num_classes)
        for param in self.encoder.parameters():
            param.requires_grad_(True)
        
    def get_label_feats(self, aims_ids):
        label_feats = self.encoder(**aims_ids).last_hidden_state[:, 0, :]
        return label_feats
        
    def forward(self, inputs, aims_ids):
        with torch.no_grad():
            self.temperature.clamp_(0.01,0.5)
        hiddens = self.encoder(**inputs).last_hidden_state
        label_feats = self.get_label_feats(aims_ids)
        hiddens = self.attn(hiddens, label_feats, inputs['attention_mask'])
        hiddens = F.normalize(hiddens, dim=-1)
        label_feats = F.normalize(label_feats, dim=-1)
        logits = torch.einsum("bd, cd->bc", hiddens, label_feats)/self.temperature
        outputs = {
            "logits": logits, "cls_feats": hiddens, "label_feats": label_feats
        }
        return outputs