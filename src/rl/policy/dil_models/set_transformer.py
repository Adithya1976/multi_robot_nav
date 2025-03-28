import math
from typing import Optional
import torch.nn as nn
import torch
import torch.nn.functional as F


class SetTransformer(nn.Module):
    def __init__(self, dim_input,  dim_output, device=None, num_outputs=1, 
            num_inds=32, dim_hidden=256, num_heads=4, ln=False):
        super(SetTransformer, self).__init__()
        self.isab1 = ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln)
        self.isab2 = ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln)
        self.pma =PMA(dim_hidden, num_heads, num_outputs, ln=ln)
        self.sab1 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.sab2 = SAB(dim_hidden, dim_hidden, num_heads, ln=ln)
        self.fc = nn.Linear(dim_hidden, dim_output)

    def forward(self, X, key_padding_mask=None):
        X = self.isab1(X, key_padding_mask=key_padding_mask)
        X = self.isab2(X, key_padding_mask=None)
        X = self.pma(X, key_padding_mask=None)
        X = self.sab1(X, key_padding_mask=None)
        X = self.sab2(X, key_padding_mask=None)
        return self.fc(X).squeeze()
    
class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K, key_padding_mask=None):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        # Split embeddings for multihead attention manually
        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)  # (B*num_heads, q_len, dim_split)
        K_ = torch.cat(K.split(dim_split, 2), 0)  # (B*num_heads, k_len, dim_split)
        V_ = torch.cat(V.split(dim_split, 2), 0)  # (B*num_heads, k_len, dim_split)

        # Compute attention scores
        scores = Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V)
        if key_padding_mask is not None:
            # key_padding_mask: (B, k_len)
            # Expand to (B, 1, k_len) and then repeat for each head
            B = key_padding_mask.size(0)
            expanded_mask = key_padding_mask.unsqueeze(1).repeat(self.num_heads, 1, 1)
            scores = scores.masked_fill(expanded_mask, float('-inf'))
        A = torch.softmax(scores, dim=2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = self.ln0(O) if hasattr(self, 'ln0') else O
        O = O + F.relu(self.fc_o(O))
        O = self.ln1(O) if hasattr(self, 'ln1') else O
        return O

class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        return self.mab(X, X, key_padding_mask)

class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask)
        return self.mab1(X, H)

class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X, key_padding_mask=None):
        return self.mab(self.S.repeat(X.size(0), 1, 1), X, key_padding_mask=key_padding_mask)