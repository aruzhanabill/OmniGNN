import torch
from torch import nn
import torch.nn.functional as F

__all__ = [
    "GraphAttentionLayer",
    "OmniGNNLayer",
    "TransformerBlockWithALiBi",
    "TemporalEncoder",
    "NodePredictor",
]

##################################
###      GRAPH ATTENTION       ###
##################################

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer with edge attributes and multi-head attention. 
    As described in this paper: `"Graph Attention Networks" <https://arxiv.org/pdf/1710.10903.pdf>`.

    This operation can be mathematically described as:

        β_ij = a^T(W h_i || W h_j || *W* edge_attr_ij) # updated to include edge attributes 
        α_ij^k = softmax_j(e_ij) = exp(LeakyReLU(β_ij)) / Σ_j'∈ Ni exp(LeakyReLU(β_ij'))    

        We aggregate representations from multiple heads and use average pooling to update the target node’s representation:
        h_i = σ(1/K Σ_k=1^K Σ_j'∈ Ni (α_ij^k W^k h_j))
        
        where h_i and h_j are the feature vector representations of nodes i and j respectively, W is a learnable weight matrix,
        a is an attention mechanism that computes the attention coefficients β_ij, and σ is an activation function.

    Supports batched input, edge attributes, and multi-relational edges.

    """
    def __init__(self, in_features, out_features, n_heads, edge_attr_dim=0, concat=True, dropout=0.4, leaky_relu_slope=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        self.edge_attr_dim = edge_attr_dim

        if concat:
            assert out_features % n_heads == 0
            self.out_features = out_features
            self.n_hidden = out_features // n_heads
        else:
            self.n_hidden = out_features
            self.out_features = out_features

        self.W = nn.Parameter(torch.empty(in_features, self.n_hidden * n_heads)) # A shared linear transformation, parametrized by a weight matrix W is applied to every node
        self.a = nn.Parameter(torch.empty(n_heads, 2 * self.n_hidden + edge_attr_dim, 1)) # Attention weights

        self.leakyrelu = nn.LeakyReLU(leaky_relu_slope)
        self.softmax = nn.Softmax(dim=1)
        self.reset_parameters()

    def reset_parameters(self):
        """
        Reinitialize learnable parameters.
        """
        nn.init.xavier_normal_(self.W)
        nn.init.xavier_normal_(self.a)
    
    def forward(self, h: torch.Tensor, adj_mat: torch.Tensor, edge_attr: torch.Tensor = None):
        B, N, _ = h.shape

        if adj_mat.dim() == 2:
            adj_mat = adj_mat.unsqueeze(0).expand(B, -1, -1) # (N,N) ➜ (B,N,N)
        if edge_attr is None:
            edge_attr = torch.zeros((B, N, N, self.edge_attr_dim), device=h.device)
        elif edge_attr.dim() == 3:
            edge_attr = edge_attr.unsqueeze(0).expand(B, -1, -1, -1) # (N,N,E) ➜ (B,N,N,E)

        h_trans = torch.matmul(h, self.W)                      # (B, N, H*D)
        h_trans = F.dropout(h_trans, self.dropout, self.training)
        h_trans = h_trans.view(B, N, self.n_heads, self.n_hidden).permute(0, 2, 1, 3)  # (B, H, N, D)

        outputs = []
        for b in range(B):
            beta_scores = []
            for head in range(self.n_heads):
                h_i = h_trans[b, head].unsqueeze(1).repeat(1, N, 1) # (N, N, D)
                h_j = h_trans[b, head].unsqueeze(0).repeat(N, 1, 1) # (N, N, D)
                edge_input = torch.cat([h_i, h_j, edge_attr[b]], dim=-1)
                beta = torch.matmul(edge_input, self.a[head]).squeeze(-1) # (N, N)
                beta_scores.append(beta)
            
            beta = torch.stack(beta_scores) # (H, N, N)
            mask = adj_mat[b] > 0
            mask = mask.unsqueeze(0).expand(self.n_heads, -1, -1)
            beta = torch.where(mask, beta, torch.full_like(beta, -9e15))

            attn = self.softmax(beta)
            attn = F.dropout(attn, self.dropout, self.training)
            h_prime = torch.matmul(attn, h_trans[b]) # (H, N, D)

            if self.concat:
                h_out = h_prime.permute(1, 0, 2).reshape(N, self.out_features) # (N, H*D)
            else:
                h_out = h_prime.mean(dim=0) # (N, D)
            
            outputs.append(h_out.unsqueeze(0)) # (1, N, out_features)

        return torch.cat(outputs, dim=0) # (B, N, out_features)

###################################
###      OMNIGNN LAYER          ###
###################################

class OmniGNNLayer(nn.Module):
    """
        Applies a GraphAttentionLayer for each meta-path and aggregates them via attention.
        Defines one layer of the hierarchical multi-relational graph embedding
        Math:
        h_vi = σ(Σ_j=1^3 Softmax(W h_ij) h_ij)
        ŷ_vt = σ(W_1 z_vt + b_1)
        Z = softmax(QK^T / sqrt(d) + m * P + M)V
        Applies one GAT layer per meta-path's subgraph (adj + edge_attr),
        then aggregates them via attention to produce a unified node embedding.
    """
    def __init__(self, in_features, out_features, n_heads, edge_attr_dim, meta_paths):
        super().__init__()
        self.meta_paths = meta_paths
        self.gat_layers = nn.ModuleDict({
            mp: GraphAttentionLayer(in_features, out_features, n_heads, edge_attr_dim, concat=False)
            for mp in meta_paths
        })
        self.meta_path_attn = nn.Linear(out_features, 1)

    def forward(self, x, adj_dict, edge_attr_dict):
        H_meta = [self.gat_layers[mp](x, adj_dict[mp], edge_attr_dict[mp]) for mp in self.meta_paths]
        meta_embeddings = torch.stack(H_meta, dim=1)  # (B, n_meta, N, D)

        logits = self.meta_path_attn(meta_embeddings).squeeze(-1)
        attn_weights = F.softmax(logits, dim=1)  # (B, n_meta, N)
        output = torch.sum(attn_weights.unsqueeze(-1) * meta_embeddings, dim=1)  # (B, N, D)
        return output

############################################
###     TRANSFORMER WITH ALiBi BIAS      ###
############################################

class TransformerBlockWithALiBi(nn.Module):
    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def get_alibi_bias(self, seq_len, device):
        slopes = torch.tensor([2 ** (-(8.0 / self.n_heads) * i) for i in range(self.n_heads)], device=device)
        rel_pos = torch.arange(seq_len, device=device).unsqueeze(0) - torch.arange(seq_len, device=device).unsqueeze(1)
        rel_pos = -rel_pos.clamp(min=0)
        return slopes[:, None, None] * rel_pos[None, :, :]

    def forward(self, x, alibi_bias=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / self.head_dim ** 0.5
        if alibi_bias is None:
            alibi_bias = self.get_alibi_bias(T, x.device).unsqueeze(0)
        scores = scores + alibi_bias

        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, C)
        out = self.out_proj(out)

        x = x + self.dropout(out)
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x

###################################
###       TEMPORAL ENCODER      ###
###################################

class TemporalEncoder(nn.Module):
    def __init__(self, embed_dim, n_heads=4, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlockWithALiBi(embed_dim, n_heads) for _ in range(n_layers)
        ])

    def forward(self, x, alibi_bias=None):
        for layer in self.layers:
            x = layer(x, alibi_bias=alibi_bias)
        return x[:, -1]  # Use last time step output

###################################
###     NODE PREDICTION HEAD    ###
###################################

class NodePredictor(nn.Module):
    def __init__(self, input_dim, output_dim, n_nodes):
        super().__init__()
        self.predictors = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for _ in range(n_nodes)
        ])

    def forward(self, x):  # x: (B, n_nodes, input_dim)
        return torch.cat([self.predictors[i](x[:, i]) for i in range(len(self.predictors))], dim=1)