import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss

class MetaTransformerLayer(nn.Module):
    """Functional Transformer Layer supporting meta-learning parameter passing."""
    def __init__(self, embed_dim, num_heads, dropout=0.4):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.linear1 = nn.Linear(embed_dim, embed_dim * 4)
        self.linear2 = nn.Linear(embed_dim * 4, embed_dim)
        
        self.dropout = dropout

    def functional_forward(self, x, params, prefix):
        B, L, D = x.shape
        residual = x
        
        q = F.linear(x, params[f'{prefix}.q_proj.weight'], params[f'{prefix}.q_proj.bias'])
        k = F.linear(x, params[f'{prefix}.k_proj.weight'], params[f'{prefix}.k_proj.bias'])
        v = F.linear(x, params[f'{prefix}.v_proj.weight'], params[f'{prefix}.v_proj.bias'])
        
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1) 
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        out = F.linear(out, params[f'{prefix}.out_proj.weight'], params[f'{prefix}.out_proj.bias'])
        
        x = residual + F.dropout(out, p=self.dropout, training=self.training)
        residual = x
        
        ff = F.linear(x, params[f'{prefix}.linear1.weight'], params[f'{prefix}.linear1.bias'])
        ff = F.gelu(ff)
        ff = F.dropout(ff, p=self.dropout, training=self.training)
        ff = F.linear(ff, params[f'{prefix}.linear2.weight'], params[f'{prefix}.linear2.bias'])
        ff = F.dropout(ff, p=self.dropout, training=self.training)
        
        x = residual + ff
        return x

class PathwayGatedTokenizer(nn.Module):
    def __init__(self, num_proteins, embed_dim, pathway_mask, unknown_indices):
        super().__init__()
        self.num_proteins = num_proteins
        self.embed_dim = embed_dim
        self.unknown_indices = unknown_indices
        
        self.mid_dim = 32 
        self.shared_linear = nn.Linear(1, self.mid_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_proteins, self.mid_dim))
        self.gate_logits = nn.Parameter(torch.zeros(num_proteins))
        self.register_buffer('pathway_mask', pathway_mask)
        self.out_projector = nn.Linear(self.mid_dim, embed_dim)
        
        self.num_unknown = len(unknown_indices)
        self.hidden_mult = 32 
        
        if self.num_unknown > 0:
            self.unknown_mlp = nn.Linear(self.num_unknown, self.hidden_mult * embed_dim)
            self.unknown_ln = nn.LayerNorm(embed_dim)

    def functional_forward(self, x, params):
        B = x.shape[0]
        x_reshaped = x.unsqueeze(-1)
        
        w_shared = params['tokenizer.shared_linear.weight']
        b_shared = params['tokenizer.shared_linear.bias']
        x_emb = F.linear(x_reshaped, w_shared, b_shared) 
        
        p_emb = params['tokenizer.pos_embedding']
        x_emb = x_emb + p_emb 
        
        g_logits = params['tokenizer.gate_logits']
        gate = torch.sigmoid(g_logits) 
        x_gated = x_emb * gate.view(1, -1, 1) 
        
        pathway_tokens = torch.einsum('kn, bnd -> bkd', self.pathway_mask, x_gated)
        
        w_out = params['tokenizer.out_projector.weight']
        b_out = params['tokenizer.out_projector.bias']
        pathway_tokens = F.linear(pathway_tokens, w_out, b_out) 
        
        final_tokens = pathway_tokens
        if self.num_unknown > 0:
            x_unk = x[:, self.unknown_indices]
            w_unk = params['tokenizer.unknown_mlp.weight']
            b_unk = params['tokenizer.unknown_mlp.bias']
            
            unk_features = F.linear(x_unk, w_unk, b_unk)
            unk_tokens = unk_features.view(B, self.hidden_mult, self.embed_dim)
            
            ln_w = params.get('tokenizer.unknown_ln.weight', self.unknown_ln.weight)
            ln_b = params.get('tokenizer.unknown_ln.bias', self.unknown_ln.bias)
            unk_tokens = F.layer_norm(unk_tokens, (self.embed_dim,), ln_w, ln_b)
            
            final_tokens = torch.cat([pathway_tokens, unk_tokens], dim=1)
            
        return final_tokens, gate

class ProphetBioGateModel(nn.Module):
    def __init__(self, num_features, config, pathway_mask=None, unknown_indices=None):
        super().__init__()
        self.config = config
        
        if pathway_mask is None or unknown_indices is None:
            raise ValueError("Model requires 'pathway_mask' and 'unknown_indices'!")

        self.tokenizer = PathwayGatedTokenizer(
            num_proteins=num_features,
            embed_dim=config.embed_dim,
            pathway_mask=pathway_mask,
            unknown_indices=unknown_indices
        )
        
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.embed_dim))
        self.tf_layers = nn.ModuleList([
            MetaTransformerLayer(config.embed_dim, config.num_heads, dropout=config.dropout_rate)
            for _ in range(config.num_layers)
        ])
        
        self.classifier = nn.Linear(config.embed_dim, 1)
        self.shortcut_proj = nn.Linear(num_features, config.embed_dim)
        
        self.alphas = nn.ParameterDict()
        for name, param in self.named_parameters():
            if 'alphas' not in name:
                self.alphas[name.replace('.', '_')] = nn.Parameter(
                    torch.ones_like(param) * config.inner_lr
                )

    def functional_forward(self, x, params):
        is_3d = x.dim() == 3
        if is_3d:
            B_tasks, N_samples, P = x.shape
            x_reshaped = x.reshape(B_tasks * N_samples, P)
        else:
            x_reshaped = x
            B = x.shape[0] 
        
        x_seq, gate_values = self.tokenizer.functional_forward(x_reshaped, params)
        
        cls_token = params.get('cls_token', self.cls_token)
        curr_B = x_seq.shape[0]
        cls_token_expand = cls_token.expand(curr_B, -1, -1)
        x_seq = torch.cat((cls_token_expand, x_seq), dim=1)
        x_seq = F.dropout(x_seq, p=self.config.dropout_rate, training=self.training)
        
        for i in range(self.config.num_layers):
            x_seq = self.tf_layers[i].functional_forward(x_seq, params, prefix=f'tf_layers.{i}')
        
        cls_out = x_seq[:, 0, :]
        cls_out = F.dropout(cls_out, p=self.config.dropout_rate, training=self.training)
        
        shortcut = F.linear(x_reshaped, params['shortcut_proj.weight'], params['shortcut_proj.bias'])
        shortcut = F.dropout(shortcut, p=self.config.dropout_rate, training=self.training)
        
        final_feature = cls_out + shortcut 
        
        logits = F.linear(final_feature, params['classifier.weight'], params['classifier.bias'])
        
        if is_3d:
            logits = logits.view(B_tasks, N_samples, 1)
            
        return logits, gate_values