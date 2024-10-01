import torch
import torch.nn as nn

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = torch.arange(half_dim, device=device)
        emb = 1 / (0.01 ** (2 * (emb // 2) / half_dim))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_hidden: int = 1,
                 latent_dim: int = 32,
                 dropout: float = 0.,
                 batch_norm: bool = False
                 ) -> None:
        super(MLP, self).__init__()

        layers = [
            nn.Linear(input_dim, latent_dim),
            nn.PReLU(),
        ]
        
        if batch_norm:
            layers.append(nn.BatchNorm1d(latent_dim))  # BatchNorm after the first Linear layer

        for _ in range(n_hidden):
            layers += [
                nn.Dropout(dropout),
                nn.Linear(latent_dim, latent_dim),
                nn.PReLU(),
            ]
            if batch_norm:
                layers.append(nn.BatchNorm1d(latent_dim))  # BatchNorm after each hidden layer

        layers += [nn.Linear(latent_dim, output_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLPPosEmb(MLP):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 n_hidden: int = 1,
                 latent_dim: int = 32,
                 dropout: float = 0.,
                 batch_norm: bool = False
                 ) -> None:
        pose_embedding_dim = latent_dim // 2
        
        super(MLPPosEmb, self).__init__(
            input_dim + pose_embedding_dim - 1,
            output_dim,
            n_hidden,
            latent_dim,
            dropout,
            batch_norm
        )
        
        self.pos_embedding = SinusoidalPosEmb(pose_embedding_dim)
        
    def forward(self, x):
        phase = x[:, -1]
        pose_emb_phase = self.pos_embedding(phase)
        
        # Fix concatenation dimension (should be `dim=1`)
        x_pos_emb = torch.cat((x[:, :-1], pose_emb_phase), dim=1)
        return self.mlp(x_pos_emb)