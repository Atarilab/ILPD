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
        emb = 1 / (0.0001 ** (2 * (emb // 2) / half_dim))
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
        pose_embedding_dim = latent_dim // 4
        
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
    
class LSTMHist(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 state_dim: int,
                 history_length: int,
                 n_layers: int,
                 n_hidden: int,
                 latent_dim: int,
                 dropout: float,
                 batch_norm: bool,
                 ) -> None:
        super(LSTMHist, self).__init__()
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.history_length = history_length
        self.history_dim = state_dim * self.history_length
        
        self.state_encoder = MLP(state_dim, latent_dim, 1, latent_dim * 2, batch_norm=batch_norm)
        
        # LSTM for encoding history data
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout
        )
        
        self.cond_dim = input_dim - state_dim
        self.action_layer = MLPPosEmb(
            input_dim=latent_dim + self.cond_dim,
            output_dim=output_dim,
            n_hidden=n_hidden,
            latent_dim=latent_dim,
            dropout=dropout,
            batch_norm=batch_norm
        )
        
    def forward(self, x):
        # Split input into history, state, and conditioning data
        history, cond = torch.split(x, [self.history_dim, self.cond_dim], dim=-1)
        
        # Reshape the history to [batch_size, history_length, state_dim] before encoding
        history = history.reshape(-1, self.state_dim)
        history = self.state_encoder(history).reshape(-1, self.history_length, self.latent_dim)

        # Encode the history using LSTM
        _, (h_n, _) = self.lstm(history)  # h_n: [n_layers, batch_size, latent_dim]
        history_encoded = h_n[-1]  # Take the last layer's hidden state [batch_size, latent_dim]
        
        # Concatenate the encoded history with the current state

        # Concatenate with conditioning data and pass through the action layer
        encoded_inputs = torch.cat((history_encoded, cond), dim=-1)
        actions = self.action_layer(encoded_inputs)
        
        return actions
