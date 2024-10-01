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
        emb = 1 / (0.001 ** (2 * (emb // 2) / half_dim))
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

class CVAE_MLP(nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_dim: int,
                 latent_dim: int,
                 n_hidden: int,
                 state_dim: int,
                 history_length: int,
                 dropout: float,
                 batch_norm: bool,
                 ) -> None:
        super(CVAE_MLP, self).__init__()
        
        self.state_dim = state_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.latent_dim = latent_dim
        self.history_dim = state_dim * history_length
        
        # Dimensions for goal conditioning and history
        encoder_input_dim = output_dim + self.history_dim
        decoder_input_dim = latent_dim + input_dim
        
        # Conditional encoder and decoder
        self.cond_encoder = MLP(encoder_input_dim, 2*latent_dim, n_hidden, hidden_dim, batch_norm=batch_norm, dropout=dropout)
        
        # Conditional decoder
        self.cond_decoder = MLPPosEmb(decoder_input_dim, output_dim, n_hidden, hidden_dim, batch_norm=batch_norm, dropout=dropout)

    def encode(self, x, state_history):
        """
        Encodes the input and state history to produce latent mean and variance.
        Args:
            x (Tensor): The input data.
            state_history (Tensor): The history states to encode.
        Returns:
            mu (Tensor): Mean of the latent space.
            logvar (Tensor): Log variance of the latent space.
        """
        # Concatenate input and state history
        encoder_inputs = torch.cat((x, state_history), dim=-1)
        encoding = self.cond_encoder(encoder_inputs)
        
        # Get mean and log variance
        mu, logvar = torch.split(encoding, [self.latent_dim, self.latent_dim], dim=-1)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: sample z ~ N(mu, std^2) from latent space.
        Args:
            mu (Tensor): Mean of the latent space.
            logvar (Tensor): Log variance of the latent space.
        Returns:
            z (Tensor): Sampled latent variable.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from standard normal
        return mu + eps * std  # Reparameterized sample

    def decode(self, z, conditioning):
        """
        Decodes the latent variable z along with conditioning information to produce the output.
        Args:
            z (Tensor): Latent variable from latent space.
            conditioning (Tensor): Conditioning data (state history and goal condition).
        Returns:
            recon (Tensor): Reconstructed output.
        """
        # Concatenate latent variable, history, and goal condition
        decoder_inputs = torch.cat((z, conditioning), dim=-1)
        # Reconstruct the output
        recon = self.cond_decoder(decoder_inputs)
        return recon

    def forward(self, x, conditioning = None):
        """
        Forward pass through the CVAE.
        Args:
            x (Tensor): Input data.
            conditioning (Tensor): Conditioning data (state history and goal condition).
        Returns:
            recon (Tensor): Reconstructed output.
            mu (Tensor): Mean of the latent space.
            logvar (Tensor): Log variance of the latent space.
        """
        # Train / validation
        if not conditioning is None:
            # Encoding: get mean and log variance
            state_history = conditioning[:, :self.history_dim]
            mu, logvar = self.encode(x, state_history)
                
            # Reparameterization trick: sample z from latent space
            z = self.reparameterize(mu, logvar)
                
            # Decoding: reconstruct the output using the latent variable and conditioning
            state_goal_conditioning = conditioning[:, self.history_dim - self.state_dim:]
            recon = self.decode(z, state_goal_conditioning)
            return recon, mu, logvar
        
        # Sample
        else:
            return self.sample(x)
    
    def sample(self, conditioning):
        """
        Sample from the prior distribution (N(0, I)) and decode to reconstruct output.
        Args:
            conditioning (Tensor): Conditioning data (state history and goal condition).
        Returns:
            recon (Tensor): Reconstructed output based on sampled latent variable.
        """
        # Sample from standard normal distribution with zero mean and unit variance
        B = len(conditioning)
        mu = torch.zeros((B, self.latent_dim), device=conditioning.device)
        logvar = torch.zeros((B, self.latent_dim), device=conditioning.device)
        
        # Reparameterization trick: sample z from latent space
        z = self.reparameterize(mu, logvar)
        
        # Decoding: reconstruct the output using the latent variable and conditioning
        state_goal_conditioning = conditioning[:, self.history_dim - self.state_dim:]
        recon = self.decode(z, state_goal_conditioning)
        
        return recon