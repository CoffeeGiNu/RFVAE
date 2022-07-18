import torch
import torch.nn as nn
import numpy as np


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, 
        out_channels, device='cpu', name=None) -> None:
        super(Encoder, self).__init__()
        self.name = name
        self.device = device
        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_channels//2,
            kernel_size=3,
            stride=2,
            padding=1)
        self.conv_2 = nn.Conv2d(
            in_channels=hidden_channels//2,
            out_channels=hidden_channels,
            kernel_size=3,
            stride=2,
            padding=1)
        self.mean_layer = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=0)
        self.logvar_layer = nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=0)
        self.activation = nn.Softplus()
    
    def forward(self, x):
        x = self.activation(self.conv_1(x))
        x = self.activation(self.conv_2(x))
        mean = self.mean_layer(x)
        log_var = self.logvar_layer(x)

        return mean, log_var


class Decoder(nn.Module):
    def __init__(self, in_features, hidden_features, 
        out_channels, out_size, device='cpu', name=None) -> None:
        super(Decoder, self).__init__()
        self.name = name
        self.device = device
        self.in_features = in_features
        self.out_channels = out_channels
        self.out_size = out_size
        self.affine_1 = nn.Linear(
            in_features=in_features+2,
            out_features=hidden_features//2)
        self.affine_2 = nn.Linear(
            in_features=hidden_features//2,
            out_features=hidden_features)
        self.affine_3 = nn.Linear(
            in_features=hidden_features,
            out_features=hidden_features)
        self.affine_4 = nn.Linear(
            in_features=hidden_features,
            out_features=out_channels)
        self.activation = nn.Softplus()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        i = np.linspace(-1, 1, self.out_size[0])
        j = np.linspace(-1, 1, self.out_size[1])
        ij = np.expand_dims(np.array(np.meshgrid(i, j)), axis=0)
        ij = np.transpose(ij, (0, 1, 3, 2))
        ij = torch.tensor(ij, dtype=torch.float, device=self.device)
        ij = torch.broadcast_to(ij, (x.shape[0], ij.shape[1], ij.shape[2], ij.shape[3]))
        x = torch.broadcast_to(x, (x.shape[0], x.shape[1], ij.shape[2], ij.shape[3]))
        x = torch.concat((x, ij), dim=1)
        x = torch.reshape(x, (-1, self.in_features+2))

        x = self.activation(self.affine_1(x))
        x = self.activation(self.affine_2(x))
        x = self.activation(self.affine_3(x))
        x = self.sigmoid(self.affine_4(x))
        reconstraction = torch.reshape(x, 
            (-1, self.out_channels, self.out_size[0], self.out_size[1]))

        return reconstraction


class ResolutionFreeVariationalAutoEncoder(nn.Module):
    def __init__(self, 
        in_channels, enc_hidden_channels, dim_latent,
        dec_hidden_features, dec_out_channels, out_size,
        encoder=None, decoder=None, 
        latent_size=(1, 1), device='cpu', name=None) -> None:
        super(ResolutionFreeVariationalAutoEncoder, self).__init__()
        self.name = name
        self.device = device
        self.dim_latent = dim_latent
        if not encoder:
            self.encoder = Encoder(
                in_channels=in_channels,
                hidden_channels=enc_hidden_channels,
                out_channels=dim_latent,
                device=device,)
        if not decoder:
            self.decoder = Decoder(
                in_features=dim_latent,
                hidden_features=dec_hidden_features,
                out_channels=dec_out_channels,
                out_size=out_size,
                device=device,)
        self.pool = nn.AdaptiveAvgPool2d(latent_size)
    
    def reparametrization(self, mean, log_var):
        eps = torch.randn(mean.shape, device=self.device)
        z = mean + eps * torch.exp(log_var)

        return z
    
    def forward(self, x):
        mean, log_var = self.encoder(x)
        mean, log_var = self.pool(mean), self.pool(log_var)
        latent = self.reparametrization(mean, log_var)
        outputs = self.decoder(latent)

        return dict(outputs=outputs, mean=mean, 
            log_var=log_var, latent=latent)