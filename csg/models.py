"""
GNN model
==========================================
Author: Aylin Kallmayer
Last updated: 2025-03-19
==========================================
Description:
This module defines the Graph Attention Network (GAT) based models used for scene grammar tasks.
It provides the GATEncoder, an inner product decoder for reconstructing graph structure and node features,
and the GATAutoencoder that combines the encoder and decoder into an end-to-end model.
"""

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GATConv, BatchNorm

class GATEncoder(torch.nn.Module):
    """
    Graph Attention Network Encoder
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the GATEncoder.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        """
        super(GATEncoder, self).__init__()
        self.conv1 = GATConv(in_channels, 2 * out_channels, heads=2)
        self.bn1 = BatchNorm(2 * out_channels * 2)
        self.conv2 = GATConv(2 * out_channels * 2, out_channels, heads=2)
        self.bn2 = BatchNorm(out_channels * 2)
        self.conv3 = GATConv(out_channels * 2, out_channels, heads=2)
        self.linear = Linear(out_channels * 2, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass through the GATEncoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor.
        edge_index : torch.Tensor
            Edge index tensor.
        edge_weight : torch.Tensor, optional
            Edge weight tensor.
        
        Returns
        -------
        torch.Tensor
            Encoded output tensor.
        """
        x = self.conv1(x, edge_index, edge_weight).relu()
        x = self.bn1(x)
        x = self.conv2(x, edge_index, edge_weight).relu()
        x = self.bn2(x)
        x = self.conv3(x, edge_index, edge_weight).relu()
        return self.linear(x)

class InnerProductDecoder(torch.nn.Module):
    """
    Inner Product Decoder for reconstructing graph structure and node features.
    """
    def __init__(self, latent_dim, out_channels):
        """
        Initialize the InnerProductDecoder.
        
        Parameters
        ----------
        latent_dim : int
            Dimension of the latent space.
        out_channels : int
            Number of output channels (node features).
        """
        super(InnerProductDecoder, self).__init__()
        self.fc = Linear(latent_dim, out_channels)

    def forward(self, z, edge_index, sigmoid=True):
        """
        Forward pass for the decoder.
        
        Parameters
        ----------
        z : torch.Tensor
            Latent representation of nodes.
        edge_index : torch.Tensor
            Edge index tensor.
        sigmoid : bool, optional
            Whether to apply the sigmoid function to the output.
        
        Returns
        -------
        torch.Tensor, torch.Tensor
            Reconstructed adjacency matrix and node features.
        """
        # Reconstruct adjacency using the inner product of latent representations.
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        adj_recon = torch.sigmoid(value) if sigmoid else value
        # Reconstruct node features through a linear transformation.
        node_features_recon = self.fc(z)
        return adj_recon, node_features_recon

    def forward_all(self, z, sigmoid=True):
        """
        Decode the entire latent space into a dense adjacency matrix.
        
        Parameters
        ----------
        z : torch.Tensor
            The latent space representation.
        sigmoid : bool, optional
            Whether to apply the sigmoid function to the output.
        
        Returns
        -------
        torch.Tensor
            The reconstructed dense adjacency matrix.
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj

class GATAutoencoder(torch.nn.Module):
    """
    Graph Attention Network Autoencoder.
    """
    def __init__(self, in_channels, hidden_channels, latent_dim, out_channels):
        """
        Initialize the GATAutoencoder.
        
        Parameters
        ----------
        in_channels : int
            Number of input channels (node features).
        hidden_channels : int
            Number of hidden channels.
        latent_dim : int
            Dimension of the latent space.
        out_channels : int
            Number of output channels (node features).
        """
        super(GATAutoencoder, self).__init__()
        self.encoder = GATEncoder(in_channels, latent_dim)
        self.decoder = InnerProductDecoder(latent_dim, out_channels)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Forward pass for the autoencoder.
        
        Parameters
        ----------
        x : torch.Tensor
            Input node features.
        edge_index : torch.Tensor
            Edge index tensor.
        edge_weight : torch.Tensor, optional
            Edge weight tensor.
        
        Returns
        -------
        torch.Tensor, torch.Tensor
            Reconstructed adjacency matrix and node features.
        """
        z = self.encoder(x, edge_index, edge_weight)
        adj_recon, node_features_recon = self.decoder(z, edge_index)
        return adj_recon, node_features_recon
