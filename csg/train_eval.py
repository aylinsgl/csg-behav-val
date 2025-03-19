"""
Training and evaluation functions for GNN
==========================================
Author: Aylin Kallmayer
Last updated: 2025-03-19
==========================================
Description:
This module contains functions for training and evaluating graph neural network models,
as well as for generating embeddings (inference) from trained models. It includes functions
for computing losses, training and evaluation loops, saving model checkpoints, and generating
target node embeddings.
"""

import os
import torch
import torch.nn.functional as F
import pandas as pd

##############################################
# Loss Functions
##############################################
def loss_function(adj_original, adj_recon, features_original, features_recon, edge_index):
    """
    Compute the combined loss for adjacency reconstruction and node feature reconstruction using MSE.
    
    Parameters
    ----------
    adj_original : torch.Tensor
        Original adjacency representation (not used explicitly in this function).
    adj_recon : torch.Tensor
        Reconstructed adjacency values.
    features_original : torch.Tensor
        Original node features.
    features_recon : torch.Tensor
        Reconstructed node features.
    edge_index : torch.Tensor
        Edge index tensor.
    
    Returns
    -------
    float
        The total loss value as the sum of adjacency and node feature losses.
    """
    # Adjacency reconstruction loss (MSE) on the edges
    edge_targets = torch.ones(edge_index.size(1), device=adj_recon.device)
    adj_loss = F.mse_loss(adj_recon, edge_targets)
    # Node feature reconstruction loss (MSE)
    features_loss = F.mse_loss(features_recon, features_original)
    return adj_loss + features_loss

def loss_function_control(features_original, features_recon):
    """
    Compute loss using only node feature reconstruction (MSE).
    
    Parameters
    ----------
    features_original : torch.Tensor
        Original node features.
    features_recon : torch.Tensor
        Reconstructed node features.
    
    Returns
    -------
    float
        The mean squared error loss for node features.
    """
    return F.mse_loss(features_recon, features_original)

##############################################
# Training & Evaluation
##############################################
def train_epoch(model, data_loader, optimizer, use_control_loss=False):
    """
    Train the model for one epoch.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to train.
    data_loader : torch_geometric.loader.DataLoader
        DataLoader providing training data.
    optimizer : torch.optim.Optimizer
        Optimizer for model parameters.
    use_control_loss : bool, optional
        If True, use the control loss (node features only).
    
    Returns
    -------
    float
        Average training loss for the epoch.
    """
    model.train()
    total_loss = 0
    for data in data_loader:
        optimizer.zero_grad()
        adj_recon, features_recon = model(
            data.x.float(),
            data.edge_index.long(),
            edge_weight=data.edge_attr
        )
        if use_control_loss:
            loss = loss_function_control(data.x.float(), features_recon)
        else:
            loss = loss_function(
                data.edge_index.float(),
                adj_recon,
                data.x.float(),
                features_recon,
                data.edge_index
            )
        loss.backward()
        optimizer.step()
        total_loss += loss.item() if not torch.isnan(loss) else 0
    return total_loss / len(data_loader)

def evaluate_epoch(model, data_loader, use_control_loss=False):
    """
    Evaluate the model for one epoch.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to evaluate.
    data_loader : torch_geometric.loader.DataLoader
        DataLoader providing evaluation data.
    use_control_loss : bool, optional
        If True, use the control loss (node features only).
    
    Returns
    -------
    float
        Average evaluation loss for the epoch.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in data_loader:
            adj_recon, features_recon = model(
                data.x.float(),
                data.edge_index.long(),
                edge_weight=data.edge_attr
            )
            if use_control_loss:
                loss = loss_function_control(data.x.float(), features_recon)
            else:
                loss = loss_function(
                    data.edge_index.float(),
                    adj_recon,
                    data.x.float(),
                    features_recon,
                    data.edge_index
                )
            total_loss += loss.item() if not torch.isnan(loss) else 0
    return total_loss / len(data_loader)

def save_model(model, out_string, epoch, max_epoch):
    """
    Save the model's state dictionary to disk.
    
    Parameters
    ----------
    model : torch.nn.Module
        The model to save.
    out_string : str
        Base filename for saving the model.
    epoch : int
        Current epoch number.
    max_epoch : str
        Maximum epoch number (used to determine file naming).
    """
    if not os.path.exists("results/state_dicts"):
        os.makedirs("results/state_dicts")
    if str(epoch) == max_epoch:
        torch.save(model.state_dict(), f"results/state_dicts/{out_string}_EPOCH{epoch}.pt")
    else:
        torch.save(model.state_dict(), f"/Volumes/Extreme SSD/02_computational_scene_grammar/state_dicts/{out_string}_EPOCH{epoch}.pt")

##############################################
# Generating Embeddings / Inference
##############################################
def generate_target_node_embedding(model, data_loader, consistency, binary=False, anchor_weighted=False):
    """
    Generate target node embeddings from the provided data loader.
    
    Parameters
    ----------
    model : torch.nn.Module
        The trained model.
    data_loader : torch_geometric.loader.DataLoader
        DataLoader for the test or inference data.
    consistency : str
        Consistency type used for filtering target nodes.
    binary : bool, optional
        Whether the model uses binary settings.
    anchor_weighted : bool, optional
        Whether the model uses anchor weighted settings.
    
    Returns
    -------
    list
        A list of target node embeddings (as numpy arrays).
    """
    skipped_scenes_file = f'results/node_representations/{consistency}_skipped_scenes.txt'
    seg_report = pd.read_csv('results/scene_grammar_reports/SCEGRAM/segmentation_report_with_scores.csv')
    
    # Unique scene IDs (excluding scene 12)
    scene_ids_seg_report = seg_report.scene_id.unique()
    scene_ids_seg_report = scene_ids_seg_report[scene_ids_seg_report != 12]
    
    # Load target data
    targets = pd.read_csv(f'results/graphs/SCEGRAM/data_targets_{consistency}.csv')
    targets = targets[targets['scene_id'] != 11]  # Exclude scenes not in segmentation report
    targets = targets[targets['scene_id'] != 56]  # Exclude scenes not in segmentation report
    targets = targets[targets['scene_id'] != 12]  # Exclude scenes with only 2 objects
    model.eval()

    skipped_scenes = []
    target_embeddings = []

    for n, data in enumerate(data_loader):
        n_scene = scene_ids_seg_report[n]
        # Filter target data for the current scene
        targets_scene = targets[targets['scene_id'] == n_scene].reset_index(drop=True)
        try:
            # Get target node information
            target_node_id = targets_scene[targets_scene['is_target'] == 1]['object_id_graph'].values[0]
            target_name = targets_scene[targets_scene['is_target'] == 1]['object_name'].values[0]
            print(f"Scene {n_scene}, target: {target_name}, node id: {target_node_id}")
            
            # Compute target node embedding
            x = data.x.float()
            pos_edge_index = data.edge_index
            with torch.no_grad():
                try:
                    z = model.encode(x, pos_edge_index, data.edge_attr)
                except AttributeError:
                    z = model.encoder(x, pos_edge_index, edge_weight=data.edge_attr)
            target_node_embedding = z[target_node_id]
            target_embeddings.append(target_node_embedding.detach().numpy())
        except IndexError:
            print(f"No target node found for Scene {n_scene}. Skipping this scene.")
            skipped_scenes.append(n_scene)
            
    # Write skipped scenes to a file
    with open(skipped_scenes_file, 'w') as f:
        for scene in skipped_scenes:
            f.write(f'{scene}\n')

    return target_embeddings
