"""
Training GNN
==========================================
Author: Aylin Kallmayer
Last updated: 2025-03-19
==========================================
Description:
This script trains a GNN Autoencoder using configuration parameters provided in a JSON file.
It loads training parameters, merges any command-line overrides, iterates over multiple model
configurations (node and edge feature combinations), and performs multiple training runs.
Training progress is logged via TensorBoard and model checkpoints are saved.

Usage:
    python train.py --config config.json [--epochs <override>]
"""

import argparse
import json
import os
import torch
from torch.utils.tensorboard import SummaryWriter

# Import your GNN models and training functions from the refactored modules
from csg.models import GATAutoencoder
from csg.train_eval import train_epoch, evaluate_epoch, save_model
from csg.data_io import load_graphs, create_dataset

# Example configs dictionary (node features, edge features) to iterate over
model_configs = {
    0: (["uniform"], ["uniform"]),
    1: (["category"], ["uniform"]),
    2: (["uniform"], ["distance", "angle"]),
    4: (["category"], ["distance", "angle"]),
}

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Train a GNN Autoencoder using a config file")
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the configuration JSON file")
    # You can add additional CLI overrides here if desired
    parser.add_argument("--epochs", type=int, help="Override number of epochs")
    return parser.parse_args()

def load_config(config_path):
    """
    Load training parameters from a JSON configuration file.
    
    Parameters:
        config_path (str): Path to the JSON config file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, 'r') as f:
        return json.load(f)

def merge_config_args(config, args):
    """
    Merge command-line arguments with the loaded config.
    Any CLI argument that is not None will override the corresponding value in the config.
    
    Parameters:
        config (dict): The loaded configuration.
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        dict: Updated configuration.
    """
    if args.epochs is not None:
        config["epochs"] = args.epochs
    # Extend with other overrides as needed.
    return config

def main_run(run, config):
    """
    Train the model for a single run using the provided configuration.
    
    Parameters:
        run (int): The current run number.
        config (dict): Dictionary containing configuration parameters.
    """
    # Unpack configuration parameters
    hidden_channels = config["hidden_channels"]
    epochs = config["epochs"]
    mod = config["mod"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    stimset = config["stimset"]
    consistency = config["consistency"]
    binary = config["binary"]
    anchor_weighted = config["anchor_weighted"]
    control_model = config["control_model"]

    modname = mod + "BOW" if control_model else mod

    # Determine node and edge strings based on config values
    if "word_embedding" in config["node_features"] or binary:
        nodes = "uniform_object_anchor_diag_word2vec"
    else:
        nodes = "uniform_object_anchor_diag"
    if binary:
        edges = f"uniform_distance_phraseprop_angle_binary_weighted-{anchor_weighted}"
    else:
        edges = "uniform_distance_phraseprop_angle"

    node_features = config["node_features"]
    edge_features = config["edge_features"]
    node_features_string = "_".join(node_features)
    edge_features_string = "_".join(edge_features)

    # Determine the number of node features (for example, 300 for Word2Vec embeddings)
    if "word_embedding" in node_features:
        num_node_features = 300
    else:
        num_node_features = len(node_features)

    # Construct output string for logging and checkpointing
    out_string = (
        f"h{hidden_channels}_mod{modname}_lr{lr}_bs{batch_size}_stimset{stimset}"
        f"_edge-{edge_features_string}_node-{node_features_string}_binary-{binary}"
        f"_weighted-{anchor_weighted}_{run}"
    )

    # Load data and create datasets
    print("Loading graphs...")
    if stimset == "ADEK20K":
        graphs_train = load_graphs(dataset=stimset, nodes=nodes, edges=edges)
        graphs_test = load_graphs(dataset="SCEGRAM", nodes=nodes, edges=edges, consistency=consistency)
        print("Graphs loaded. Creating datasets...")
        _, data_loader_train = create_dataset(
            stimset,
            graphs_train,
            node_feats=node_features,
            edge_feats=edge_features,
            batch_size=batch_size,
            shuffle=True,
        )
        _, data_loader_test = create_dataset(
            "SCEGRAM",
            graphs_test,
            node_feats=node_features,
            edge_feats=edge_features,
            batch_size=batch_size,
            shuffle=False,
        )
        print(
            f"Number of batches: {len(data_loader_train)} (train), {len(data_loader_test)} (test)"
        )
    else:
        raise ValueError(f"Unsupported stimset: {stimset}")

    # Ensure valid model type
    assert mod in ["GCN", "GAT", "CustomGAT"], f"Unknown model type: {mod}"

    # Create the autoencoder model
    model = GATAutoencoder(
        in_channels=num_node_features,
        hidden_channels=hidden_channels * 2,  # as per your design
        latent_dim=hidden_channels,
        out_channels=num_node_features
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.8)

    print(model)

    # TensorBoard writer for logging
    writer = SummaryWriter(f"runs/{out_string}")

    # Training loop with early stopping
    print("Starting training...")
    early_stopping_patience = 10
    no_improvement_epochs = 0
    best_val_loss = float('inf')

    for epoch in range(epochs):
        train_loss = train_epoch(model, data_loader_train, optimizer, use_control_loss=control_model)
        val_loss = evaluate_epoch(model, data_loader_test, use_control_loss=control_model)
        
        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")
        scheduler.step()
        writer.add_scalar("training_loss", train_loss, epoch)
        writer.add_scalar("validation_loss", val_loss, epoch)
        
        # Save model checkpoint
        save_model(model, out_string, epoch, max_epoch=str(epochs - 1))

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1
            if no_improvement_epochs >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch+1}")
                save_model(model, out_string, epoch, max_epoch=str(epoch))
                break

def main():
    """
    Main function: parses command-line arguments, loads configuration from a JSON file,
    merges CLI overrides, and iterates over model configurations for multiple runs.
    """
    # Parse command-line arguments
    args = parse_args()

    # Load configuration from file
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} does not exist.")
    config = load_config(args.config)

    # Merge command-line overrides with the configuration
    config = merge_config_args(config, args)

    # Iterate over custom configurations for node and edge features
    for c in model_configs:
        print(f"=== Running config {c}: node features {model_configs[c][0]}, edge features {model_configs[c][1]} ===")
        config["node_features"], config["edge_features"] = model_configs[c]
        
        for run in range(config["n_runs"]):
            print(f"Run {run} for config {c}")
            main_run(run, config)

if __name__ == "__main__":
    main()
