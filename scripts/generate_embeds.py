"""
Embedding Generator
==========================================
Author: Aylin Kallmayer
Last updated: 2025-03-19
==========================================
Description:
This script creates graph and node embeddings using a pretrained GNN model.
It loads parameters from a JSON configuration file, optionally merges command-line
overrides, and iterates over multiple runs and configurations (different
node/edge feature combinations). The resulting embeddings are saved to disk.

Usage:
    python generate_embeddings.py --config config.json [--epoch 49]
    
Arguments:
    --config : Path to the configuration JSON file (default: config.json)
    --epoch  : Optional override for the epoch number to load the model from
"""

import argparse
import json
import os
import torch

from csg.data_io import load_graphs, create_dataset, save_representations, load_model_with_largest_epoch
from csg.train_eval import generate_target_node_embedding

# Dictionary containing model configurations: keys represent a config ID,
# and the values are tuples with node features and edge features lists.
model_configs = {
    0: (["uniform"], ["uniform"]), # Control graph
    1: (["category"], ["uniform"]), # What graph
    2: (["uniform"], ["distance", "angle"]), # Where graph
    4: (["category"], ["distance", "angle"]), # What and Where graph
}

def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Create embeddings using a pretrained GNN model with a config file"
    )
    parser.add_argument("--config", type=str, default="config.json",
                        help="Path to the configuration JSON file")
    parser.add_argument("--epoch", type=int, help="Override the epoch number")
    return parser.parse_args()

def load_config(config_path):
    """
    Load parameters from a JSON configuration file.
    
    Parameters:
        config_path (str): Path to the JSON config file.
    
    Returns:
        dict: Configuration parameters.
    """
    with open(config_path, "r") as f:
        return json.load(f)

def merge_config_args(config, args):
    """
    Merge command-line arguments with the loaded configuration.
    Any argument not None will override the corresponding config value.
    
    Parameters:
        config (dict): The loaded configuration.
        args (argparse.Namespace): Parsed command-line arguments.
    
    Returns:
        dict: Updated configuration.
    """
    if hasattr(args, "epoch") and args.epoch is not None:
        config["epoch"] = args.epoch
    return config

def main_run(run, config):
    """
    Create embeddings for one run using the provided configuration.
    
    Parameters:
        run (int): The current run number.
        config (dict): The configuration parameters.
    """
    # Unpack parameters from config
    hidden_channels = config["hidden_channels"]
    mod = config["mod"]
    lr = config["lr"]
    batch_size = config["batch_size"]
    stimset = config["stimset"]
    consistency = config["consistency"]
    node_features = config["node_features"]
    edge_features = config["edge_features"]
    trainset = config["trainset"]
    binary = config["binary"]
    anchor_weighted = config["anchor_weighted"]

    # Determine nodes and edges strings based on configuration values
    if "word_embedding" in node_features or binary:
        nodes = "uniform_object_anchor_diag_word2vec"
    else:
        nodes = "uniform_object_anchor_diag"
    if binary:
        edges = f"uniform_distance_phraseprop_angle_binary_weighted-{anchor_weighted}"
    else:
        edges = "uniform_distance_phraseprop_angle"

    # Determine number of node features
    if "word_embedding" in node_features:
        num_node_features = 300
    else:
        num_node_features = len(node_features)

    node_features_string = "_".join(node_features)
    edge_features_string = "_".join(edge_features)

    # Construct output string for logging and checkpointing
    out_string = (f"h{hidden_channels}_mod{mod}_lr{lr}_bs{batch_size}_stimsetADEK20K_"
                  f"edge-{edge_features_string}_node-{node_features_string}_binary-{binary}_"
                  f"weighted-{anchor_weighted}_{run}")
    print(out_string)

    # Load graphs and create a DataLoader based on the stimulus set
    if stimset == "SCEGRAM":
        graphs = load_graphs(dataset="SCEGRAM", nodes=nodes, edges=edges, consistency=consistency)
        _, data_loader_test = create_dataset("SCEGRAM", graphs,
                                             node_feats=node_features, edge_feats=edge_features,
                                             batch_size=1, shuffle=False)
    elif stimset == "ADEK20K":
        graphs = load_graphs(dataset=stimset, nodes=nodes, edges=edges)
        _, data_loader_test = create_dataset("ADEK20K", graphs,
                                             node_feats=node_features, edge_feats=edge_features,
                                             batch_size=1, shuffle=False)
    else:
        raise ValueError(f"Unsupported stimset: {stimset}")

    # Load the pretrained model from the state dict with the highest epoch number
    model = load_model_with_largest_epoch(out_string, mod, num_node_features, hidden_channels)
    if model is None:
        print("No model found. Exiting.")
        return

    model.eval()

    # Compute graph representations by averaging encoder outputs for each graph
    graph_reps_avg = []
    for data in data_loader_test:
        x = data.x.float()
        pos_edge_index = data.edge_index
        with torch.no_grad():
            z = model.encoder(x, pos_edge_index, edge_weight=data.edge_attr)
        graph_reps_avg.append(torch.mean(z, dim=0))

    if stimset == "SCEGRAM":
        print(f"Graph representations shape: {len(graph_reps_avg)}, consistency: {consistency}")
    else:
        print(f"Graph representations shape: {len(graph_reps_avg)}")

    # Generate target node representation using the provided function
    target_node_rep = generate_target_node_embedding(model, data_loader_test, consistency,
                                                       binary=binary, anchor_weighted=anchor_weighted)

    # Save the representations to disk
    rep_path_avg = (f"results/graph_representations/{stimset}_{consistency}_{out_string}_"
                    f"trainset{trainset}_epochFINAL_avg")
    rep_path_node = (f"results/node_representations/{stimset}_{consistency}_{out_string}_"
                     f"trainset{trainset}_epochFINAL_node")
    save_representations(graph_reps_avg, rep_path_avg)
    save_representations(target_node_rep, rep_path_node)

def main():
    """
    Main function: parses arguments, loads configuration, and iterates over runs and model configurations.
    """
    args = parse_args()
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file {args.config} does not exist.")
    config = load_config(args.config)
    config = merge_config_args(config, args)

    # Loop over runs and, for SCEGRAM, iterate over multiple consistency types
    for run in range(config["n_runs"]):
        if config["stimset"] == "SCEGRAM":
            for c in ["CON", "SEM", "SYN", "EXSYN", "EXSEMSYN"]:
                config["consistency"] = c
                for k, (nf, ef) in model_configs.items():
                    config["node_features"] = nf
                    config["edge_features"] = ef
                    main_run(run, config)
        elif config["stimset"] == "ADEK20K":
            for k, (nf, ef) in model_configs.items():
                config["node_features"] = nf
                config["edge_features"] = ef
                main_run(run, config)
        else:
            raise ValueError(f"Unsupported stimset: {config['stimset']}")

if __name__ == "__main__":
    main()
