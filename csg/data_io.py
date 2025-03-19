"""
Data I/O
==========================================
Author: Aylin Kallmayer
Last updated: 2025-03-19
==========================================
Description:
This module provides functions for loading and processing data, creating datasets, 
and handling pretrained models for graph neural networks.
"""

import pickle
import glob
import re
import json
import torch
import numpy as np
import pandas as pd
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import from_networkx
from torch_geometric.nn import GAE
from csg.models import GATAutoencoder

def load_activations(category, layer, train_test='train'):
    """
    Load activations for a single category.

    Parameters
    ----------
    category : str
        Category name.
    layer : int
        Layer number.
    train_test : str, optional
        'train' or 'test' images.

    Returns
    -------
    acts : numpy.array
        Array of activations.
    """
    path = f'/Volumes/Extreme SSD/GAN_EEG_comp_sce_gram/generator_activations/{category}/{category}_layer{layer}_acts.npy'
    print(path)
    acts = np.load(path)
    if train_test == 'train':
        return acts[30:1030, :]
    elif train_test == 'test':
        return acts[0:30, :]

def load_image_scores(category, object_score='anchor_status_freq'):
    """
    Load image scores for a single category.
    
    Parameters  
    ----------
    category : str
        Category name.
    object_score : str, optional
        Object score to use, either 'anchor_status_freq' or 'diagnosticity'.

    Returns
    -------
    df1 : pandas.DataFrame
        DataFrame for 30 test images.
    df2 : pandas.DataFrame
        DataFrame for 1000 train images.
    """
    path = f'data/reports/{category}_scegram_report.csv'
    df = pd.read_csv(path)
    df_anch_max = df.groupby('number')[object_score].max().reset_index()
    df_agg = df.loc[df.groupby('number')[object_score].idxmax()].reset_index(drop=True)
    # Separate into two DataFrames based on the 'number' column.
    df1 = df_agg[df_agg['number'].between(1, 30)].reset_index(drop=True)
    df2 = df_agg[~df_agg['number'].between(1, 30)].reset_index(drop=True)
    return df1, df2

def load_graphs(dataset, nodes, edges, consistency=None):
    """
    Load graphs for a given dataset, node type, and edge type.

    Parameters
    ----------
    dataset : str
        Dataset name.
    nodes : str
        Node type.
    edges : str
        Edge type.
    consistency : str, optional
        Consistency flag (used for SCEGRAM).

    Returns
    -------
    graphs : list
        List of graphs.
    """
    if dataset == 'ADEK20K':
        graphs_path = 'results/graphs/ADEK20K'
        with open(f"{graphs_path}/{dataset}_node{nodes}_edge{edges}_graphs.json", 'rb') as handle:
            graphs = pickle.load(handle)
        return graphs
    if dataset == 'SCEGRAM':
        graphs_path = 'results/graphs/SCEGRAM'
        path = f"{graphs_path}/{dataset}_{consistency}_node{nodes}_edge{edges}_graphs.json"
        print(path)
        with open(path, 'rb') as handle:
            graphs = pickle.load(handle)
        print(len(graphs))
        return graphs

def create_dataset(stimset, graphs_all, node_feats, edge_feats, batch_size=1, shuffle=True, normalize=True):
    """
    Create a dataset and a DataLoader.

    Parameters
    ----------
    stimset : str
        Stimulus set.
    graphs_all : list
        List of graphs.
    node_feats : list
        Node feature names.
    edge_feats : list
        Edge feature names.
    batch_size : int, optional
        Batch size.
    shuffle : bool, optional
        Whether to shuffle the dataset.
    normalize : bool, optional
        Whether to normalize features.

    Returns
    -------
    dataset : list
        List of processed graphs.
    data_loader : torch_geometric.loader.DataLoader
        DataLoader for the dataset.
    """
    dataset = []
    print(f"Total number of graphs: {len(graphs_all)}")
    if stimset in ["ADEK20K", "SCEGRAM"]:
        for n, g in enumerate(graphs_all):
            if g == 12:  # Skip scene 12 (only 2 objects)
                print("skipping scene 12")
                continue
            try:
                D = from_networkx(graphs_all[g], group_node_attrs=node_feats, group_edge_attrs=edge_feats)
                if normalize:
                    # Normalize node features
                    if D.x is not None:
                        D.x = D.x.float()
                        D.x = (D.x - D.x.mean(dim=0)) / (D.x.std(dim=0) + 1e-5)
                    # Normalize edge attributes
                    if D.edge_attr is not None:
                        D.edge_attr = D.edge_attr.float()
                        D.edge_attr = (D.edge_attr - D.edge_attr.mean(dim=0)) / (D.edge_attr.std(dim=0) + 1e-5)
                dataset.append(D)
            except KeyError:
                print(f"KeyError: Scene {g} could not be processed.")
                continue
        print(f"Number of graphs in the dataset: {len(dataset)}")
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
        return dataset, data_loader

def load_pretrained_model(state_dict_path, mod, num_features, hidden_channels):
    """
    Load a pretrained model from a state dictionary.

    Parameters
    ----------
    state_dict_path : str
        Path to the state dictionary.
    mod : str
        Model type.
    num_features : int
        Number of node features.
    hidden_channels : int
        Number of hidden channels.

    Returns
    -------
    model : torch.nn.Module
        Pretrained model.
    """
    if mod == 'GCN':
        model = GAE(GCNEncoder(num_features, hidden_channels))
    elif mod == 'GAT':
        model = GAE(GATEncoder(num_features, hidden_channels))
    elif mod in ['CustomGAT', 'CustomGATBOW']:
        model = GATAutoencoder(num_features, hidden_channels * 2, hidden_channels, num_features)
    else:
        raise ValueError(f"Model type {mod} not recognized.")
    state_dict = torch.load(state_dict_path)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def load_model_with_largest_epoch(out_string, mod, num_node_features, hidden_channels):
    """
    Find and load the model with the largest epoch number from state dictionary files.

    Parameters
    ----------
    out_string : str
        Base part of the filename to search for.
    mod : str
        Model type.
    num_node_features : int
        Number of node features.
    hidden_channels : int
        Number of hidden channels.

    Returns
    -------
    model : torch.nn.Module or None
        Loaded model if found; otherwise, None.
    """
    state_dict_pattern = f"results/state_dicts/{out_string}_EPOCH*.pt"
    matching_files = glob.glob(state_dict_pattern)
    regex_pattern = r"EPOCH(\d+)"
    if matching_files:
        matching_files.sort(key=lambda x: int(re.search(regex_pattern, x).group(1)), reverse=True)
        state_dict_path = matching_files[0]
        model = load_pretrained_model(state_dict_path, mod, num_node_features, hidden_channels)
        print(f"Loaded model from {state_dict_path}")
        return model
    else:
        print("No matching state dict file found.")
        print(f"Pattern: {state_dict_pattern}")
        return None

def save_representations(graph_reps, rep_path):
    """
    Save the graph representations to disk.

    Parameters
    ----------
    graph_reps : list
        List of graph representations.
    rep_path : str
        Path (without extension) where the representations will be saved.
    """
    with open(f"{rep_path}.json", 'wb') as handle:
        pickle.dump(graph_reps, handle)

def load_excluded_indices(path_trainset, path_testset, single_cat=False):
    """
    Load excluded indices for train and test sets.

    Parameters
    ----------
    path_trainset : str
        Path to the trainset indices.
    path_testset : str
        Path to the testset indices.
    single_cat : bool, optional
        If True, load indices for a single category.

    Returns
    -------
    excluded_train : dict
        Excluded indices and filenames for the trainset.
    excluded_test : dict
        Excluded indices and filenames for the testset.
    """
    if single_cat:
        with open(path_trainset, 'r') as f:
            excluded_train = json.load(f)
        with open(path_testset, 'r') as f:
            excluded_test = json.load(f)
    else:
        with open(path_trainset, 'r') as f:
            excluded_train = json.load(f)
            excluded_train['names'] = [f'/Volumes/Extreme SSD/09_GeneratedImagesSegmentation/train/{x}' for x in excluded_train['names']]
        with open(path_testset, 'r') as f:
            excluded_test = json.load(f)
            excluded_test['names'] = [f'/Volumes/Extreme SSD/09_GeneratedImagesSegmentation/val/{x}' for x in excluded_test['names']]
    return excluded_train, excluded_test
