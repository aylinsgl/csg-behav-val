# Computational Scene Grammar

Author: Aylin Kallmayer  
Last updated: 2025-03-19  

## Overview

Computational Scene Grammar is a research project for generating scene grammar representations leveraging graph neural networks and unsupervised learning:

- Loading and preprocessing scene graph data.
- Defining and training Graph Neural Network (GNN) models (with a focus on Graph Attention Networks).
- Generating embeddings for the SCEGRAM database.
- Evaluating models on classification and consistency tasks.

## Repository Structure

```
.
├── csg
│   ├── __init__.py
│   ├── data_io.py             # Data loading, processing, and model utility functions
│   ├── models.py              # Model definitions (e.g., GATEncoder, GATAutoencoder)
│   ├── train_eval.py          # Training, evaluation, and inference functions
│   └── utils.py               # Utility functions for computing centroids and loading masks
├── scripts
│   ├── train.py               # Script to train the GNN autoencoder using a configuration file
│   ├── create_embeddings.py   # Script to generate embeddings from a pretrained model
│   └── segmentation_reports.py# Script to generate segmentation reports for datasets
│   └── validate.ipynb         # Jupyter notebook to evaluate embeddings on behavioral data
│   └── GraphGeneration.ipynb  # Jupyter notebook to generate graphs
├── config.json                # Sample configuration file for training and inference parameters
├── README.md                  # This file
└── requirements.txt           # List of Python dependencies and versions
```

## Installation

Clone the Repository:

```
git clone https://github.com/aylinsgl/csg-behav-val.git
cd csg-behav-val
```

Inside the directory, install:

```
pip install .
```

## Usage

### Training a Model
To train the GNN autoencoder, use the script in the scripts folder:

```
python scripts/train.py --config config.json --epochs 50
```

This script will:

- Load training parameters from config.json.
- Optionally override parameters with command-line arguments (e.g., --epochs).
- Train the model while logging progress with TensorBoard.
- Save model checkpoints in results/state_dicts/.

### Generating Embeddings
To generate embeddings using a pretrained model, run:

```
python scripts/create_embeddings.py --config config.json
```

The script loads the model (based on the highest epoch checkpoint) and produces graph and node embeddings saved in `results/graph_representations/` and `results/node_representations/`.

### Configuration
The repository uses a JSON configuration file (config.json) to store all training and model parameters. An example configuration might look like:

```
{
  "hidden_channels": 10,
  "epochs": 50,
  "mod": "CustomGAT",
  "lr": 0.0005,
  "batch_size": 15,
  "stimset": "ADEK20K",
  "consistency": "CON",
  "node_features": ["uniform"],
  "edge_features": ["uniform"],
  "binary": false,
  "anchor_weighted": false,
  "n_runs": 50,
  "control_model": true
}
```

You can override specific parameters via command-line arguments when running the scripts.
