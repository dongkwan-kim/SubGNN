from pathlib import Path
import sys

sys.path.insert(0, '..')  # add config to path
import config as general_config

# Flags
GENERATE_SYNTHETIC_G = True  # whether to generate synthetic graph with below specified properties
GENERATE_NODE_EMB = True  # whether to generate node embeddings

# Random Seed
RANDOM_SEED = 42

DESIRED_PROPERTY = "density"  # density, cut_ratio, cc, coreness

# Parameters for generating synthetic subgraphs with specific properties
# https://github.com/mims-harvard/SubGNN/tree/main/prepare_dataset#prepare-dataset
if DESIRED_PROPERTY == "density":
    BASE_GRAPH_TYPE = "barabasi_albert"
    SUBGRAPH_TYPE = "bfs"
    N_SUBGRAPHS = 250
    N_CONNECTED_COMPONENTS = 1
    N_SUBGRAPH_NODES = 20
    FEATURES_TYPE = "one_hot"
    N = 1000
    P = 0.5
    Q = 0
    M = 5
    N_BINS = 3
    SUBGRAPH_GENERATOR = "complete"
    MODIFY_GRAPH_FOR_PROPERTIES = True

elif DESIRED_PROPERTY == "cut_ratio":
    BASE_GRAPH_TYPE = 'barabasi_albert'
    SUBGRAPH_TYPE = 'plant'
    N_SUBGRAPHS = 250
    N_CONNECTED_COMPONENTS = 1
    N_SUBGRAPH_NODES = 20
    FEATURES_TYPE = 'one_hot'
    N = 5000
    P = 0.5  # default
    Q = 0  # not used
    M = 5  # default
    N_BINS = 3  # not used
    SUBGRAPH_GENERATOR = 'complete'
    MODIFY_GRAPH_FOR_PROPERTIES = True

elif DESIRED_PROPERTY == "cc":
    BASE_GRAPH_TYPE = 'barabasi_albert'
    SUBGRAPH_TYPE = 'staple'
    N_SUBGRAPHS = 250
    N_CONNECTED_COMPONENTS = None
    N_SUBGRAPH_NODES = 15
    FEATURES_TYPE = 'one_hot'
    N = 1000
    P = 0.5  # default
    Q = 0
    M = 5  # default
    N_BINS = 2  # not used
    SUBGRAPH_GENERATOR = 'extended_barabasi_albert'
    MODIFY_GRAPH_FOR_PROPERTIES = True

elif DESIRED_PROPERTY == "coreness":
    BASE_GRAPH_TYPE = 'duplication_divergence_graph'
    SUBGRAPH_TYPE = 'plant'
    N_SUBGRAPHS = 30  # Number of subgraphs per coreness value; N_SUBGRAPHS = 30 results in ~250 total subgraphs
    N_CONNECTED_COMPONENTS = 1
    N_SUBGRAPH_NODES = 20
    FEATURES_TYPE = 'one_hot'
    N = 5000
    P = 0.7
    Q = 0  # not used (Q = 1 - P)
    M = 1
    N_BINS = 3
    SUBGRAPH_GENERATOR = 'duplication_divergence_graph'
    MODIFY_GRAPH_FOR_PROPERTIES = True

# Output directory ('density' as an example)
DATASET_DIR = Path(general_config.PROJECT_ROOT) / DESIRED_PROPERTY.upper()

DENSITY_EPSILON = 0.01
DENSITY_RANGE = [0.05, 0.25, 0.45]
CUT_RATIO_EPSILON = 0.001
CUT_RATIO_RANGE = [0.005, 0.0125, 0.02]
K_HOPS_RANGE = [0.12, 0.5, 1.0]
BA_P_RANGE = [0.1, 0.5, 0.9]
CC_RANGE = [1, 1, 1, 1, 5, 6, 7, 8, 9, 10]

MAX_TRIES = 100

# Parameters for training node embeddings for base graph
CONV = "graphsaint_gcn"
MINIBATCH = "GraphSaint"
POSSIBLE_BATCH_SIZES = [512, 1024]
POSSIBLE_HIDDEN = [128, 256]
POSSIBLE_OUTPUT = [64]
POSSIBLE_LR = [0.001, 0.005]
POSSIBLE_WD = [5e-4, 5e-5]
POSSIBLE_DROPOUT = [0.4, 0.5]
POSSIBLE_NB_SIZE = [-1]
POSSIBLE_NUM_HOPS = [1]
POSSIBLE_WALK_LENGTH = [32]
POSSIBLE_NUM_STEPS = [32]
EPOCHS = 100

# Flags for precomputing similarity metrics
CALCULATE_SHORTEST_PATHS = False  # Calculate pairwise shortest paths between all nodes in the graph
CALCULATE_DEGREE_SEQUENCE = False  # Create a dictionary containing degrees of the nodes in the graph
CALCULATE_EGO_GRAPHS = False  # Calculate the 1-hop ego graph associated with each node in the graph
OVERRIDE = False  # Overwrite a similarity file even if it exists
N_PROCESSSES = 4  # Number of cores to use for multi-processsing when precomputing similarity metrics
