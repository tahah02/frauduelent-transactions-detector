import os


# Paths
DATA_DIR = 'data'
MODELS_DIR = 'models'
CLEAN_CSV_PATH = 'data/Clean.csv'
FEATURED_DATASET_PATH = 'data/featured_dataset.csv'
MODEL_PATH = 'models/isolation_forest.pkl'

# Transfer Type Mappings
TRANSFER_TYPE_MAPPING = {
    'S': 'Overseas',
    'I': 'Ajman', 
    'L': 'UAE',
    'Q': 'Quick',
    'O': 'Own'
}

TRANSFER_TYPE_ENCODED = {'S': 4, 'I': 1, 'L': 2, 'Q': 3, 'O': 0}
TRANSFER_TYPE_RISK = {'S': 0.9, 'I': 0.1, 'L': 0.2, 'Q': 0.5, 'O': 0.0}


def ensure_data_dir():

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)


def get_clean_csv_path():
    return CLEAN_CSV_PATH


def get_feature_engineered_path():
    return FEATURED_DATASET_PATH


def get_model_path():
    return MODEL_PATH
