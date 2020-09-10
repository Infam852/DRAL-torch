LABEL_MAPPING_PETS = {
    0: 'Cat',
    1: 'Dog'
}

LABEL_MAPPING_RPS = {
    0: 'Rock',
    1: 'Paper',
    2: 'Scissors'
}

CONFIG_PETIMAGES = {
    'img_size': 64,
    'labels': {
        'data/PetImages/Cat': 0,
        'data/PetImages/Dog': 1
    },
    'n_train': 8000,
    'n_eval': 2000,
    'n_test': 3000,
    'data': {
        'x_path': 'data/x_cats_dogs_skimage.npy',
        'y_path': 'data/y_cats_dogs_skimage.npy'
    },
    'max_reward': 5,
    'max_queries': 1000,
    'query_punishment': 0.5,
    'left_queries_punishment': 5,
    'reward_treshold': 0.92,
    'reward_multiplier': 4,
}

CONFIG_RPS = {
    'img_size': 64,
    'labels': {
        'data/RPS_dataset/rock': 0,
        'data/RPS_dataset/paper': 1,
        'data/RPS_dataset/scissors': 2
    },
    'n_train': 1000,
    'n_eval': 500,
    'n_test': 500,
    'data': {
        'x_path': 'data/x_rps_skimage.npy',
        'y_path': 'data/y_rps_skimage.npy'
    },
    'max_reward': 5,
    'max_queries': 20,
    'query_punishment': 0.5,
    'left_queries_punishment': 5,
    'reward_treshold': 0.7,
    'reward_multiplier': 4,
}

CONFIG = CONFIG_RPS
LABEL_MAPPING = LABEL_MAPPING_RPS
