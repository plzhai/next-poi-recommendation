from datasets.dataset import LBSNDataset
import os
import sampler
import utils
from models.models_for_open_data import GeoSAN


data_path = 'datasets/brightkite/'
filename_raw = os.path.join(data_path, "totalCheckins.txt")
filename_clean = os.path.join(data_path, "LSBNDataset.data")

if os.path.isfile(filename_clean):
    dataset = utils.unserialize(filename_clean)
else:
    dataset = LBSNDataset(filename_raw)
    utils.serialize(dataset, filename_clean)

trainset, testset = dataset.split()

config = {}
config['model_params'] = {}
config['dataset'] = {}

config['model_params']['sequence_length'] = 100  # the time steps, i.e., T.
config['model_params']['prediction_type'] = "sequential" # or sequential
config['model_params']['num_negative_samples'] = 100
# self._selected_features = config['model_params']['selected_feature_names']
config['dataset']['name'] = 'brightkite'
config['dataset']['n_user'] = dataset.n_user
config['dataset']['n_loc'] = dataset.n_loc
config['dataset']['n_time'] = dataset.n_time
config['dataset']['n_region'] = dataset.n_loc
config['dataset']['user_embedding_size'] = 50
config['dataset']['poi_embedding_size'] = 50
config['dataset']['time_embedding_size'] = 50
config['dataset']['reg_embedding_size'] = 50

# 2. settings for the network
config['model_params']['transformer'] = {}
config['model_params']['transformer']['nhid_ffn'] = 128
config['model_params']['transformer']['num_encoder_blocks'] = 2
config['model_params']['transformer']['d_model'] = 64
config['model_params']['transformer']['nhead_enc'] = 4
# self.src_square_mask = config['model_params']['transformer']['mask']
# self.src_binary_mask

config['model_params']['transformer']['scaled'] = True
config['model_params']['dropout_rate'] = 0.5  # default 0.5
config['model_params']['use_attention_as_decoder'] = False  # default False
config['model_params']['loss'] = {}
config['model_params']['loss']['weighted'] = False # default False
config['model_params']['loss']['temperature'] = 1.0

# 3. settings for training

config['model_params']['learning_rate'] = 0.001
config['model_params']['nb_epoch'] = 100
config['model_params']['batch_size'] = 128
config['model_params']['max_patience'] = 10
config['model_params']['path_log'] = 'results_log'
config['model_params']['dev_size'] = 0.2

config['model_params']['path_model'] = 'checkpoints'
config['dataset']['NegativeSampler'] = "UniformNegativeSampler"

model = GeoSAN(config)

model.fit(trainset, testset)
