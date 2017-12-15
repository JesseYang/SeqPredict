from easydict import EasyDict as edict

cfg = edict()

cfg.weight_decay = 1e-4

cfg.network = [
    {"name": "fc_1", 'num': 32 },
    {"name": "relu_1", 'k': 0.1 },
    {"name": "fc_2", 'num': 32 },
    {"name": "relu_2", 'k': 0.1 },
    {"name": "fc_3", 'num': 64 },
    {"name": "relu_3", 'k': 0.1 },
    {"name": "fc_4", 'num': 64 },
    {"name": "relu_4", 'k': 0.1 },
    {"name": "dp_1", 'prob': 0.2 },
    {"name": "fc_5", 'num': 2 },
]

cfg.train_x_f_name = 'pred_more_data_alpha_900_10/pred_3/0_train_x'
cfg.train_y_f_name = 'pred_more_data_alpha_900_10/pred_3/0_train_y'
cfg.test_x_f_name = 'pred_more_data_alpha_900_10/pred_3/0_test_x'
cfg.test_y_f_name = 'pred_more_data_alpha_900_10/pred_3/0_test_y'

cfg.feature_len = 62

# cfg.train_list = ["lishanghuang_train.txt"]
# cfg.test_list = "lishanghuang_test.txt"

