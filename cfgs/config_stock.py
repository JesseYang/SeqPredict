from easydict import EasyDict as edict

cfg = edict()

cfg.weight_decay = 1e-4

cfg.train_x_f_name = 'pred_more_data_alpha_900_10/pred_3/0_train_x'
cfg.train_y_f_name = 'pred_more_data_alpha_900_10/pred_3/0_train_y'
cfg.test_x_f_name = 'pred_more_data_alpha_900_10/pred_3/0_test_x'
cfg.test_y_f_name = 'pred_more_data_alpha_900_10/pred_3/0_test_y'

cfg.feature_len = 54

# cfg.train_list = ["lishanghuang_train.txt"]
# cfg.test_list = "lishanghuang_test.txt"

