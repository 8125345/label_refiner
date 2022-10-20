# 数据集整体数量控制，充分利用被下采样的数据集
mul_times = 1
# export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels
# # export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels/label_refiner
# export PYTHONPATH=$PYTHONPATH:/data/projects/LabelModels/label_refiner/models

# /Users/xyz/PycharmProjects/XYZ_projects/LabelModels/label_refiner
# 训练时使用的数据集
train_dataset = [
    ("maestro_new", 1. * mul_times),  # 70419
]
# 基础配置，如果train_config中存在同名变量，会用train_config变量覆盖basic_config配置
basic_config = {
    "model_root": "/data/projects/LabelModels/lag_prediction/train_output_models",
    "train_log_root": "/data/projects/LabelModels/lag_prediction/train_log",
    "train_comment_root": "/data1/projects/LabelModels/lag_prediction/train_comment",
}

# ==================================================================================
# ==================================================================================

# 本次训练配置
train_config = {
    "model_name": f"cornet_20220923_0",
    "gpuids": [2],
    "train_batchsize": 32,
    "val_batchsize": 32,
    "add_maestro": True,
    "pos_weight": 1,
    "pretrain_model": "/data/projects/LabelModels/lag_prediction/train_output_models/cornet.h5",
    "model_structure": "cornet",
    "lr": 0.001 / (1),  # 初始学习率
    "dataset_path": "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220916_0_train",  # todo，先使用次数据测试
    "rec_loss_fun": "weighted_bce",
    "comment": "频谱和midi延时预测模型",
}
