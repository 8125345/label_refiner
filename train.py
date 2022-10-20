"""
强制拼接模型
使用新数据集，数据使用大块存储
取消reducelr和early stop，模型稳定后再使用这个
todo 为什么大batch训练会出错？当batch size 设置为128 * 20时，验证集日志结果异常（auc=0.5，val_loss变化极慢），当设为128时正常
"""
import sys
sys.path.insert(0, '/data/projects/LabelModels/label_refiner')

import json
import os.path
import sys
import atexit

import tensorflow as tf
from train_config import basic_config, train_config, train_dataset

# 合并配置
basic_config.update(train_config)
train_config = basic_config

# GPU配置策略
gpuids = train_config["gpuids"]  # 显卡选择，支持多卡[0, 2, 3]
distributed_flag = False
distributed_strategy = None
# 调试的时候注意，如果调起GPU后，后面没有其他代码，会报NoneType错误，框架特性并非BUG
if gpuids:
    if len(gpuids) > 1:
        distributed_flag = True
    print(f"使用GPU:{gpuids}")
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        # gpu_id = gpuid
        try:
            gpu_list = list()
            for gpu_index in gpuids:
                gpu_list.append(gpus[gpu_index])
            tf.config.experimental.set_visible_devices(gpu_list, 'GPU')
            for gpu_index in gpuids:
                tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")

            distributed_strategy = tf.distribute.MirroredStrategy()
            print('Number of devices: {}'.format(distributed_strategy.num_replicas_in_sync))

        except RuntimeError as e:
            print(e)
    else:
        raise (f"GPU状态异常，未检测到GPU。期望启动GPU{gpuids}")
else:
    print("只使用CPU")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

from tensorflow import keras
from data_process.dataset_maker import make_dataset
from label_refiner.models.models import custom_load_model

# ========================================
# 配置参数
epochs = 1000000  # 无限训练
# ========================================
# 模型参数
commit = """
强制拼接模型
"""
# debug = True  # 调试模式使用最小buffer_size，加快数据读取
debug = False  # 正常训练模式

if debug:
    comfirm = input("现在是dubug模式，请确认是否执行（y/n）\n")
    if comfirm == "y":
        pass
    else:
        print("退出程序")
        sys.exit()
train_batchsize = train_config["train_batchsize"]
val_batchsize = train_config["val_batchsize"]

if debug:
    train_batchsize = 128 * 1

model_root = train_config["model_root"]  # 模型保存路径
assert os.path.exists(model_root)
train_log_root = train_config["train_log_root"]  # 日志保存路径
assert os.path.exists(train_log_root)
train_comment_root = train_config["train_comment_root"]  # 实验说明保存路径
assert os.path.exists(train_comment_root)

root_path = train_config["dataset_path"]  # 数据集路径
assert os.path.exists(root_path)
model_name = train_config["model_name"]  # 本次实验模型名称
add_maestro = train_config["add_maestro"]  # 【待调整】是否添加meastro数据集
pretrain_model = train_config["pretrain_model"]  # 预训练模型路径
pos_weight = train_config["pos_weight"]  # 正样本权重
lr = train_config["lr"]  # 初始学习率
rec_loss_fun = train_config["rec_loss_fun"]  # 损失函数
model_structure = train_config["model_structure"]  # 模型结构
train_comment = str(train_config)  # 直接对当前配置做快照，保存下来
dataset_name_list = train_dataset  # 数据集列表名称

# 用于控制数据集生成器使用什么方式切分数据
if model_structure == "label_refiner_double_crnn":
    data_split_mode = "default"

else:
    raise Exception("模式错误")

# ========================================
train_tfrecord_path_list = list()  # 训练数据
val_tfrecord_path_list = list()  # 验证数据
dataset_info_list = list()  # 数据集信息，包含数据集所含元素数量

# 根据规则过滤数据集
num_train_samples = 0
num_valid_samples = 0
for dataset_item in dataset_name_list:

    dataset_name, ratio = dataset_item

    # if (dataset_name != "maestro") or (dataset_name == "maestro" and add_maestro is True):
    # 由于历史实验表明maestro数据集和其他实验同时进行时效果会下降，因此单独针对是否添加maestro做控制
    if ("maestro" not in dataset_name) or ("maestro" in dataset_name and add_maestro is True):
        train_path = os.path.join(root_path, f"train_{dataset_name}.tfrecord")
        val_path = os.path.join(root_path, f"val_{dataset_name}.tfrecord")
        dataset_info_path = os.path.join(root_path, f"dataset_info_{dataset_name}.json")
        assert os.path.exists(train_path)
        assert os.path.exists(val_path)
        assert os.path.exists(dataset_info_path)

        with open(dataset_info_path, "r") as f:
            dataset_info_dict = json.load(f)
        num_train_samples += dataset_info_dict["train_num"]
        num_valid_samples += dataset_info_dict["val_num"]

        train_tfrecord_path_list.append({
            "path": train_path, "ratio": ratio, "num": dataset_info_dict["train_num"],
        })
        val_tfrecord_path_list.append({
            "path": val_path, "ratio": ratio, "num": dataset_info_dict["val_num"],
        })

model_path = os.path.join(model_root, model_name + ".h5")
train_log_path = os.path.join(train_log_root, f"train_log_{model_name}" + ".csv")  # 记录训练结果
train_comment_path = os.path.join(train_comment_root, f"train_comment_{model_name}" + ".txt")  # 记录本次训练配置

# 避免之前实验结果被覆盖
if os.path.exists(train_comment_path):
    comfirm = input(f"实验{train_comment_path}已存在，是否覆盖实验结果？（y/n）\n")
    if comfirm == "y" or comfirm == "Y":
        print("继续程序，覆盖之前记录")
    else:
        print("退出程序")
        sys.exit()

# 保存本次实验说明
with open(train_comment_path, "w") as f:
    f.writelines(train_comment)
    #
    f.writelines(str(dataset_name_list))
# ========================================
# 模型配置参数
model_kwarg = {
    "lr": lr,
    "compile_pretrain": False,
    "pos_weight": pos_weight,
    "rec_loss_fun": rec_loss_fun,
    "model_structure": model_structure,
}

# ========================================

# 使用/不使用GPU时模型加载
if distributed_flag:
    assert distributed_strategy is not None
    with distributed_strategy.scope():
        model = custom_load_model(pretrain_model, **model_kwarg)
else:
    model = custom_load_model(pretrain_model, **model_kwarg)


# 保存退出时额外保存一次模型
@atexit.register
def save_model_before_exit():
    exit_model_path = os.path.join(model_root, model_name + "_exit" + ".h5")
    print(f"保存临退出时最后模型: {exit_model_path}")
    model.save(exit_model_path)
    print(f"模型保存完成")


print(f"损失函数:{rec_loss_fun}")
print(f"模型保存路径：{model_path}")
print(f"使用gpu:{gpuids}")
print(commit)

if debug:
    buffer_size = 1  # 便于快速测试
    print(f"调试模式, buffer_size: {buffer_size}")
else:
    buffer_size = num_train_samples  # 全量shuffle训练数据

# 数据集构建
train_dataset = make_dataset(train_tfrecord_path_list, epochs, "train", batchsize=train_batchsize, shuffle=True,
                             buffer_size=buffer_size,
                             distributed_flag=distributed_flag,
                             distributed_strategy=distributed_strategy,
                             data_split_mode=data_split_mode,
                             )
val_dataset = make_dataset(val_tfrecord_path_list, 1, "val", batchsize=val_batchsize, shuffle=False,
                           distributed_flag=distributed_flag,
                           distributed_strategy=distributed_strategy,
                           data_split_mode=data_split_mode,
                           )

print(f"数据集路径:{root_path}")
print(f"训练集数据量：{num_train_samples}")
print(f"验证集数据量：{num_valid_samples}")
print(f"buffer数据集总量:{buffer_size}")
print(f"实验内容：{train_comment}")

# 回调函数
callbacks = [
    keras.callbacks.ModelCheckpoint(model_path, save_weights_only=False, save_best_only=True),
    # keras.callbacks.EarlyStopping(patience=es_patience),  # 谨慎使用
    # keras.callbacks.ReduceLROnPlateau(factor=rlr_factor, patience=rlr_patience, verbose=1), # 谨慎使用
    keras.callbacks.CSVLogger(train_log_path, separator=",", append=False),
    # keras.callbacks.LearningRateScheduler(scheduler)
]

# 训练模型
history = model.fit(
    train_dataset,
    steps_per_epoch=num_train_samples // train_batchsize,
    validation_data=val_dataset,
    validation_steps=num_valid_samples // val_batchsize,
    epochs=epochs,
    callbacks=callbacks,
)
print(history.history)
