"""
声音识别模型数据加载脚本，无数据增强
输出label 8
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import math
import tensorflow as tf

seg_frame_num = 640  # 单条长序列中帧长度
n_mels = 229  # mel频谱bank数量
key_num = 88  # 琴键数量

chunk_frame_num = 320  # 模型输入帧长度
# midi_frame_num = 320  # 模型输出midi序列长度
chunk_in_seg_num = int(seg_frame_num / chunk_frame_num)  # 长序列中chunk的数量

rng = tf.random.Generator.from_seed(123, alg='philox')


def _parse_function(example_proto):
    """
    解析一条tfrecord数据
    :param example_proto:
    :return:
    """

    features = tf.io.parse_single_example(
        example_proto,
        features={
            'path': tf.io.FixedLenFeature([], tf.string),
        }
    )
    return features


def load_and_parse_data(path):
    """
    加载并解析数据
    :param path:
    :return:
    """

    serialized = tf.io.read_file(path)
    data = tf.io.parse_tensor(serialized, tf.float32)

    return data


def split_feature_label(data):
    """
    标注校准算法数据切分，使用ground true label动态生成输入数据。
    :param data:
    :return:
    """
    seed = rng.make_seeds(2)[0]
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    mel_feature = data[:, n_mels: 2 * n_mels]  # 获得音频特征
    label = data[:, n_mels * 2:]  # n * 88  获得标签

    # 对标签做数据增强
    # 1. 随机偏移，模拟音频和数据对齐误差
    label_feature = tf.pad(label, tf.constant([[5, 5], [0, 0]]), mode='CONSTANT', constant_values=0)
    label_feature = tf.image.stateless_random_crop(
        value=label_feature,
        size=(chunk_frame_num, key_num),
        seed=new_seed)

    # 2. 随机缩放+pad，模拟midi拉长缩短误差问题。
    label_feature = tf.expand_dims(label_feature, 2)  # 增加channel，避免resize api报错
    scale = tf.random.uniform([], minval=1 - 0.009375, maxval=1 + 0.009375, dtype=tf.float32)  # 每320帧伸长或者缩短3帧
    label_feature = tf.image.resize(
        label_feature, [tf.cast(chunk_frame_num * scale, tf.int32), key_num],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # todo 避免羽化，是否应该用NEAREST_NEIGHBOR？  tf.cast([chunk_frame_num * scale, key_num], tf.int32)
    # 处理数据到指定长度
    label_feature = tf.image.resize_with_crop_or_pad(
        label_feature, chunk_frame_num, key_num
    )
    label_feature = tf.squeeze(label_feature, 2)
    pos_cnt = tf.math.count_nonzero(label_feature)
    pos_positions = tf.where(label_feature)

    # 3. 随机去除标记，去掉10%，模拟标注FN
    # todo 有的label含有两帧，未来需要同时移动
    drop_num = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=0.2) * tf.cast(pos_cnt, tf.float32),
                       tf.int32)  # 丢弃数量最多为正样本的10%
    idxs = tf.range(tf.shape(pos_positions)[0])
    ridxs = tf.random.shuffle(idxs)[:drop_num]
    drop_idxs = tf.gather(pos_positions, ridxs)
    label_feature = tf.tensor_scatter_nd_update(label_feature, drop_idxs, tf.zeros(drop_num))  # 在随机坐标位置置1

    # 4. 随机增加噪声，比gt多10%，模拟标注FP。此处pos_cnt为真是pos数量
    noise_num = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=0.1) * tf.cast(pos_cnt, tf.float32),
                        tf.int32)  # 噪声数量最多为正样本的10%
    rand_pos = tf.random.uniform(
        shape=[noise_num, 2],
        minval=0,
        maxval=1)  # 生成随机坐标
    rand_pos = tf.cast(rand_pos * [chunk_frame_num, key_num], tf.int32)
    label_feature = tf.tensor_scatter_nd_update(label_feature, rand_pos,
                                                tf.ones(tf.cast(noise_num, tf.int32)))  # 在随机坐标位置置1
    feature_dict = {
        "input_1": mel_feature,  # 频谱图
        "input_2": label_feature  # 噪声标签
    }
    return feature_dict, label


def concat_data(data):
    """
    单条数据维度调整，便于之后数据增强
    :param data:
    :return:
    """

    feature = data[:, : 293120]
    midi = data[:, 293120:]

    # print("特征标签", tf.shape(feature)[0])  # , midi.shape

    feature = tf.reshape(feature, (640, 229, 2))
    midi = tf.reshape(midi, (640, 88))
    concat = tf.concat((feature[:, :, 0], feature[:, :, 1], midi), axis=1)  # 640 * (229 + 229 + 88)
    # print("concat_data", concat)
    return concat


def parse_data_val(example_proto):
    """
    验证集中仅使用非增强数据
    :param example_proto:
    :return:
    """
    features = _parse_function(example_proto)
    path = features["path"]
    # seed = rng.make_seeds(2)[0]
    # new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    data = load_and_parse_data(path)
    data = concat_data(data)  # 调整数据维度，便于下一步裁剪
    # # 随机数据裁剪，每次取用不同时间范围的数据
    # data = tf.image.stateless_random_crop(value=data,
    #                                       size=(seg_frame_num - chunk_frame_num, n_mels * 2 + key_num),
    #                                       seed=new_seed)

    # 单条数据扩增为多个短片段，验证集数据没有使用增强
    chunks = tf.reshape(data, (chunk_in_seg_num, chunk_frame_num, n_mels * 2 + key_num))
    # print("parse_data_val chunks", chunks.shape)
    return chunks


def parse_data_train(example_proto):
    """
    训练数据加载和动态增强
    :param example_proto:
    :return:
    """
    features = _parse_function(example_proto)
    path = features["path"]
    seed = rng.make_seeds(2)[0]
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]

    data = load_and_parse_data(path)
    data = concat_data(data)  # 调整数据维度，便于下一步裁剪
    chunks = tf.reshape(data, (chunk_in_seg_num, chunk_frame_num, n_mels * 2 + key_num))  # 无数据增强

    # # 随机数据裁剪，每次取用不同时间范围的数据
    # data = tf.image.stateless_random_crop(value=data,
    #                                       size=(seg_frame_num - chunk_frame_num, n_mels * 2 + key_num),
    #                                       seed=new_seed)
    # # 单条数据扩增为多个短片段
    # chunks = tf.reshape(data, (chunk_in_seg_num - 1, chunk_frame_num, n_mels * 2 + key_num))
    # # print("parse_data_train chunks", chunks.shape)
    return chunks


def make_dataset(tfrecord_path_list, epoch, dataset_type, batchsize=32, shuffle=False, buffer_size=1000,
                 distributed_flag=False, distributed_strategy=None, data_split_mode="default"):
    """
    数据集中包含增强信息
    :param distributed_strategy: 多卡策略
    :param distributed_flag: 是否多卡训练
    :param dataset_type: 数据集类型["train", "val"]
    :param tfrecord_path_list: 数据集列表
    :param epoch:
    :param batchsize:
    :param shuffle:
    :param buffer_size:
    :param data_split_mode:

    :return:
    """
    assert dataset_type in ["train", "val"]
    assert isinstance(tfrecord_path_list, list)
    assert data_split_mode in ["default"]

    # tfrecord_path_list中的每个字典记录路径和比例{path, ratio}
    dataset_list = list()

    # 这部分代码用来控制数据采样，可以过采样或者欠采样
    for item in tfrecord_path_list:
        dataset_path = item["path"]
        dataset_ratio = item["ratio"]
        dataset_num = item["num"]
        assert 0 <= dataset_ratio

        single_dataset = tf.data.TFRecordDataset(dataset_path)
        # 使用数据集采样比例对数据集进行采样
        # cardinality = single_dataset.cardinality().numpy()
        # assert cardinality > 0
        cardinality = dataset_num

        take_cnt = int(cardinality * dataset_ratio)  # 获得数据集总数
        repeat_cnt = int(math.ceil(dataset_ratio))  # 向上取整

        # dataset_ratio设为0代表不使用此数据集，直接跳过，避免可能存在的异常
        if dataset_ratio > 0:
            single_dataset = single_dataset \
                .repeat(repeat_cnt) \
                .take(take_cnt)

            dataset_list.append(single_dataset)

        print(f"type: {dataset_type}\tdataset: {dataset_path}\ttotal: {cardinality}\tsampled: {take_cnt}")

    # 合并多个数据集
    ds = tf.data.Dataset.from_tensor_slices(dataset_list)
    dataset = ds.interleave(lambda x: x, cycle_length=1, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        print(f"shuffle buffer_size:{buffer_size}")
        dataset = dataset.shuffle(buffer_size=buffer_size)

    if data_split_mode == "default":
        parse_data_train_fun = parse_data_train
        parse_data_val_fun = parse_data_val
    else:
        pass

    # 训练数据集会随机切片增强，测试数据集不做增强
    if dataset_type == "train":
        dataset = dataset.map(
            parse_data_train_fun,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=64  # tf默认只使用一半cpu
        )
    else:
        dataset = dataset.map(
            parse_data_val_fun,
            # num_parallel_calls=tf.data.experimental.AUTOTUNE,
            num_parallel_calls=64
        )

    # 把切分后的数据合并到batch维度
    dataset = dataset.unbatch()

    if data_split_mode == "default":
        split_fun = split_feature_label
    else:
        pass

    # 把数据处理为训练所需shape
    dataset = dataset.map(
        split_fun,
        # num_parallel_calls=tf.data.experimental.AUTOTUNE,
        num_parallel_calls=64  # tf默认只使用一半cpu
    )

    dataset = dataset.batch(batchsize)
    dataset = dataset.prefetch(
        tf.data.experimental.AUTOTUNE
    )
    dataset = dataset.repeat(count=epoch)

    # 多卡策略
    if distributed_flag:
        options = tf.data.Options()
        options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
        # options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
        dataset = dataset.with_options(options)
        dataset = distributed_strategy.experimental_distribute_dataset(dataset)

    return dataset


if __name__ == '__main__':
    pass
    # item[0][0, :, :, 1].numpy()
    # item[1][0, :, :].numpy()

    # # 抽检数据
    # root_path = "/data1/dataset/audio/rec_dataset/dataset_info_concat/dataset_20220826_0_train"
    # assert os.path.exists(root_path)
    # train_tfrecord_path = os.path.join(root_path, "train_ai-tagging.tfrecord")
    # val_tfrecord_path = os.path.join(root_path, "val_ai-tagging.tfrecord")
    #
    # train_dataset = make_dataset([train_tfrecord_path], 2, "train", batchsize=4, shuffle=False, buffer_size=100,
    #                              data_split_mode="single_input_midi")
    # val_dataset = make_dataset([val_tfrecord_path], 1, "val", batchsize=128 * 20, shuffle=False, buffer_size=100,
    #                            data_split_mode="single_input_midi")
    #
    # # for item in train_dataset:
    #
    # cnt = 0
    # for item in val_dataset:
    #     idx_in_batch = 0
    #     item0_bgm = item[0][idx_in_batch, :, :, 0].numpy()
    #     item0_mix = item[0][idx_in_batch, :, :, 1].numpy()
    #
    #     print(np.array_equal(item0_bgm, item0_mix))
    #     print(np.any(item[1][idx_in_batch, :, :].numpy()))
    #
    #     # 一个batch
    #     # print(item)
    #     print(1)
    #     # break
    #     cnt += 1
    #
    # print(cnt)
