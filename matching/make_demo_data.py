import os

import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import json
import tensorflow as tf
import numpy
import imageio

seg_frame_num = 640  # 单条长序列中帧长度
n_mels = 229  # mel频谱bank数量
key_num = 88  # 琴键数量

chunk_frame_num = 640  # 模型输入帧长度
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


def split_data(data):
    """
    解析数据
    :param data:
    :return:
    """

    feature = data[:, : 293120]
    midi = data[:, 293120:]

    # feature = tf.reshape(feature, (640, 229, 2))
    midi = tf.reshape(midi, (640, 88))

    return midi


def aug_data(data):
    label = data
    # label_feature = data  # todo debug

    seed = rng.make_seeds(2)[0]
    new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
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
    drop_num = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=0.1) * tf.cast(pos_cnt, tf.float32),
                       tf.int32)  # 丢弃数量最多为正样本的10%
    idxs = tf.range(tf.shape(pos_positions)[0])
    ridxs = tf.random.shuffle(idxs)[:drop_num]
    drop_idxs = tf.gather(pos_positions, ridxs)
    label_feature = tf.tensor_scatter_nd_update(label_feature, drop_idxs, tf.zeros(drop_num))  # 在随机坐标位置置1

    # 4. 随机增加噪声，比gt多10%，模拟标注FP。此处pos_cnt为真是pos数量
    noise_num = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=0.1) * tf.cast(pos_cnt, tf.float32),
                        tf.int32)  # 噪声数量最多为正样本的10%
    print(noise_num)
    rand_pos = tf.random.uniform(
        shape=[noise_num, 2],
        minval=0,
        maxval=1)  # 生成随机坐标
    rand_pos = tf.cast(rand_pos * [chunk_frame_num, key_num], tf.int32)
    label_feature = tf.tensor_scatter_nd_update(label_feature, rand_pos,
                                                tf.ones(tf.cast(noise_num, tf.int32)))  # 在随机坐标位置置1

    return label_feature


def get_point_mat():
    # 获得json
    json_path = "/deepiano_data/yuxiaofei/work/data_0718/serialize_changpu_delay_metronome/ai-tagging_train.json"
    with open(json_path, "r") as f:
        data = json.load(f)
    # ".serialized"数据
    test_index = 3  # 随机选择测试片段
    serialized_path = data[str(test_index)]  # 序列化数据路径
    data = load_and_parse_data(serialized_path)
    # 反序列标签数据到矩阵
    label_mat = split_data(data)
    # check_np = label_mat.numpy()
    # 生成随机增强的数据
    noise_label_mat = aug_data(label_mat)

    return label_mat, noise_label_mat


def get_coord_form_mat(mat_data):
    xs, ys = np.where(mat_data == 1)
    xs = xs.reshape((-1, 1))
    ys = ys.reshape((-1, 1))

    return np.hstack((xs, ys))

def get_point_demo():
    label_mat, noise_label_mat = get_point_mat()
    coord_label = get_coord_form_mat(label_mat)
    coord_noise_label = get_coord_form_mat(noise_label_mat)


    # 下面用来检查数据
    # 根据后续操作反转样本
    label_mat = 1 - label_mat
    noise_label_mat = 1 - noise_label_mat

    # 保存图片
    imageio.imwrite("data/label.png", label_mat)
    imageio.imwrite("data/noise_label.png", noise_label_mat)


    return coord_label, coord_noise_label


def gen_geomloss_data():
    # 生成geomloss用的数据
    label_mat, noise_label_mat = get_point_mat()

    # 根据后续操作反转样本
    label_mat = 1 - label_mat
    noise_label_mat = 1 - noise_label_mat

    # 补边
    row_num, col_num = label_mat.shape

    zero_pad = np.ones((row_num, row_num - col_num))
    print(label_mat.shape)
    print(noise_label_mat.shape)

    # 保存图片
    imageio.imwrite("data/label.png", np.concatenate((label_mat, zero_pad), axis=1))

    test = np.concatenate((noise_label_mat, np.ones((row_num, row_num - col_num - 100))), axis=1)
    test = np.concatenate((np.ones((row_num, 100)), test), axis=1)

    imageio.imwrite("data/noise_label.png", test)


if __name__ == '__main__':
    # gen_geomloss_data()  # 生成geomloss用的数据
    # coord_label, coord_noise_label = get_point_demo()  # 生成ICP用的数据
    a = 1
