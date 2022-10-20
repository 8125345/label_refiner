import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np

chunk_frame_num = 4
key_num = 3
label_feature = tf.constant([
    [1, 0, 1],
    [1, 1, 2],
    [0, 1, 3],
    [0, 0, 4]
], dtype=tf.float32)
pos_cnt = tf.math.count_nonzero(label_feature)
pos_positions = tf.where(label_feature)

print(f"pos_cnt:{pos_cnt}")

# 3. 随机去除标记，去掉10%，模拟标注FN
# todo 有的label含有两帧，未来需要同时移动
drop_num = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=0.5) * tf.cast(pos_cnt, tf.float32), tf.int32)  # 丢弃数量最多为正样本的10%

print(drop_num)
idxs = tf.range(tf.shape(pos_positions)[0])
ridxs = tf.random.shuffle(idxs)[:drop_num]
drop_idxs = tf.gather(pos_positions, ridxs)
label_feature = tf.tensor_scatter_nd_update(label_feature, drop_idxs, tf.ones(drop_num) * 777)  # 在随机坐标位置置1
print(label_feature)


# 4. 随机增加噪声，比gt多10%，模拟标注FP。此处pos_cnt为真是pos数量
noise_num = tf.cast(tf.random.uniform(shape=[], minval=0, maxval=0.5) * tf.cast(pos_cnt, tf.float32),
                    tf.int32)  # 噪声数量最多为正样本的10%
rand_pos = tf.random.uniform(
    shape=[noise_num, 2],
    minval=0,
    maxval=1)  # 生成随机坐标
rand_pos = tf.cast(rand_pos * [chunk_frame_num, key_num], tf.int32)
label_feature = tf.tensor_scatter_nd_update(label_feature, rand_pos,
                                            tf.ones(tf.cast(noise_num, tf.int32)) * 666)  # 在随机坐标位置置1
print(noise_num)
print(label_feature)

# ran_num = tf.random.uniform(shape=[], minval=0, maxval=1)
# ran_num = tf.cast(ran_num * 4, tf.int32)  # 确定噪声数量
#
# print(f"噪声数量{ran_num}")
# rand_arr = tf.random.uniform(
#     shape=[ran_num, 2],
#     minval=0,
#     maxval=1)
# rand_arr = rand_arr * [4, 3]
# rand_arr = tf.cast(rand_arr, tf.int32)
#
# print(rand_arr)
#
# # 更新tensor权重
# # updates
# ans = tf.tensor_scatter_nd_update(tensor, rand_arr, tf.zeros(ran_num))
# print(ans)
# # rng = tf.random.Generator.from_seed(123, alg='philox')
# # label = tf.constant([[1, 2, 3], [4, 5, 6]])
# #
# # for i in range(10):
# #     seed = rng.make_seeds(2)[0]
# #     new_seed = tf.random.experimental.stateless_split(seed, num=1)[0, :]
# #     label_feature = tf.pad(label, tf.constant([[2, 2], [0, 0]]), mode='CONSTANT', constant_values=-100)
# #     label_feature = tf.image.stateless_random_crop(
# #         value=label_feature,
# #         size=(2, 3),
# #         seed=new_seed)
# #     print(label_feature)
#
#
# # t = tf.constant([[1, 2, 3], [4, 5, 6]])
# # paddings = tf.constant([[0, 1, ], [2, 3]])
# # label_feature = tf.pad(t, paddings, mode='CONSTANT', constant_values=-100)
# # print(label_feature)
