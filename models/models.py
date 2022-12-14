import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from tensorflow.keras.layers import Conv2D, Input, Add, Multiply, UpSampling2D, Conv2DTranspose, DepthwiseConv2D, \
    Dropout, MaxPooling2D, Concatenate, Lambda, Reshape, LayerNormalization, Resizing, AveragePooling2D
from tensorflow.keras.models import Model
from keras import backend
from keras.applications import imagenet_utils

from keras.activations import softmax
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K


# from models.metrics import cAUC, cPrecision, cRecall


def relu(x):
    return layers.ReLU()(x)


def hard_sigmoid(x):
    return layers.ReLU(6.)(x + 3.) * (1. / 6.)


def hard_swish(x):
    return layers.Multiply()([x, hard_sigmoid(x)])


def conv_bn_act(x, filters, kernel_size, stride, activation, block_id):
    prefix = 'conv_bn_act_{}/'.format(block_id)
    x = layers.Conv2D(
        filters,
        kernel_size=kernel_size,
        strides=stride,
        padding='same',
        use_bias=False,
        name=prefix + 'expand')(
        x)
    x = layers.BatchNormalization(
        epsilon=1e-3,
        momentum=0.999,
        name=prefix + 'expand/BatchNorm')(
        x)
    if activation is not None:
        x = activation(x)
    return x


def block1(x, filters, kernel_size=3, stride=1, conv_shortcut=True, name=None):
    """A residual block.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=stride, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=stride, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = block1(x, filters, stride=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = block1(x, filters, conv_shortcut=False, name=name + '_block' + str(i))
    return x


def bilstm_decoder(x, units=256):
    # RNNs.
    x = keras.layers.Bidirectional(
        keras.layers.LSTM(units, return_sequences=True), name="cudnn_lstm"
    )(x)

    return x


def double_crnn(audio_feature_shape, noise_label_feature_shape, save_path=None):
    # -----------------------------------------
    # ?????????????????????
    audio_feature_input = keras.Input(shape=audio_feature_shape)  # ????????????
    noise_label_feature_input = keras.Input(shape=noise_label_feature_shape)  # ????????????

    # ????????????
    # audio_feature = inputs[:, :, :, 0:1]
    _, row, col, ch = tf.shape(audio_feature_input)
    # audio_feature = Lambda(lambda x: x)(audio_feature_input)
    audio_feature = tf.image.resize(audio_feature_input, (row, 256))  # ??????????????????????????????????????????

    # label??????
    _, row, col, ch = tf.shape(noise_label_feature_input)
    # noise_label_feature = Lambda(lambda x: x)(noise_label_feature_input)
    noise_label_feature = tf.image.resize(noise_label_feature_input, (row, 96))  # ?????????????????????

    # -----------------------------------------
    # ??????????????????
    # resnetv1 50
    x = conv_bn_act(audio_feature, 64, 7, (1, 2), relu, "conv1")

    def stack_fn(x):
        x = stack1(x, 64, 3, stride1=1, name='conv2')
        x = stack1(x, 96, 4, stride1=(1, 2), name='conv3')
        x = stack1(x, 144, 6, stride1=(1, 2), name='conv4')
        return x

    x = stack_fn(x)  # ???????????? 640 32
    # -----------------------------------------
    # ??????????????????
    x_n = conv_bn_act(noise_label_feature, 64, 7, (1, 3), relu, "label_conv1")  # ???????????????????????????
    # -----------------------------------------
    # ???????????????
    x = Concatenate()([x, x_n])  # 208
    x = conv_bn_act(x, 312, 3, (1, 2), relu, "conv5_0")  # 16

    # ??????????????????????????????????????????????????????midi??????
    name = "conv5_h"
    x_h = block1(x, 312, stride=(2, 2), name=name + '_block1')
    x_h = block1(x_h, 624, stride=(2, 2), name=name + '_block2')

    x_h = UpSampling2D((2, 2))(x_h)
    x_h = block1(x_h, 312, stride=(1, 1), name=name + '_block3')
    x_h = UpSampling2D((2, 2))(x_h)
    x_h = block1(x_h, 156, stride=(1, 1), name=name + '_block4')

    # ????????????????????????????????????
    x_l = stack1(x, 156, 3, stride1=(1, 1), name='conv5_l')

    x = Concatenate()([x_h, x_l])  # 312
    x = stack1(x, 512, 3, stride1=(1, 2), name='conv6')  # 8
    # x = conv_bn_act(x, 256, 1, 1, relu, "conv6")  # ???????????????

    # -----------------------------------------

    _, row, col, ch = backend.int_shape(x)
    new_shape = (-1, col * ch)
    x = keras.layers.Reshape(target_shape=new_shape, name="flatten_end")(x)
    x = keras.layers.Dense(768, activation="relu", name="fc_end")(x)  # ????????????????????????
    lstm_output = bilstm_decoder(x, units=512)
    output_midi = keras.layers.Dense(
        88, activation="sigmoid", name="onset_probs"
    )(lstm_output)

    model = Model(inputs=[audio_feature_input, noise_label_feature_input], outputs=[output_midi],
                  name=f"label_refiner_double_crnn")

    if save_path is not None:
        print(f"??????????????????{save_path}")
        model.save(save_path)
    return model


def weighted_bce(weight):
    """
    ???????????????,??????output????????????sigmoid??????????????????[0, 1]
    :param weight:
    :return:
    """
    _EPSILON = 1e-7

    def loss(target, output):
        # ??????????????????????????????????????????????????????????????????
        # https://github.com/keras-team/keras/blob/5a7a789ee9766b6a594bd4be8b9edb34e71d6500/keras/backend/tensorflow_backend.py#L3275
        # if not from_logits transform back to logits
        _epsilon = tf.convert_to_tensor(_EPSILON, output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1 - _epsilon)
        output = tf.math.log(output / (1 - output))

        # return tf.nn.sigmoid_cross_entropy_with_logits(labels=target,
        #                                                logits=output)
        # https://tensorflow.google.cn/api_docs/python/tf/nn/weighted_cross_entropy_with_logits
        # return tf.nn.weighted_cross_entropy_with_logits(labels=target,
        #                                                 logits=output,
        #                                                 pos_weight=weight,
        #                                                 name=None)
        return tf.math.reduce_mean(
            tf.nn.weighted_cross_entropy_with_logits(
                labels=target,
                logits=output,
                pos_weight=weight,
                name=None))

    return loss


def custom_load_model(pretrain_model, lr=0.0001, compile_pretrain=False, pos_weight=1,
                      rec_loss_fun="weighted_bce", model_structure="label_refiner_double_crnn",
                      first_decay_steps=20000):
    """
    ????????????
    :param rec_loss_fun: ??????????????????????????????
    :param lr: ???????????????
    :param compile_pretrain: ?????????????????????????????????????????????????????????
    :param pretrain_model: ???????????????????????????None
    :return:
    """
    if rec_loss_fun == "weighted_bce":
        # ???????????????bce
        rec_loss = weighted_bce(pos_weight)
    elif rec_loss_fun == "bce":
        # bce
        rec_loss = "binary_crossentropy"
    else:
        raise Exception(f"???????????????{rec_loss_fun}?????????")

    # ???????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
    assert pretrain_model is not None
    print(f"?????????????????????:{pretrain_model}")
    # ??????????????????????????????????????????????????????

    model = load_model(pretrain_model, compile=False)
    lr_decayed_fn = (tf.keras.optimizers.schedules.CosineDecayRestarts(lr, first_decay_steps))

    adam = tf.keras.optimizers.Adam(learning_rate=lr_decayed_fn)
    opt = adam

    loss = rec_loss
    loss_weights = 1

    # label_range = None  # (0, 8)  # ??????????????????????????????label?????????label?????????????????????[0, 8)
    # metric_range = (3, 5)  # ???????????????????????????
    #
    # metrics = [
    #     # ????????????
    #     cAUC(label_range=label_range, metric_range=metric_range),
    #     cPrecision(label_range=label_range, metric_range=metric_range),
    #     cRecall(label_range=label_range, metric_range=metric_range),
    #     cRecall(thresholds=0.1, label_range=label_range, metric_range=metric_range)
    # ]
    metrics = [
        tf.keras.metrics.AUC(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall(),
        tf.keras.metrics.Recall(thresholds=0.1),
    ]

    model.compile(
        optimizer=opt,
        loss=loss,
        loss_weights=loss_weights,
        metrics=metrics,
    )
    model.summary()
    return model


if __name__ == '__main__':
    # ????????????
    pretrain_model = "/data/projects/LabelModels/spliter_detector/train_output_models/label_refiner_double_crnn_2.h5"
    # model = custom_load_model(
    #     pretrain_model, lr=0.0001, compile_pretrain=False, pos_weight=1,
    #     rec_loss_fun="weighted_bce", model_structure="label_refiner_double_crnn",
    #     first_decay_steps=20000)

    # ??????????????????
    model = double_crnn(
        audio_feature_shape=(None, 229, 1),
        noise_label_feature_shape=(None, 88, 1),
        save_path=pretrain_model,
    )
    model.summary()

    import numpy as np

    a_f = np.random.random((1, 480, 229, 1))
    n_f = np.random.random((1, 480, 88, 1))

    out = model.predict([a_f, n_f])

    print(out.shape)
