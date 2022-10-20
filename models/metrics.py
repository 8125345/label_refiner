"""
重写keras评估指标方法，支持y_true和y_predict不等长
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# import abc
# import types
# from typing import List, Tuple, Union
# import warnings
#
from keras import activations
# from keras import backend
# from keras.engine import base_layer
# from keras.engine import base_layer_utils
# from keras.engine import keras_tensor
# from keras.losses import binary_crossentropy
# from keras.losses import categorical_crossentropy
# from keras.losses import categorical_hinge
# from keras.losses import hinge
# from keras.losses import kullback_leibler_divergence
# from keras.losses import logcosh
# from keras.losses import mean_absolute_error
# from keras.losses import mean_absolute_percentage_error
# from keras.losses import mean_squared_error
# from keras.losses import mean_squared_logarithmic_error
# from keras.losses import poisson
# from keras.losses import sparse_categorical_crossentropy
# from keras.losses import squared_hinge
# from keras.saving.saved_model import metric_serialization
# from keras.utils import generic_utils
# from keras.utils import losses_utils
from keras.utils import metrics_utils
# from keras.utils.generic_utils import deserialize_keras_object
# from keras.utils.generic_utils import serialize_keras_object
# from keras.utils.generic_utils import to_list
# from keras.utils.tf_utils import is_tensor_or_variable
# import numpy as np
# import tensorflow.compat.v2 as tf
#
# from tensorflow.python.util.tf_export import keras_export
# from tensorflow.tools.docs import doc_controls

import tensorflow as tf


# metrics = [tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]

def clip_fun(x, x_range):
    return x[:, x_range[0]: x_range[1], :]  # lstm batch, frame, keys
    # return x[:, x_range[0]: x_range[1], :, :]  # cnn


def clip_input(x, input_range):
    # 把y_true裁剪为和y_pred一样的维度
    if input_range is None:
        return x
    else:
        if len(input_range) != 2 or input_range[1] <= input_range[0]:
            raise ValueError(f"input_range错误，当前：{input_range}")
        else:
            # todo lstm
            # x = x[:, input_range[0]: input_range[1], :]
            x = clip_fun(x, input_range)
            # x = x[:, input_range[0]: input_range[1], :, :]
        return x


def clip_metric(y_true, y_pred, metric_range):
    return clip_fun(y_true, metric_range), clip_fun(y_pred, metric_range)


class cAUC(tf.keras.metrics.AUC):
    def __init__(self,
                 num_thresholds=200,
                 curve='ROC',
                 summation_method='interpolation',
                 name=None,
                 dtype=None,
                 thresholds=None,
                 multi_label=False,
                 num_labels=None,
                 label_weights=None,
                 from_logits=False,
                 label_range=None,
                 metric_range=None):
        self.label_range = label_range
        self.metric_range = metric_range
        super(cAUC, self).__init__(
            num_thresholds=num_thresholds,
            curve=curve,
            summation_method=summation_method,
            name=name,
            dtype=dtype,
            thresholds=thresholds,
            multi_label=multi_label,
            num_labels=num_labels,
            label_weights=label_weights,
            from_logits=from_logits)

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates confusion matrix statistics.
        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        y_true = clip_input(y_true, self.label_range)
        y_true, y_pred = clip_metric(y_true, y_pred, self.metric_range)
        deps = []
        if not self._built:
            self._build(tf.TensorShape(y_pred.shape))

        if self.multi_label or (self.label_weights is not None):
            # y_true should have shape (number of examples, number of labels).
            shapes = [
                (y_true, ('N', 'L'))
            ]
            if self.multi_label:
                # TP, TN, FP, and FN should all have shape
                # (number of thresholds, number of labels).
                shapes.extend([(self.true_positives, ('T', 'L')),
                               (self.true_negatives, ('T', 'L')),
                               (self.false_positives, ('T', 'L')),
                               (self.false_negatives, ('T', 'L'))])
            if self.label_weights is not None:
                # label_weights should be of length equal to the number of labels.
                shapes.append((self.label_weights, ('L',)))
            deps = [
                tf.compat.v1.debugging.assert_shapes(
                    shapes, message='Number of labels is not consistent.')
            ]

        # Only forward label_weights to update_confusion_matrix_variables when
        # multi_label is False. Otherwise the averaging of individual label AUCs is
        # handled in AUC.result
        label_weights = None if self.multi_label else self.label_weights

        if self._from_logits:
            y_pred = activations.sigmoid(y_pred)

        with tf.control_dependencies(deps):
            return metrics_utils.update_confusion_matrix_variables(
                {
                    metrics_utils.ConfusionMatrix.TRUE_POSITIVES:
                        self.true_positives,
                    metrics_utils.ConfusionMatrix.TRUE_NEGATIVES:
                        self.true_negatives,
                    metrics_utils.ConfusionMatrix.FALSE_POSITIVES:
                        self.false_positives,
                    metrics_utils.ConfusionMatrix.FALSE_NEGATIVES:
                        self.false_negatives,
                },
                y_true,
                y_pred,
                self._thresholds,
                thresholds_distributed_evenly=self._thresholds_distributed_evenly,
                sample_weight=sample_weight,
                multi_label=self.multi_label,
                label_weights=label_weights)


class cPrecision(tf.keras.metrics.Precision):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None,
                 label_range=None,
                 metric_range=None):
        self.label_range = label_range
        self.metric_range = metric_range
        super(cPrecision, self).__init__(thresholds=thresholds,
                                         top_k=top_k,
                                         class_id=class_id,
                                         name=name,
                                         dtype=dtype, )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false positive statistics.
        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range `[0, 1]`.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        y_true = clip_input(y_true, self.label_range)
        y_true, y_pred = clip_metric(y_true, y_pred, self.metric_range)
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_POSITIVES: self.false_positives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)


class cRecall(tf.keras.metrics.Recall):
    def __init__(self,
                 thresholds=None,
                 top_k=None,
                 class_id=None,
                 name=None,
                 dtype=None,
                 label_range=None,
                 metric_range=None):
        self.label_range = label_range
        self.metric_range = metric_range
        super(cRecall, self).__init__(thresholds=thresholds,
                                      top_k=top_k,
                                      class_id=class_id,
                                      name=name,
                                      dtype=dtype, )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates true positive and false negative statistics.
        Args:
          y_true: The ground truth values, with the same dimensions as `y_pred`.
            Will be cast to `bool`.
          y_pred: The predicted values. Each element must be in the range `[0, 1]`.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be a
            `Tensor` whose rank is either 0, or the same rank as `y_true`, and must
            be broadcastable to `y_true`.
        Returns:
          Update op.
        """
        y_true = clip_input(y_true, self.label_range)
        y_true, y_pred = clip_metric(y_true, y_pred, self.metric_range)
        return metrics_utils.update_confusion_matrix_variables(
            {
                metrics_utils.ConfusionMatrix.TRUE_POSITIVES: self.true_positives,
                metrics_utils.ConfusionMatrix.FALSE_NEGATIVES: self.false_negatives
            },
            y_true,
            y_pred,
            thresholds=self.thresholds,
            thresholds_distributed_evenly=self._thresholds_distributed_evenly,
            top_k=self.top_k,
            class_id=self.class_id,
            sample_weight=sample_weight)


if __name__ == '__main__':
    import numpy as np

    m = cPrecision(thresholds=0.5)

    label = np.array([0, 0, 0, 0.1]).reshape((1, 1, 4, 1))
    predict = np.array([0, 0, 0, 0.6]).reshape((1, 1, 4, 1))

    m.update_state(label, predict)
    ans = m.result().numpy()
    print(ans)
