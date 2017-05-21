import numpy as np
import tensorflow as tf

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib


def cnn_hand_digit_classifier_model(features, labels, mode):
    """
        A CNN model for Handwritten Digit classification.
        Network structure:
        Convolutional Layer #1 > Pooling Layer #1 > Convolutional Layer #2 > Pooling Layer #2 >
        Dense Layer #1 > Dense Layer #2 (Logits Layer)
    """

    conv1_filter_size = 32
    conv1_kernel_size = [5, 5]
    max_pooling1_pool_size = [2, 2]
    conv2_filter_size = 64
    max_pooling2_pool_size = [2, 2]
    dense1_unit = 1024
    dense2_unit = 1024

    # Input Layer
    input_layer = tf.reshape(features, [-1, 28, 28, 1])

    # Convolutional Layer #1
    with tf.name_scope("conv2d-1-%s" % conv1_filter_size):
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=conv1_filter_size,
            kernel_size=conv1_kernel_size,
            strides=(1, 1),
            padding="same",
            activation=tf.nn.relu
        )

    # Pooling Layer #1
    with tf.name_scope("max_pooling2d-1"):
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=max_pooling1_pool_size, strides=2)

    # Convolutional Layer #2
    with tf.name_scope("conv2d-2-%s" % conv2_filter_size):
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=conv2_filter_size,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu
        )

    # Pooling Layer #2
    with tf.name_scope("max_pooling2d-2"):
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=max_pooling2_pool_size, strides=2)

    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    with tf.name_scope("desne1-%s" % dense1_unit):
        dense1 = tf.layers.dense(inputs=pool2_flat, units=dense1_unit, activation=tf.nn.relu)

    with tf.name_scope("dropout1"):
        dropout1 = tf.layers.dropout(inputs=dense1, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

    with tf.name_scope("desne2-%s" % dense2_unit):
        dense2 = tf.layers.dense(inputs=dropout1, units=dense2_unit, activation=tf.nn.relu)

    with tf.name_scope("dropout1"):
        dropout2 = tf.layers.dropout(inputs=dense2, rate=0.5, training=mode == learn.ModeKeys.TRAIN)

    # Logits Layer
    with tf.name_scope("logits"):
        logits = tf.layers.dense(inputs=dropout2, units=10)

    loss = None
    train_op = None

    # Calculate Loss (for both TRAIN and EVAL modes)
    if mode != learn.ModeKeys.INFER:
        onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
        loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

    # Configure the Training Op (for TRAIN mode)

    if mode == learn.ModeKeys.TRAIN:
        with tf.name_scope('train'):
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.contrib.framework.get_global_step(),
                learning_rate=0.001,
                optimizer="Adam"
            )

    predictions = {
        "classes": tf.argmax(input=logits, axis=1, name="classes_tensor"),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Return a ModelFnOps object
    return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)
