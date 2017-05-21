import numpy as np
import argparse
import tensorflow as tf
from tensorflow.contrib import learn
from cnn_hand_digit_classifier import cnn_hand_digit_classifier_model

def main(_):
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='save',
                        help='dir to save checkpoint in')
    args = parser.parse_args()

    tf.logging.set_verbosity(tf.logging.INFO)
    mnist = learn.datasets.load_dataset("mnist")
    train_features = mnist.train.images  # Returns np.array
    train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    # Train the model
    runConfig = tf.contrib.learn.RunConfig(
        save_summary_steps=100,
        save_checkpoints_secs=None,
        save_checkpoints_steps=100,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000
    )
    estimator = learn.Estimator(
        model_fn=cnn_hand_digit_classifier_model,
        model_dir=args.checkpoint_dir,
        config=runConfig
    )
    cnn_hand_digit_classifier = learn.SKCompat(estimator)

    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

    cnn_hand_digit_classifier.fit(
        x=train_features,
        y=train_labels,
        batch_size=500,
        steps=1000,
        monitors=[logging_hook]
    )

    metrics = {
        "accuracy": learn.MetricSpec(
                        metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    eval_results = cnn_hand_digit_classifier.score(x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)

if __name__ == "__main__":
  tf.app.run()
