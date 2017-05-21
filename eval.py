import numpy as np
import argparse
import tensorflow as tf
from tensorflow.contrib import learn
from cnn_hand_digit_classifier import cnn_hand_digit_classifier_model

def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default='save',
                        help='dir to save checkpoint in')
    args = parser.parse_args()
    mnist = learn.datasets.load_dataset("mnist")
    eval_data = mnist.test.images  # Returns np.array
    eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

    estimator = learn.Estimator(
        model_fn=cnn_hand_digit_classifier_model,
        model_dir=args.checkpoint_dir,
    )

    cnn_hand_digit_classifier = learn.SKCompat(estimator)

    metrics = {
        "accuracy": learn.MetricSpec(
                        metric_fn=tf.metrics.accuracy, prediction_key="classes"),
    }

    eval_results = cnn_hand_digit_classifier.score(x=eval_data, y=eval_labels, metrics=metrics)
    print(eval_results)


if __name__ == "__main__":
  tf.app.run()
