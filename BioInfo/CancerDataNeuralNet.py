from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Data sets
DATA_TRAINING = "training_data.csv"
DATA_TEST = "test_data.csv"

# Load datasets.
training_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DATA_TRAINING,
                                                                   target_dtype=np.int,
                                                                   features_dtype=np.float32,
                                                                   target_column=30)

test_set = tf.contrib.learn.datasets.base.load_csv_with_header(filename=DATA_TEST,
                                                               target_dtype=np.int,
                                                               features_dtype=np.float32,
                                                               target_column=30)

# Specify that all features have real-value data
feature_columns = [tf.contrib.layers.real_valued_column("", dimension=30)]

# Build 3 layer DNN with 10, 20, 10 units respectively.
classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_columns,
                                            hidden_units=[10, 20, 10],
                                            n_classes=3,
                                            model_dir="/tmp/data_model")

# Fit model
classifier.fit(x=training_set.data, y=training_set.target, steps=2000)

accuracy_score = classifier.evaluate(x=test_set.data, y=test_set.target)["accuracy"]
print('Accuracy: {0:f}'.format(accuracy_score))
