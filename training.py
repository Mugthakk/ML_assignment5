import tensorflow as tf
import numpy as np
from preprocessing import get_train_test_set
from cnn import cnn_model_fn

train_set, test_set = get_train_test_set()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))


train_features = []
train_labels = []
for i in range(len(train_set)):
    train_features.append(train_set[i][0])
    train_labels.append(train_set[i][1])
test_features = []
test_labels = []
for i in range(len(test_set)):
    test_features.append(test_set[i][0])
    test_labels.append(test_set[i][1])
train_features = np.array(train_features, dtype=np.float32)
train_labels = np.asarray(train_labels, dtype=np.int32)
test_features = np.array(test_features, dtype=np.float32)
test_labels = np.asarray(test_labels, dtype=np.int32)

classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="savedmodels_3cnn_tweak/",
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_features},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True,
    queue_capacity=1000,
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_features},
    y=test_labels,
    batch_size=100,
    num_epochs=1,
    shuffle=False,
    queue_capacity=1000,
)


classifier.train(
    input_fn=train_input_fn,
    steps=10000
)

eval_results = classifier.evaluate(input_fn=eval_input_fn)
print(eval_results)

feature_spec = {"x": tf.placeholder(shape=[400], dtype=tf.float32)}
input_receiver_fn = tf.estimator.export.build_raw_serving_input_receiver_fn(feature_spec)
classifier.export_savedmodel(export_dir_base="savedmodels_3cnn_tweak/",
                             serving_input_receiver_fn=input_receiver_fn,
                             as_text=True)
