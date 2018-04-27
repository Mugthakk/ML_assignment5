import tensorflow as tf
import numpy as np
from preprocessing import get_train_test_set
from cnn import cnn_model_fn

train_set, test_set = get_train_test_set()
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))

classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="/savedmodels",
)

tensors_to_log = {"probabilities", "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log,
    every_n_iter=50,
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_set[:, 0]},
    y=train_set[:, 1],
    batch_size=100,
    num_epochs=None,
    shuffle=True,
    queue_capacity=200,
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_set[:, 0]},
    y=test_set[:, 1].astype(np.int32),
    batch_size=100,
    num_epochs=1,
    shuffle=False,
    queue_capacity=200,
)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

classifier.train(
    input_fn=train_input_fn,
    steps=20000,
    hooks=[logging_hook]
)

#tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
