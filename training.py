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
train_features = np.array(train_features, dtype=np.float16)
train_labels = np.asarray(train_labels, dtype=np.int32)
test_features = np.array(test_features, dtype=np.float16)
test_labels = np.asarray(test_labels, dtype=np.int32)

classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn,
    model_dir="savedmodels2/",
)

tensors_to_log = {"probabilities", "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log,
    every_n_iter=50,
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": train_features},
    y=train_labels,
    batch_size=100,
    num_epochs=None,
    shuffle=True,
    queue_capacity=200,
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": test_features},
    y=test_labels,
    batch_size=100,
    num_epochs=1,
    shuffle=False,
    queue_capacity=200,
)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)
for i in range(20):
    classifier.train(
        input_fn=train_input_fn,
        steps=1000
    )

    eval_results = classifier.evaluate(input_fn=eval_input_fn)
    print("results after ",i+1,"x 1000 steps\n",eval_results)

#tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
