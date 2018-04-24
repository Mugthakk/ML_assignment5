import tensorflow as tf
import numpy as np
from preprocessing import get_train_test_set

train_set, test_set = get_train_test_set()

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))

classifier = tf.estimator.DNNClassifier(
    hidden_units=[321, 123, 32],
    n_classes=26,
    feature_columns=[tf.feature_column.numeric_column(key="pixels", shape=[len(train_set[0][0]),], dtype=tf.float32, default_value=0.0)],
    model_dir="savedmodels/",
    dropout=0.1,
    optimizer=tf.train.GradientDescentOptimizer(
        learning_rate=0.2,
        use_locking=False,
        name='GradientDescent'
    ),
)


train_input_fn = tf.estimator.inputs.numpy_input_fn(x={"pixels": train_set[:, 0]}, y=train_set[:, 1].astype(np.int32),
                                    batch_size=50, num_epochs=5, shuffle=True, queue_capacity=200)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"pixels": test_set[:, 0]}, y=test_set[:, 1].astype(np.int32),
                                    batch_size=50, num_epochs=1, shuffle=False, queue_capacity=200)

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
