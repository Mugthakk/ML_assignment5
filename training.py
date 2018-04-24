import tensorflow as tf
import numpy as np
import _pickle as pickle

with open("train_set.pickle", "rb") as p:
    train_set = pickle.load(p)

with open("test_set.pickle", "rb") as p:
    test_set = pickle.load(p)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))

classifier = tf.estimator.DNNClassifier(
    hidden_units=[75, 50, 35],
    n_classes=26,
    feature_columns=[tf.feature_column.numeric_column(key=str(i)) for i in range(len(train_set[0][0]))],
    model_dir="savedmodels/",
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.1,
        l2_regularization_strength=0.1,
        use_locking=False,
        name='ProximalAdagrad'
    )
)

train_features = {str(i): np.array([train_set[j][0][i] for j in range(len(train_set))]) for i in range(len(train_set[0][0]))}
test_features = {str(i): np.array([train_set[j][0][i] for j in range(len(test_set))]) for i in range(len(test_set[0][0]))}
train_labels = np.array([ord(train_set[i][1])-97 for i in range(len(train_set))])
test_labels = np.array([ord(test_set[i][1])-97 for i in range(len(test_set))])


def input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    dataset.shuffle(1000).repeat().batch(batch_size)
    return dataset


#train_spec = tf.estimator.TrainSpec(input_fn=lambda:input_fn(train_features, train_labels, 128))
train_input_fn = tf.estimator.inputs.numpy_input_fn({str(i): train_features[str(i)] for i in range(len(train_features))}, train_labels,
                                    batch_size=128, num_epochs=1, shuffle=True, queue_capacity=512)
eval_input_fn = tf.estimator.inputs.numpy_input_fn({str(i): test_features[str(i)] for i in range(len(test_features))}, test_labels,
                                    batch_size=128, num_epochs=1, shuffle=False, queue_capacity=512)
#eval_spec = tf.estimator.EvalSpec(input_fn=lambda:input_fn(test_features, test_labels, 128))

train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
classifier.export_savedmodel("savedmodels/")