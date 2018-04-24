import tensorflow as tf
import _pickle as pickle
import numpy as np
import preprocessing
import _pickle as pickle

NUM_TAGS = 10 # The top most used tags to classify
TRAIN_PERCENTAGE = 0.8 # The percentage of all data to train on
WORD_USAGE_CAP = 100 # The number of times a word must have been used in total to be considered a feature
#TODO: Uncomment these and run once if you havent before

clean_posts, words_in_clean_posts = preprocessing.get_clean_tag_threads(WORD_USAGE_CAP)
train_labels, train_frame, test_labels, test_frame, index_to_tag, tag_to_index, words = preprocessing.generate_tensorflow_input_single_tags(clean_posts, words_in_clean_posts, TRAIN_PERCENTAGE, NUM_TAGS)
pickle.dump(train_labels, open("pickles/train_labels.pickle", "wb"))
pickle.dump(train_frame, open("pickles/train_frame.pickle", "wb"))
pickle.dump(test_labels, open("pickles/test_labels.pickle", "wb"))
pickle.dump(test_frame, open("pickles/test_frame.pickle", "wb"))
pickle.dump(index_to_tag, open("pickles/index_to_tag.pickle", "wb"))
pickle.dump(tag_to_index, open("pickles/tag_to_index.pickle", "wb"))
pickle.dump(words, open("pickles/words.pickle", "wb"))

with open("pickles/train_labels.pickle", "rb") as f:
    train_labels = pickle.load(f)
with open("pickles/train_frame.pickle", "rb") as f:
    train_frame = pickle.load(f)
with open("pickles/test_labels.pickle", "rb") as f:
    test_labels = pickle.load(f)
with open("pickles/test_frame.pickle", "rb") as f:
    test_frame = pickle.load(f)
with open("pickles/index_to_tag.pickle", "rb") as f:
    index_to_tag = pickle.load(f)
with open("pickles/tag_to_index.pickle", "rb") as f:
    tag_to_index = pickle.load(f)
with open("pickles/words.pickle", "rb") as f:
    words = pickle.load(f)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True))

classifier = tf.estimator.DNNClassifier(
    hidden_units=[len(train_frame)//2, 512, 128, 32],
    n_classes=NUM_TAGS,
    feature_columns=[tf.feature_column.numeric_column(key=words[i]) for i in range(len(train_frame))],
    model_dir="single-savedmodels/",
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
        initial_accumulator_value=0.1,
        l1_regularization_strength=0.1,
        l2_regularization_strength=0.1,
        use_locking=False,
        name='ProximalAdagrad'
    )
)

input_fn = tf.estimator.inputs.numpy_input_fn({words[i]: train_frame[i] for i in range(len(train_frame))}, train_labels,
                                              batch_size=128, num_epochs=5, shuffle=True, queue_capacity=512)

eval_input_fn = tf.estimator.inputs.numpy_input_fn({words[i]: test_frame[i] for i in range(len(test_frame))}, test_labels,
                                                   batch_size=128, num_epochs=5, shuffle=False, queue_capacity=512)

train_spec = tf.estimator.TrainSpec(input_fn=input_fn)
eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

print("Starting training on (words, posts): ")
print(train_frame.shape)
print("With", NUM_TAGS, "classes")
tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
classifier.export_savedmodel("single-savedmodels/")


print("I am finished!")