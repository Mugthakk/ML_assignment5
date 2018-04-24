import re
import json
from collections import defaultdict
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import numpy as np
from random import shuffle, seed
from scipy.stats import zscore


# Regex to match what labels/features Tensorflow accepts
tensor_labels = re.compile(r"[A-Za-z0-9.][A-Za-z0-9_.\-/]*")
tensor_words = re.compile(r"[A-Za-z0-9_.\-/]*")

# Regex to map misspellings of tags to proper ones
ble_reg = re.compile(r'bluetooth low energy')
softdev_reg_s = re.compile(r'\bs([0-9]{2})\b')
softdev_reg_no_s = re.compile(r'\b([0-9]{3})\b')
nrf_re = re.compile(r'nrf(([0-9]{2})[0-9]{3})')
sdk_re = re.compile(r'sdk([0-9]{1,2})')
pca_re = re.compile(r'\b10040\b')

# Read stopwords to be used in cleaning
stop_words = [word.strip("\n") for word in open("common-english-words.txt")]
more_stop_words = [word.strip("\n") for word in open("more_stop_words.txt")]

# Initialize stemmer for word-cleaning
stemmer = PorterStemmer()

# This is a list of 4-tuples of threads (title, [tags], question text, thread_id) from the forum
threads = [(entry["title"], entry["tags"] if entry["tags"] else None, entry["question"]["text"], entry["t_id"]) for entry in json.load(open("data.json"))["threads"]]


# Takes in a 4-tuple and returns the same one with it's tags cleaned
def clean_thread_tags(thread):
    new_tags = regex_tags(thread[1])
    return thread[0], new_tags, thread[2], thread[3]


# Takes a list of strings (tags) as input and returns this list with the tags cleaned according to regexes
def regex_tags(tags_list):
    splitted_tags_list = [tag for tag in tags_list]
    new_tags_list = []
    for t in splitted_tags_list:
        t = ble_reg.sub('ble', t)
        t = softdev_reg_s.sub('s\\g<1>0', t)
        t = softdev_reg_no_s.sub('s\\g<1>', t)
        # Tensorflow will not accept this label, so it must be removed.
        if not tensor_labels.match(t): continue
        t = nrf_re.sub('nrf\\1 nrf\\2', t)
        t = sdk_re.sub('sdk sdk\\1', t)
        t = pca_re.sub('pca10040', t)
        if t.count(" "):
            for tag in t.split(" "):
                new_tags_list.append(tag)
        else:
            new_tags_list.append(t)
    return new_tags_list


# Returns a list of threads of 4-tuples of threads (title, [tags], question text, thread_id) with tags present, with tags being cleaned
def get_clean_tag_threads(word_usage_cap):
    return clean_forum_posts([thread for thread in threads if thread[1] is not None], word_usage_cap)


# Takes list of thread 4-tuples as input
# Returns a list of forums posts with the forum posts cleaned of stopwords, as well as the words stemmed
def clean_forum_posts(thread_tuples, word_usage_cap):
    new_tuples = []
    word_frequencies = defaultdict(int)
    for thread_tuple in thread_tuples:
        # TODO: Add the title in with these words, as a lot of information also lies within the title
        question_text = thread_tuple[0] + " " + thread_tuple[2]
        words = "".join([word.lower() for word in question_text if word.lower() not in stop_words and word.lower() not in more_stop_words])
        words_after_nltk = [stemmer.stem(word) for word in word_tokenize(words)]
        words = [word for word in words_after_nltk if word not in stop_words and word not in more_stop_words
                 and tensor_labels.match(word) and str.isalnum(word)]
        new_tuples.append((thread_tuple[0], thread_tuple[1], words, thread_tuple[3]))
        for word in words:
            word_frequencies[word] += 1
    actual_words = list(x for x, y in word_frequencies.items() if y > word_usage_cap)
    return new_tuples, actual_words


# Takes a list of threads with tags as input as well as precentage to use for training and how many tags to use (x topmost)
# Returns tensorflow_input for single_tag model of all threads that contained one or more of the topmost tags
# Variables returned: list of tag indexes and dataframe for traning, test and then the mappings of index_to_tag and tag_to_index
def generate_tensorflow_input_single_tags(clean_threads_with_tags, actual_words, train_percentage, tag_cutoff):
    top_tags_dict = get_top_tags(clean_threads_with_tags)
    top_tags = list(sorted(top_tags_dict.items(), key=lambda x: -x[1]))[:tag_cutoff]
    index_to_tag, tag_to_index = get_index_tag_mappings(top_tags)
    posts_containing_top_tag = get_threads_containing_top_tags_one_for_each(clean_threads_with_tags, top_tags)
    posts_and_tag_index = convert_tag_to_index_for_post_tag_tuples(posts_containing_top_tag, tag_to_index)
    train_labels, train_words, test_labels, test_words = generate_label_word_count_arrays(posts_and_tag_index, actual_words, train_percentage)
    return train_labels, train_words, test_labels, test_words, index_to_tag, tag_to_index, actual_words


# Returns a defaultdict of key-valye pairs: tag, occurences for each tag that has a length > 2
def get_top_tags(thread_tuples):
    tag_frequencies = defaultdict(int)
    for thread in thread_tuples:
        for tag in thread[1]:
            if len(tag) < 3: continue
            tag_frequencies[tag] = tag_frequencies[tag] + 1
    return tag_frequencies


# Takes top_tags_list as input and generates two dicts mapping int->tag, tag->int
def get_index_tag_mappings(top_tags):
    index_to_tag_map = dict((i, top_tags[i][0]) for i in range(len(top_tags)))
    tag_to_index_map = dict((top_tags[i][0], i) for i in range(len(top_tags)))
    return index_to_tag_map, tag_to_index_map


# Takes list of top tags and thread tuples as inputs and returns a list of 2-tuples (forum_post, tag)
# for every matching tag of a post with tags in top_tags
def get_threads_containing_top_tags_one_for_each(threads, top_tags, upper_bound = 1000):
    threads_containing_top_tags = []
    counts = [0 for _ in top_tags]
    for thread in threads:
        for i in range(len(top_tags)):
            if counts[i] > upper_bound: continue
            if top_tags[i][0] in thread[1]:
                counts[i] += 1
                threads_containing_top_tags.append((thread[2], top_tags[i][0]))

    return threads_containing_top_tags


# Takes a list of forum_post, tag tuples and returns one with tags replaces by index from index_dict
def convert_tag_to_index_for_post_tag_tuples(thread_tag_tuples, tag_to_index_dict):
    return [(thread, tag_to_index_dict[tag]) for thread, tag in thread_tag_tuples]


# Takes forum_post, tag-index tuples and train_percentage and outputs
# Returns the labels and corresponding count of each word in actual words with a train_percentage/1-tran_percentage split
# Between train and test lists
def generate_label_word_count_arrays(threads_index_tuples, actual_words, train_percentage):
    # Set the random seed for repeatability, shuffle the threads to prevent skewedness of trends in the test/train-split
    seed(81549300)
    shuffle(threads_index_tuples)

    train_words, test_words = [[] for _ in range(len(actual_words))], [[] for _ in range(len(actual_words))]
    train_labels, test_labels = [], []
    for forum_post, index in threads_index_tuples[:int(len(threads_index_tuples)*train_percentage)]:
        for i in range(len(actual_words)):
            train_words[i].append(forum_post.count(actual_words[i]))
        train_labels.append(index)
    for forum_post, index in threads_index_tuples[int(len(threads_index_tuples) * train_percentage):]:
        for i in range(len(actual_words)):
            test_words[i].append(forum_post.count(actual_words[i]))
        test_labels.append(index)
    train_words = np.array([zscore(np.array(sublist)) for sublist in train_words])
    train_labels = np.array(train_labels)
    test_words = np.array([zscore(np.array(sublist)) for sublist in test_words])
    test_labels = np.array(test_labels)
    return train_labels, train_words, test_labels, test_words
