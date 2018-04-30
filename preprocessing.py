import matplotlib.pyplot as plt
from PIL import Image
from os import walk
import numpy as np
from sklearn.decomposition import PCA
# from sklearn.preprocessing import scale, minmax_scale
from skimage.feature import canny, hog
import _pickle as pickle


def get_image_tuples(method="edge"):
    train_vectors, train_labels = [], []
    for dirpath, dirnames, filenames in walk("./chars74k-lite"):
        for filename in filenames:
            pic = np.array(Image.open(dirpath+"/"+filename))

            if method == "edge":
                # Canny does edge-detection and returns array of true/false for pixel with edge
                pic = np.vectorize(lambda x: int(x))(canny(pic))
            elif method == "hog":
                # use HOG (http://scikit-image.org/docs/dev/auto_examples/features_detection/plot_hog.html)
                feature_descriptor, pic = hog(pic, orientations=4, pixels_per_cell=(2,2), cells_per_block=(1,1), visualise=True)

            # Reshape to single vector, general statement but second arument could just be 20*20=400 for us
            pic_as_array = np.reshape(pic, len(pic[0]) * len(pic))
            train_vectors.append(pic_as_array)
            train_labels.append(ord(filename[0])-97)
    train_vectors = np.vectorize(lambda x: x/255.0)(train_vectors)
    return train_vectors, train_labels


def pca_reduce_dims(image_vectors, new_n_features):
    # Map dimensionality from 400 -> new_n_features using PCA
    '''
    If we want to reduce the dimensionality by projecting the original data onto a vector such that
    the squared projection error is minimized in all directions, we can simply project the data onto the largest eigenvectors.
    This is called Principal Component Analysis, such projection also decorrelates the feature space.
    See:
    '''
    pca = PCA(n_components=new_n_features)
    return pca.fit_transform(image_vectors)


def one_hot_encoding_alphabet_char(char_to_encode):
    # ord("a") = 97, hence a is 0
    return np.array([int(x == ord(char_to_encode)-97) for x in range(26)])


def display_image(image, original_width=20, original_height=20, zero_one_interval=True):
    if zero_one_interval:
        image = np.vectorize(lambda p: p*255)(image)
    plt.matshow(np.reshape(image, (original_width, original_height)))
    plt.gray()
    plt.show()


def get_train_test_set(method="none", pca=False):
    train_vectors, train_labels = get_image_tuples(method=method)

    if pca and type(pca) == int:
        # Dimension of each np.array with floats reduced from 400 to 100
        train_vectors = pca_reduce_dims(train_vectors, 100)
    else:
        train_vectors = train_vectors

    # Combine to split data set for ML
    train_tuples = np.array([(train_vectors[i], train_labels[i]) for i in range(len(train_labels))])
    np.random.seed(81549300)
    np.random.shuffle(train_tuples)
    return train_tuples[:int(len(train_tuples) * 0.8)], train_tuples[int(len(train_tuples) * 0.8):]


if __name__ == "__main__":
    # A list of np.array of 0-1 floats indicating pixel density for each on index i corresponds to label on index i
    train_vectors, train_labels = get_image_tuples(method="hog")
    display_image(train_vectors[0], 20, 20, False)

    # Dimension of each np.array with floats reduced from 400 to 100
    reduced_dim_train_vectors = pca_reduce_dims(train_vectors, 100)
    #reduced_dim_train_vectors = train_vectors

    # Combine to split data set for ML
    train_tuples = np.array([(reduced_dim_train_vectors[i], train_labels[i]) for i in range(len(train_labels))])
    np.random.seed(81549300)
    np.random.shuffle(train_tuples)
    train_set, test_set = train_tuples[:int(len(train_tuples)*0.8)], train_tuples[int(len(train_tuples)*0.8):]

    # Test that we can map it back again:
    display_image(train_set[0][0], 10, 10, False)



    # Pickle the preprocessed data so we do not need to preprocess before training
    #pickle.dump(train_set, open("train_set.pickle", "wb"))
    #pickle.dump(test_set, open("test_set.pickle", "wb"))
