from scipy import stats
import matplotlib.pyplot as plt
from numpy import reshape, array, float16
from PIL import Image
from os import walk, path
import numpy as np
import random


def get_image_tuples():
    train_tuples = []
    for dirpath, dirnames, filenames in walk("./chars74k-lite"):
        for filename in filenames:
            pic = np.array(Image.open(dirpath+"/"+filename))
            pic_as_array = np.reshape(pic, len(pic[0]) * len(pic)) # This would be 20*20=400 for us, but this is general
            pic_as_array = np.vectorize(lambda p: p/255.0)(pic_as_array) # Map onto 0-1 interval
            pic_array_and_label_tuple = (pic_as_array, filename[0])
            train_tuples.append(pic_array_and_label_tuple)
    return train_tuples


def display_image(image, original_width=20, original_height=20, zero_one_interval=True):
    if zero_one_interval:
        image = np.vectorize(lambda p: p*255)(image)
    plt.gray()
    plt.matshow(np.reshape(image, (original_width, original_height)))
    plt.show()


if __name__ == "__main__":
    # A list of tuples that contain a single np.array of pixel values mapped onto interval 0-1 and a label as a string
    train_tuples = get_image_tuples()
    
    # Test that we can map it back again:
    display_image(train_tuples[0][0])
