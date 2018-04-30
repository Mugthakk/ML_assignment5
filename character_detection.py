import numpy as np
import tensorflow as tf
import skimage.transform as ski
import PIL.Image as Image


def crop_image_input_fn(cropped_image):
    # Returns a normalized crop of the image (i.e a window) as a vector ready for input into classifier
    x, y = cropped_image.size
    return np.vectorize(lambda p: p / 255.0)(np.reshape(cropped_image, x * y))


def sliding_window(classifier, image, resize_dims, window_side_pixels=20, stride=1):
    predictions = dict()
    box_num = 0

    for dims_tuple in resize_dims:
        if dims_tuple != image.size:
            img_rescaled = Image.fromarray(ski.rescale(np.array(image), dims_tuple))
            image_x, image_y = img_rescaled.size
        else:
            image_x, image_y = image.size
        for x in range(0, image_x-window_side_pixels, stride):
            for y in range(0, image_y-window_side_pixels, stride):
                box_num += 1

                # Extracts a window by using cropping
                cropped_image = image.crop((x, y, x+window_side_pixels, y+window_side_pixels))
                preds = classifier({"x": np.reshape(cropped_image, window_side_pixels*window_side_pixels)})
                predictions[box_num] = preds
    return image


if __name__ == "__main__":

    image1 = Image.open("detection-images/detection-1.jpg")
    image2 = Image.open("detection-images/detection-2.jpg")

    predict_fn = tf.contrib.predictor.from_saved_model("savedmodels_3cnn_tweak/1525079247")

    image1x, image1y = image1.size

    print(sliding_window(
        classifier=predict_fn,
        image=image1,
        resize_dims=[(image1x, image1y)]
    ))





