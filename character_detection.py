import numpy as np
import tensorflow as tf
import skimage.transform as ski
import PIL.Image as Image
from PIL import ImageDraw, ImageFont


def bad_image(i_p_a):

    # All white = bad
    if np.sum(i_p_a) == 255 * len(i_p_a):
        return True

    column_whites = [0 for _ in range(20)]
    row_whites = [0 for _ in range(20)]

    col_sequence_length = 0
    row_sequence_length = 0

    # Check if there is any pair of columns that are mostly white, indicates a spacing between letters/bad box
    for i in range(20):
        column_whites[i] = sum([i_p_a[j] == 255 for j in range(i, 400, 20)]) >= 15
        if column_whites[i]:
            col_sequence_length += 1
            if col_sequence_length > 1:
                return True
        else:
            col_sequence_length = 0

    # Check if there is any pair of rows that are mostly white, indicates a spacing between letters/bad box
    for i in range(20):
        row_whites[i] = sum([p == 255 for p in i_p_a[40*i:40*(i+1)]]) >= 15
        if row_whites[i]:
            row_sequence_length += 1
            if row_sequence_length > 0:
                return True
        else:
            row_sequence_length = 0

    return False


def sliding_window(classifier, image, window_side_pixels=20, stride=1):
    predictions = dict()
    image_x, image_y = image.size
    ignore_to_x = False
    reset_y = 41

    # Based on the training images we ignore the outlying pixels
    for y in range(20, image_y-2*window_side_pixels, stride):
        for x in range(20, image_x-1*window_side_pixels, stride):

            # Flags to ignore if we have set a box already
            if ignore_to_x and ignore_to_x > x:
                if reset_y == y:
                    ignore_to_x = False
                else:
                    continue
            elif ignore_to_x:
                ignore_to_x = False

            # Extracts a window by using cropping, find logit-value of best as well as predicted character
            cropp = image.crop((x, y, x+window_side_pixels, y+window_side_pixels))
            cropped_image = np.reshape(cropp, window_side_pixels*window_side_pixels)

            # Check this box to see if should be classified
            if bad_image(cropped_image):
                continue

            # Make prediction from model and save logit-value as well as character predicted
            preds = classifier({"x": cropped_image})
            best_mcl = max(preds["logits"][0])
            best_char = chr(97 + np.argmax(preds["probabilities"]))

            # Rotate the picture to see if any characters were rotated
            for deg in [90, 180, 270]:
                turned = cropp.rotate(deg)
                turned_vector = np.reshape(turned, window_side_pixels*window_side_pixels)
                preds = classifier({"x": turned_vector})
                most_certain_logit = max(preds["logits"][0])
                character = chr(97+np.argmax(preds["probabilities"]))

                # Unstable classifier due to some poor examples in data set, hence 1000 to replace guess after rotate
                if character != best_char and most_certain_logit > best_mcl + 1000:
                    best_mcl = most_certain_logit
                    best_char = character

            # Check if above threshold for classification
            ignore_to_x = x + 20
            reset_y = y + 1
            predictions[(x, y)] = best_char

    # RGB so we can see the marked classifications
    rgb = Image.new("RGBA", image.size)
    rgb.paste(image)
    i = ImageDraw.Draw(rgb)

    # Draw a box to indicate a classified character along with prediction as text
    for xy, c in predictions.items():
        i.rectangle(xy=(xy[0], xy[1], xy[0]+window_side_pixels, xy[1]+window_side_pixels), outline="green")
        i.text(xy=(xy[0]+window_side_pixels/2, xy[1]+window_side_pixels+5), text=c, fill="red", font=ImageFont.load_default())

    rgb.show()

    rgb.save("image2_1000_better_to_change"+".png")


if __name__ == "__main__":

    image1 = Image.open("detection-images/detection-1.jpg")
    image2 = Image.open("detection-images/detection-2.jpg")

    predict_fn = tf.contrib.predictor.from_saved_model("savedmodels_3cnn_tweak/1525084475")

    sliding_window(
        classifier=predict_fn,
        image=image2,
        stride=1
    )



