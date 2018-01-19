"""
This module is used to create samples.
"""
import os

import random
import numpy as np
from PIL import Image

# source images infomation
source_image_dir = os.path.join(os.getcwd(), "imgs")
image_size = (300, 150)
breach_size = (46, 44)  # width and height

# direction of the images reversed
reversed_dir = os.path.join(os.getcwd(), "reversed_images")
# direction of the finial samples
samples_dir = os.path.join(os.getcwd(), "samples_samples")

iter_num = 20


def find_images(dir):
    filelist = os.listdir(dir)
    for f in filelist:
        if f.endswith(".png"):
            yield f


def image2matrix(image):
    return np.asarray(image)


def matrix2image(matrix):
    return Image.fromarray(matrix)


def random_pick(some_list, probablities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    index = -1
    for item, item_probability in zip(some_list, probablities):
        index += 1
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return index, item


def normalize(data):
    count = 0
    for x in range(len(data)):
        count += data[x]
    proportion = [d/count for d in data]
    return proportion


def read_image_with_label(dir, file):
    """Read image file with label

    :param dir: The direction of image bags.
    :param file: The file name of image
    :return: Image object and location of breach
    """
    assert type(file) == str, "File name is not string."
    f = os.path.join(dir, file)
    info = file.split("_")
    try:
        label = [int(info[x]) for x in range(1, 3)]
    except:
        print("The format of file name is not correct.")
    else:
        return Image.open(f), label


def reverse_image(image, loc, direction):
    """Reverse matrix

    :param matrix: The source matrix
    :param direction: The value 0,1,2 represent left-right, top-bottom, 180-degrees rotating.
    :return:
    """
    matrix = image2matrix(image)
    rows, cols, _ = matrix.shape
    matrix_new = np.zeros(matrix.shape, matrix.dtype)
    loc_new = loc[:]
    if direction == 0:  # left-right reversing
        for i in range(cols):
            matrix_new[:, i] = matrix[:, cols - i - 1]
        loc_new[0] = cols - 1 - loc[0] - (breach_size[0] - 1)
    elif direction == 1:  # top-bottom reversing
        for i in range(rows):
            matrix_new[i, :] = matrix[rows - i - 1, :]
        loc_new[1] = rows - 1 - loc[1] - (breach_size[1] - 1)
    elif direction == 2:  # 180-degrees rotating
        matrix_temp = np.zeros(matrix.shape, matrix.dtype)
        for i in range(cols):
            matrix_temp[:, i] = matrix[:, cols - i - 1]
        loc_new[0] = cols - 1 - loc[0] - (breach_size[0] - 1)
        for j in range(rows):
            matrix_new[j, :] = matrix_temp[rows - j - 1, :]
        loc_new[1] = rows - 1 - loc[1] - (breach_size[1] - 1)
    image_new = matrix2image(matrix_new)
    return image_new, loc_new


def random_split_matrix(matrix, boundary, direction):
    """A helper for split_image

    :param matrix:  The source matrix
    :param boundary: The boundary for spliting
    :param direction: The value 0,1,2,3 represent left,right,top,and bottom.
    :return: The matrix split, deviation over the source matrix
    """
    rows, cols, _ = matrix.shape
    matrix_new = np.zeros(matrix.shape, matrix.dtype)
    deviation = [0, 0]
    if direction == 0:  # Split the left to the right
        matrix_new[:, 0:cols - boundary - 1] = matrix[:, boundary + 1:cols]
        matrix_new[:, cols - boundary - 1:cols] = matrix[:, 0:boundary + 1]
        deviation[0] = deviation[0] - (boundary + 1)
    elif direction == 1:  # Split the right to the left
        matrix_new[:, boundary + 1:cols] = matrix[:, 0:cols - boundary - 1]
        matrix_new[:, 0:boundary + 1] = matrix[:, cols - boundary - 1:cols]
        deviation[0] = deviation[0] + (boundary + 1)
    elif direction == 2:  # Split the top to the bottom
        matrix_new[0:rows - boundary - 1, :] = matrix[boundary + 1:rows, :]
        matrix_new[rows - boundary - 1:rows, :] = matrix[0:boundary + 1, :]
        deviation[1] = deviation[1] - (boundary + 1)
    elif direction == 3:  # Split the bottom to the top
        matrix_new[boundary + 1:rows, :] = matrix[0:rows - boundary - 1, :]
        matrix_new[0:boundary + 1, :] = matrix[rows - boundary - 1:rows, :]
        deviation[1] = deviation[1] + (boundary + 1)
    return matrix_new, deviation


def split_image(image, label):
    """Split a part of image and past on the other side

    :param image: The source image
    :param label: The label of source image
    :return: The new image and label
    """
    # [left, right, top, botton]
    left_margin = label[0]
    right_margin = image_size[0] - (label[0] + breach_size[0])
    top_margin = label[1]
    bottom_margin = image_size[1] - (label[1] + breach_size[1])
    margin = [left_margin, right_margin, top_margin, bottom_margin]
    # Calculate probabilities about the margin for all directions
    probabilities = normalize(margin)
    # Pick directions and boundary
    direction, size = random_pick(margin, probabilities)
    boundary = random.randint(0, size)

    image_np = image2matrix(image)
    image_new_np, deviation = random_split_matrix(image_np, boundary, direction)
    image_new = matrix2image(image_new_np)
    label_new = [label[0] + deviation[0], label[1] + deviation[1]]

    return image_new, label_new


def produce_reversed_images(source_dir, save_dir):
    """Produce the reversed images"""
    if os.path.exists(source_dir) is False:
        os.makedirs(source_dir)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    reversed_images_index = 0
    source_img = find_images(source_dir)
    print("Reversing the source images...")
    for s in source_img:
        img, label = read_image_with_label(source_dir, s)
        for d in range(3):  # 3 kinds of reversing
            reversed_images_index += 1
            img_new, label_new = reverse_image(img, label, d)
            img_name = str(reversed_images_index) + "_" + str(label_new[0]) + "_" + str(label_new[1]) + "_.png"
            img_new.save(os.path.join(save_dir, img_name))
            if reversed_images_index % 100 == 0:
                print("%d images already be produced" % reversed_images_index)
    print("Reverse images reversed: %d" % reversed_images_index)


def produce_samples(source_dir, save_dir, iter_n):
    """Produce the samples"""
    if os.path.exists(source_dir) is False:
        os.makedirs(source_dir)
    if os.path.exists(save_dir) is False:
        os.makedirs(save_dir)

    samples_index = 0
    reversed_img = find_images(source_dir)
    print("Producing the Samples...")
    for r in reversed_img:
        img, label = read_image_with_label(source_dir, r)
        for _ in range(iter_n):
            samples_index += 1
            img_new, label_new = split_image(img, label)
            img_name = str(samples_index) + "_" + str(label_new[0]) + "_" + str(label_new[1]) + "_.png"
            img_new.save(os.path.join(save_dir, img_name))
            if samples_index % 100 == 0:
                print("%d samples already be produced" % samples_index)
    print("Produce Samples: %d" % samples_index)


if __name__ == "__main__":
    produce_reversed_images(source_image_dir, reversed_dir)
    produce_samples(reversed_dir, samples_dir, iter_num)