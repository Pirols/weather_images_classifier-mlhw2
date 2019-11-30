# Used to navigate directories
import os

# Used to process and save the pictures
from PIL import Image

# Used to randomly sample training and testing datasets
import numpy as np


def resize_picture(picture, desired_size, output_path):
    """
        resize_picture(picture, desired_size, output_path)
        takes as INPUTS:
            -picture_path: an image file, usually opened via PIL.Image.open(path)
            -desired_size: the size of the square picture we want to fit our resized picture in
            -output_path: the path to store the resized picture, if None the picture won't be saved
        DOES:
            Creates a square black image of size desired_size*desired_size,
            pastes on it the resized input picture and saves the result as output_path.
        and OUTPUTS:
            The resized picture

        DISCLAIMER:
        This code was inspired from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    """

    old_size = picture.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = picture.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    if output_path:
        new_im.save(output_path)
    
    return new_im
    