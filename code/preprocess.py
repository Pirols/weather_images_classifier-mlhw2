# Used to navigate directories
import os

# Used to process the pictures
from PIL import Image, ImageOps

# Used to save the new images
from matplotlib import pyplot as plt

def resize_picture(picture_path: str, desired_size: int, output_path: str) -> None:
    """
        resize_picture(picture_path: str, desired_size: int, output_path: str) -> None
        takes as INPUTS:
            -picture_path: the path to an image file
            -desired_size: the size of the square picture we want to fit our resized picture in
            -output_path: the path to store the resized picture
        DOES:
            Creates a square black image of size desired_size*desired_size,
            pastes on it the resized input picture and saves the result as output_path.
        and OUTPUTS:
            None

        DISCLAIMER:
        This code was taken from https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/
    """

    try:
        im = Image.open(picture_path)
    except IOError:
        return
    
    old_size = im.size

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    im = im.resize(new_size, Image.ANTIALIAS)

    new_im = Image.new("RGB", (desired_size, desired_size))
    new_im.paste(im, ((desired_size-new_size[0])//2, (desired_size-new_size[1])//2))

    plt.imshow(new_im)
    plt.savefig(output_path)

if __name__ == "__main__":

    dataset_path = './original_dataset/MWI-Dataset-1.1_2000'
    dataset_path = './resized_dataset/MWI-Dataset-1.1_2000'

    for subdir, dirs, files in os.walk(dataset_path):
        for file in files:

            