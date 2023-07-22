import numpy as np


def load_data(filename: str) -> tuple[tuple[int], dict[str, np.ndarray[tuple[int], np.dtype[np.uint8]]]]:
    """
    Load the data from the npz file
    :param filename: the name of the file
    :return: the dimensions of the images and a dictionary with the data
    """
    filename = filename.lower() + 'mnist.npz'


    data = np.load(filename)
    train = data['train_images']
    test = data['test_images']
    val = data['val_images']
    sets = {'train': train, 'test': test, 'val': val}
    return train.shape[1:], sets

def rgb2gray(rgb: np.ndarray[tuple[int], np.dtype[np.uint8]]) -> np.ndarray[tuple[int], np.dtype[np.uint8]]:
    """
    Convert an RGB image to grayscale
    :param rgb: the RGB image
    :return: the grayscale image
    """
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


# list of 2D datasets in MedMNIST
MedMNIST2d = [
    'blood',
    'breast',
    'chest',
    'derma',
    'oct',
    'organa',
    'organc',
    'organs',
    'path',
    'pneumonia',
    'retina',
    'tissue'
    ]
