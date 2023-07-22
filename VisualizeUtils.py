"""
>>>
	Image functions:
		* imwrite(filename, arr) - saves the image [arr] to file [filename]

	Video functions:
		* VideoWriter(filename, fps) - creates a video writer to file [filename]
		* VideoWriter.add(img) - add a frame [img] to the video

	Helper functions:
		* tile2d(arr, w=None)
		* zoom(img, scale=4)
		* to_rgba(x)
		* to_alpha(x)
		* to_rgb(x)
<<<
"""

from base64 import b64encode
from io import BytesIO
from PIL import Image
import tensorflow as tf
import numpy as np
import os

os.environ['FFMPEG_BINARY'] = 'ffmpeg'

try:
    import moviepy.editor as mvp
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
except ModuleNotFoundError:
    import pip
    pip.main(['install', 'moviepy'])
    import moviepy.editor as mvp
    from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter


###########
## Image ##
###########


# Save image to file


def _np2pil(arr):
    """ Converts the numpy array [arr] to a PIL image. """
    if arr.dtype in [np.float32, np.float64]:
        arr = np.uint8(np.clip(arr, 0, 1) * 255)

    return Image.fromarray(arr)


def imwrite(filename, arr):
    """ Saves the numpy array [arr] as an image to the file [filename]. """
    arr = np.asarray(arr)
    fmt = filename.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg': fmt = 'jpeg'

    with open(filename, 'wb') as file:
        _np2pil(arr).save(file, fmt, quality=95)


# Helper


def tile2d(arr, w=None):
    arr = np.asarray(arr)
    if w is None: w = int(np.ceil(np.sqrt(len(arr))))
    th, tw = arr.shape[1:3]
    pad = (w - len(arr)) % w
    arr = np.pad(arr, [(0, pad)] + [(0, 0)] * (arr.ndim - 1), 'constant')
    h = len(arr) // w
    arr = arr.reshape([h, w] + list(arr.shape[1:]))
    arr = np.rollaxis(arr, 2, 1).reshape([th * h, tw * w] + list(arr.shape[4:]))
    return arr


def zoom(img, scale=4):
    """ Rescales the image. New size = old size * [scale]. """
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


def to_rgba(x):
    return x[..., :4]


def to_alpha(x):
    return tf.clip_by_value(x[..., 3:4], 0.0, 1.0)


def to_rgb(x):
    # assume rgb premultiplied by alpha
    # rgb, a = x[..., :3], to_alpha(x)
    # return 1.0 - a + rgb

    return x[..., :3]       # hacked version TODO


###########
## Video ##
###########


class VideoWriter:

    def __init__(self, filename, fps=30.0, **kw):
        """ Create a new Video Writer to write to the file [filename]."""
        self.writer = None
        self.params = dict(filename=filename, fps=fps, **kw)

    def add(self, img):
        """ Add the frame [img] to the video. """
        img = np.asarray(img)
        if self.writer is None:
            h, w = img.shape[:2]
            self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)

        if img.dtype in [np.float32, np.float64]:
            img = np.uint8(img.clip(0, 1) * 255)

        if len(img.shape) == 2:
            img = np.repeat(img[..., None], 3, -1)

        self.writer.write_frame(img)

    def close(self):
        if self.writer:
            self.writer.close()

    def __enter__(self):
        return self

    def __exit__(self, *kw):
        self.close()


##############
## IPhython ##
##############


from IPython import display
from IPython.display import display
from IPython.display import Image as DisplayImage
from IPython.display import clear_output as DisplayClearOutput


def clear_output():
    DisplayClearOutput()


def _imencode(arr):
    """ Encodes the numpy array [arr] as a bytes object. """
    arr = np.asarray(arr)
    with BytesIO() as file:
        _np2pil(arr).save(file, 'png')
        return file.getvalue()


def imshow(a):
    """ Displays the numpy array [arr] as an image. """
    display(DisplayImage(data=_imencode(a)))
