from time import strftime
import json

import numpy as np
from scipy.special import binom
# @title Train Utilities (SamplePool, Model Export, Damage)
from google.protobuf.json_format import MessageToDict
import tensorflow as tf
from tensorflow.python.framework import convert_to_constants

from VisualizeUtils import clear_output, imshow, to_rgba, to_rgb


###################################################################################################################################
# parameter

class Parameter:

    def __init__(self, update_probability: float = 0.5, kernel_size: int = 1, survival_boundary: float = 0.1,
                 learning_schedule: tf.keras.optimizers.schedules.LearningRateSchedule =
                 tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [2e-3, 2e-3 * 0.1]),
                 data_set: str = 'retina', trainable_perception: int = 0,
                 batch_size: int = 50, epochs: int = 3, number_cell_updates: int = 500, state_mask_update_modulo: int = 10,
                 number_hidden_states=16, layer_filter_numbers: tuple = (86, 86)):
        """
        Holding all the parameter for the experiment.

        Args:
            update_probability: This is the probability for a pixel to be updated.
            kernel_size: The size of the kernel / filter in the convolutional layer.
            survival_boundary: The boundary for pixel states to be alive.
            learning_schedule: The schedule instance for the learning rate or a float learning rate.
            data_set: A string specifying the data set.
            trainable_perception: Whether the perception of the cells is a trainable filter. 0 indicates not trainable.
                if this is > 0 then it is also taken as the kernel size for the perception filter.
            number_cell_updates: The number of iterations the model is doing for every prediction
            state_mask_update_modulo: The modulo at which iterations the alive mask for the cells is computed
                (when i %state_mask_update_modulo == 0).
            batch_size: The batch size for the training.
            epochs: The epochs of the training.
            number_hidden_states: The number of hidden states for each pixel (can be 0).
            layer_filter_numbers: A tuple of the number of filters for every layer.
        """
        assert 0.0 <= update_probability <= 1.0
        self.update_probability = update_probability
        assert 0.0 <= survival_boundary <= 1.0
        self.survival_boundary = survival_boundary
        assert number_cell_updates >= 1
        self.number_cell_updates = number_cell_updates
        self.state_mask_update_modulo = state_mask_update_modulo
        assert number_hidden_states >= 0
        self.hidden_states = number_hidden_states
        assert kernel_size >= 1
        self.kernel_size = kernel_size
        self.learning_schedule = learning_schedule
        assert trainable_perception >= 0
        self.trainable_perception = trainable_perception
        assert batch_size >= 0
        self.batch_size = batch_size
        assert epochs >= 0
        self.epochs = epochs
        assert len(layer_filter_numbers) >= 1
        self.filter_numbers = layer_filter_numbers
        self.data_set = data_set


###################################################################################################################################
# model helper functions

def get_living_mask(x, survival_boundary):
    alpha = x[:, :, :, 3:4]
    return tf.nn.max_pool2d(alpha, 3, [1, 1, 1, 1], 'SAME') >= survival_boundary


def gauss2d(sigma, fsize, dtype):
    """
    Create a 2D Gaussian filter

    Args:
        sigma: width of the Gaussian filter
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* Gaussian filter as (H, W) np.array
    """
    W, H = fsize

    # create evenly spaced values in x and y direction
    x = np.arange(-W / 2 + 0.5, W / 2)
    y = np.arange(-H / 2 + 0.5, H / 2)

    # combine them to a grid
    xx, yy = np.meshgrid(x, y, sparse=True)

    # use gaussian and normalize values
    g = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return tf.convert_to_tensor(g / np.sum(g), dtype)


def LoG(shape, dtype=None):
    """
    Approximating a Laplace filter with the difference of Gaussian (sigma2 = 1.6 * sigma1) and sigma * 5 = kernel size.
    Args:
        shape:
        dtype:

    Returns:

    """
    #print(shape) # gives (1, 1, 19, 1) 19 is RGB + hidden dimensions but shouldnt that be a 3x3 filter for exaample?
    stencil_size = (shape[0] + shape[1]) / 2
    sigma = stencil_size / 5
    # approximating the Laplace Filter with two Gaussian Filter subtracted from one another
    return gauss2d(sigma * 1.6, shape, dtype) - gauss2d(sigma, shape, dtype)


def binomial2d(shape, dtype=None):
    """
    Create a 2D binomial filter

    Args:
        fsize: (W, H) dimensions of the filter
    Returns:
        *normalized* binomial filter as (H, W) np.array
    """
    # defined as w_k = hat(w_k) / sum(hat(w_l))
    # hat(w_k) = (N k)^T, k = 0, ..., N
    x_size = shape[0]
    y_size = shape[1]

    # create evenly spaced values in x and y direction
    ax = range(0, x_size)
    ay = range(0, y_size)

    # combine them to a grid
    axx, ayy = np.meshgrid(ax, ay)

    # use binomial formula from exercise and normalize values
    binom_filter = binom(x_size - 1, axx) * binom(y_size - 1, ayy)
    binom_filter /= np.sum(np.abs(binom_filter))

    return tf.convert_to_tensor(binom_filter, dtype)


###################################################################################################################################
# model

class NCAModel(tf.keras.Model):

    def __init__(self, param: Parameter, image_shape):
        """
        Initializes the model layers.

        Args:
            image_shape: The dimensions of the image. For example (128, 128, 3) for 128 x 128 RGB image.
            param: Containing the parameter of the model
        """
        super().__init__()
        self.update_probability = param.update_probability
        self.survival_boundary = param.survival_boundary
        self.update_iterations = param.number_cell_updates
        self.state_mask_update_modulo = param.state_mask_update_modulo
        self.trainable_perception = False if param.trainable_perception == 0 else True
        self.hidden_states = param.hidden_states
        self.image_shape = image_shape

        if self.trainable_perception:
            kernel_init = "glorot_uniform"
            perceive_layer = tf.keras.layers.Conv2D(1, kernel_size=param.trainable_perception, kernel_initializer=kernel_init, strides=1, padding='same')
            perceive_layer.trainable = True
        else:
            # TODO fix trainable_perception == False
            # Also, what is the right kernel_size in this case?
            kernel_init = LoG
            perceive_layer = tf.keras.layers.Conv2D(1, kernel_size=param.kernel_size, kernel_initializer=kernel_init, strides=1, padding='same')
            perceive_layer.trainable = False
        
        input_shape = (image_shape[0], image_shape[1], image_shape[2] + self.hidden_states)
        layer_list = [tf.keras.Input(shape=input_shape),
                      perceive_layer]
        for filter_num in param.filter_numbers:
            layer_list.append(tf.keras.layers.Conv2D(filter_num, param.kernel_size, activation=tf.nn.relu))
        layer_list.append(tf.keras.layers.Conv2D(image_shape[2] + self.hidden_states, param.kernel_size))
        self.dmodel = tf.keras.Sequential(layer_list)

    @tf.function
    def call(self, x):
        """
        Implements the forward pass of the model.

        Args:
            x: The input as the current state of the model.

        Returns:
            tf.Tensor
        """
        alive_mask = get_living_mask(x, self.survival_boundary)
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= (1 - self.update_probability)
        for i in range(self.update_iterations):

            if i % self.state_mask_update_modulo == 0:
                alive_mask = get_living_mask(x, self.survival_boundary)
                update_mask = tf.random.uniform(tf.shape(x[:, :, :, :1])) <= (1 - self.update_probability)

            dx = self.dmodel(x)
            mask = update_mask & alive_mask
            x += dx * tf.cast(mask, tf.float32)

        return x


###################################################################################################################################
# model helper functions

def export_model(ca: CAModel, experiment_name: str, state_dimension: int):
    """
    Saving the model in json format in working directory.
    Args:
        ca: the instance of CAModel to save
        experiment_name: The name of the experiment. <experiment_name>.json will be the file name
    """
    ca.save_weights(experiment_name)

    # TODO why saving hard coded paarameter here?
    cf = ca.call.get_concrete_function(
        x=tf.TensorSpec([None, None, None, state_dimension]))
        # fire_rate=tf.constant(0.5),
        # angle=tf.constant(0.0),
        # step_size=tf.constant(1.0))
    cf = convert_to_constants.convert_variables_to_constants_v2(cf)
    graph_def = cf.graph.as_graph_def()
    graph_json = MessageToDict(graph_def)
    graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
    model_json = {
        'format': 'graph-model',
        'modelTopology': graph_json,
        'weightsManifest': [],
    }
    with open(f'{experiment_name}.json', 'w') as f:
        json.dump(model_json, f)


###################################################################################################################################
# train helper functions

def custom_mse_loss(target, x):
    #print(f'target: {target}\nrgba stripped: {to_rgb(x)}\nx: {x}')
    return tf.reduce_mean(tf.square(to_rgb(x) - target), axis=[-2, -3, -1])  # TODO axis parameter to map on a single value for every sample?


def make_seeds(h, w, states, length):
    """
    Initializes the states with one 1.0 in the middle of the image.

    Args:
        length:
        h:
        w:
        states:

    Returns:
        tf.Tensor object with all zeros but the state in the middle of the array is 1.0.
    """
    def make_seed(_):
        x = np.zeros([h, w, states], np.float32)
        x[h // 2, w // 2, :] = 1.0
        return x

    return tf.convert_to_tensor(list(map(make_seed, range(length))), dtype=tf.float32)


###################################################################################################################################
# experiment function

def run_experiment(loaded_data, param: Parameter):
    """
    Runs the experiment with the given parameter.

    Args:
        param: The experiment parameter.

    Returns:
        The model, the train history and the evaluation results
    """
    image_dim, data_dict = loaded_data
    h, w = image_dim[:2]
    x_train = make_seeds(h, w, param.hidden_states + image_dim[2], data_dict['train'].shape[0])
    x_test = make_seeds(h, w, param.hidden_states + image_dim[2], data_dict['test'].shape[0])
    x_val = make_seeds(h, w, param.hidden_states + image_dim[2], data_dict['val'].shape[0])
    y_train = tf.convert_to_tensor(data_dict['train'], dtype=tf.float32)
    y_test = tf.convert_to_tensor(data_dict['test'], dtype=tf.float32)
    y_val = tf.convert_to_tensor(data_dict['val'], dtype=tf.float32)
    print(f'{tf.shape(y_val)} {tf.shape(y_test)} {tf.shape(y_train)}')

    # making the model
    instance_CAModel = CAModel(param, image_dim)
    instance_CAModel.dmodel.summary()
    instance_CAModel.compile(
        # Optimizer
        optimizer=tf.keras.optimizers.Adam(param.learning_schedule),
        # Loss function to minimize
        loss=custom_mse_loss,
        # List of metrics to monitor
        metrics=[custom_mse_loss],
    )

    # training
    print("Fit model on training data")
    train_history = instance_CAModel.fit(
        x_train,
        y_train,
        batch_size=param.batch_size,
        epochs=param.epochs,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=(x_val, y_val),
    )

    # Evaluate the model on the test data using `evaluate`
    print("Evaluate on test data")
    eval_results = instance_CAModel.evaluate(x_test, y_test, batch_size=param.batch_size)
    print("test loss, test acc:", eval_results)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    #print("Generate predictions for 3 samples")
    #predictions = instance_CAModel.predict(x_test[:3])
    #print("predictions shape:", predictions.shape)

    # model persistence
    export_model(instance_CAModel, state_dimension=param.hidden_states + image_dim[2],
                 experiment_name=f'train_log/{param.data_set}_{strftime("%m.%d %H:%M:%S Uhr")}')

    return instance_CAModel, train_history, eval_results


###################################################################################################################################
# sample function

def compute_samples(instance_CAModel: CAModel, samples_n: int = 5):
    h, w, a = instance_CAModel.image_shape
    seeds = make_seeds(h, w, instance_CAModel.hidden_states + a, samples_n)
    samples = instance_CAModel.predict(seeds)
    return samples[..., :a]
