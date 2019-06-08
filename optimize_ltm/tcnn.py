import keras.backend as K
import keras.layers
from keras import optimizers
from keras.engine.topology import Layer
from keras.layers import Activation, Lambda
from keras.layers import Conv1D, SpatialDropout1D
from keras.layers import Convolution1D, Dense
from keras.models import Input, Model
from typing import List, Tuple


def channel_normalization(x):
    # type: (Layer) -> Layer
    """ Normalize a layer to the maximum activation
    This keeps a layers values between zero and one.
    It helps with relu's unbounded activation
    Args:
        x: The layer to normalize
    Returns:
        A maximal normalized layer
    """
    max_values = K.max(K.abs(x), 2, keepdims=True) + 1e-5
    out = x / max_values
    return out


def wave_net_activation(x):
    # type: (Layer) -> Layer
    """This method defines the activation used for WaveNet
    described in https://deepmind.com/blog/wavenet-generative-model-raw-audio/
    Args:
        x: The layer we want to apply the activation to
    Returns:
        A new layer with the wavenet activation applied
    """
    tanh_out = Activation('tanh')(x)
    sigm_out = Activation('sigmoid')(x)
    return keras.layers.multiply([tanh_out, sigm_out])


def residual_block(x, s, i, activation, nb_filters, kernel_size, dropout_rate=0):
    # type: (Layer, int, int, str, int, int, float) -> Tuple[Layer, Layer]
    """Defines the residual block for the WaveNet TCN
    Args:
        x: The previous layer in the model
        s: The stack index i.e. which stack in the overall TCN
        i: The dilation power of 2 we are using for this residual block
        activation: The name of the type of activation to use
        nb_filters: The number of convolutional filters to use in this block
        kernel_size: The size of the convolutional kernel
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
    Returns:
        A tuple where the first element is the residual model layer, and the second
        is the skip connection.
    """
    original_x = x
    conv = Conv1D(filters=nb_filters, kernel_size=kernel_size,
                  dilation_rate=i, padding='causal'#,
                 # name='dilated_conv_%d_tanh_s%d' % (i, s)
                 )(x)
    if activation == 'norm_relu':
        x = Activation('relu')(conv)
        x = Lambda(channel_normalization)(x)
    elif activation == 'wavenet':
        x = wave_net_activation(conv)
    else:
        x = Activation(activation)(conv)

    x = SpatialDropout1D(dropout_rate#, name='spatial_dropout1d_%d_s%d_%f' % (i, s, dropout_rate)
                        )(x)

    # 1x1 conv.
    x = Convolution1D(nb_filters, 1, padding='same')(x)
    res_x = keras.layers.add([original_x, x])
    return res_x, x


def process_dilations(dilations):
    def is_power_of_two(num):
        return num != 0 and ((num & (num - 1)) == 0)

    if all([is_power_of_two(i) for i in dilations]):
        return dilations

    else:
        new_dilations = [2 ** i for i in dilations]
        #print(f'Updated dilations from {dilations} to {new_dilations} because of backwards compatibility.')
        return new_dilations


def TCN(input_layer,
        nb_filters=64,
        kernel_size=2,
        nb_stacks=1,
        dilations=None,
        go_backwards=False,
        activation='norm_relu',
        use_skip_connections=True,
        dropout_rate=0.0,
        return_sequences=True):
    """Creates a TCN layer.
    Args:
        input_layer: A tensor of shape (batch_size, timesteps, input_dim).
        nb_filters: The number of filters to use in the convolutional layers.
        kernel_size: The size of the kernel to use in each convolutional layer.
        dilations: The list of the dilations. Example is: [1, 2, 4, 8, 16, 32, 64].
        nb_stacks : The number of stacks of residual blocks to use.
        activation: The activations to use (norm_relu, wavenet, relu...).
        use_skip_connections: Boolean. If we want to add skip connections from input to each residual block.
        return_sequences: Boolean. Whether to return the last output in the output sequence, or the full sequence.
        dropout_rate: Float between 0 and 1. Fraction of the input units to drop.
    Returns:
        A TCN layer.
    """
    if go_backwards:
        input_layer = Lambda(lambda tt: K.reverse(tt,1))(input_layer)
    if dilations is None:
        dilations = [1, 2, 4, 8, 16, 32]
    x = input_layer
    x = Convolution1D(nb_filters, 1, padding='causal')(x)
    skip_connections = []
    for s in range(nb_stacks):
        for i in dilations:
            x, skip_out = residual_block(x, s, i, activation, nb_filters, kernel_size, dropout_rate)
            skip_connections.append(skip_out)
    if use_skip_connections:
        x = keras.layers.add(skip_connections)
    x = Activation('relu')(x)
    
    if go_backwards:
        x = Lambda(lambda tt: K.reverse(tt,1))(x)
    if not return_sequences:
        output_slice_index = -1
        x = Lambda(lambda tt: tt[:, output_slice_index, :])(x)
    return x

