import tensorflow as tf
from keras import layers
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
from keras.layers import *

# code for Half-UNet architecture based on the research paper
# https://github.com/yashkhandelwal2006/Half-UNet-for-medical-image-segmentation
def ghost_module(inputs):
    conv1 = Conv2D(32, 3, padding = 'same', kernel_initializer = 'he_normal')(inputs)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation('relu')(batch1)
    conv2 = SeparableConv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(act1)
    return concatenate([act1, conv2], axis = 3)

def build_half_unet_model_batch_normalization(input_shape = (256, 256, 1)):

    inputs = Input(shape=input_shape, name="image")
    x1 = ghost_module(ghost_module(inputs))
    pool1 = MaxPooling2D(pool_size=(2, 2))(x1)
    x2 = ghost_module(ghost_module(pool1))
    pool2 = MaxPooling2D(pool_size=(2, 2))(x2)
    x3 = ghost_module(ghost_module(pool2))
    pool3 = MaxPooling2D(pool_size=(2, 2))(x3)
    x4 = ghost_module(ghost_module(pool3))
    pool4 = MaxPooling2D(pool_size=(2, 2))(x4)
    x5 = ghost_module(ghost_module(pool4))

    up5 = UpSampling2D(size = (input_shape[0] // x5.shape[1], input_shape[1] // x5.shape[2]))(x5)
    up4 = UpSampling2D(size = (input_shape[0] // x4.shape[1], input_shape[1] // x4.shape[2]))(x4)
    up3 = UpSampling2D(size = (input_shape[0] // x3.shape[1], input_shape[1] // x3.shape[2]))(x3)
    up2 = UpSampling2D(size = (input_shape[0] // x2.shape[1], input_shape[1] // x2.shape[2]))(x2)

    upScaled = Add()([x1, up2, up3, up4, up5])
    all_conv = ghost_module(ghost_module(upScaled))
    final_conv = Conv2D(1, 1, activation = 'sigmoid')(all_conv)
  
    # final_conv = Conv2D(2, (1, 1), activation = 'softmax')(all_conv)
    half_unet_model = tf.keras.Model(inputs, final_conv, name="Half-UNet")
    return half_unet_model