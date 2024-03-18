from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model
import torch


def conv_block(inputs, num_filters):
    '''
    Parameters
    ----------
    inputs(keras tensor shape = float): recieve the data inputs
    num_filters(int): recieve the filters 

    Return
    -------
    Convulation block from matrix
    '''
    x = Conv2D(num_filters, 3, padding="same")(inputs)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(num_filters, 3, padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    return x

def encoder_block(inputs, num_filters):
    '''
    Parameters
    ----------

    inputs(keras tensor shape = float): recieve the data inputs
    num_filters(int): recieve the filters 

    Return
    -------
    Created a encoder block from matrix
    '''
    s = conv_block(inputs, num_filters)
    p = MaxPool2D((2, 2))(s)
    return s, p

def decoder_block(inputs, skip_features, num_filters):
    '''
    Parameters
    ----------
    inputs(keras tensor shape = float): recieve the data inputs
    skip_features(keras tensor shape = float): recieve the parameters of encoder
    num_filters(int): recieve the filters 
    
    

    Return
    -------
    
    Created aa decoder block matrix
    '''
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(inputs)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters)
    return x

def build_unet(input_shape):
    '''
    Parameters
    ----------
    input_shape(array like): image input data to processing 256 X 256 x 3
    
    Return
    -------
    Build the unet model 
    Steps of encoder reducing the matrix with filter aplication.
    Steps of decoder expaanding the matrix


    '''
    """ Input layer """
    inputs = Input(input_shape)

    """ Encoder """
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128)
    s3, p3 = encoder_block(p2, 256)
    s4, p4 = encoder_block(p3, 512)

    """ Bottleneck """
    b1 = conv_block(p4, 1024)

    """ Decoder """
    d1 = decoder_block(b1, s4, 512)
    d2 = decoder_block(d1, s3, 256)
    d3 = decoder_block(d2, s2, 128)
    d4 = decoder_block(d3, s1, 64)

    """ Output layer """
    outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

    model = Model(inputs, outputs, name="UNET")
    return model

if __name__ == "__main__":
    # if torch.cuda.is_available():
    #     device = torch.device("cuda")          
    #     print('Treinando no GPU')
    # else:
    #     device = torch.device("cpu")
    #     print('Treinando no CPU')
        
    model = build_unet((256, 256, 3))
    model.summary()
