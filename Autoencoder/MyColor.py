from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.models import Model, load_model,Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, UpSampling2D, RepeatVector, Reshape,Dropout
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
import tensorflow as tf

def Colorize():
    embed_input = Input(shape=(1000,))
    #Encoder
    encoder_input = Input(shape=(256, 256, 1,))
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_input)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(128, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same',strides=1)(encoder_output)
    encoder_output = MaxPooling2D((2, 2), padding='same')(encoder_output)
    encoder_output = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
    
    #Fusion
    fusion_output = RepeatVector(32 * 32)(embed_input) 
    fusion_output = Reshape(([32, 32, 1000]))(fusion_output)
    fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
    fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output)
    
    #Decoder
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)
    decoder_output = Conv2D(64, (4,4), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(32, (2,2), activation='relu', padding='same')(decoder_output)
    decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
    decoder_output = UpSampling2D((2, 2))(decoder_output)

    encoder_output_2 = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_input)
    encoder_output_2 = MaxPooling2D((2, 2), padding='same')(encoder_output_2)
    encoder_output_2 = Conv2D(128, (4,4), activation='relu', padding='same')(encoder_output_2)
    encoder_output_2 = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_output_2)
    encoder_output_2 = MaxPooling2D((2, 2), padding='same')(encoder_output_2)
    encoder_output_2 = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output_2)
    encoder_output_2 = Conv2D(256, (3,3), activation='relu', padding='same',strides=1)(encoder_output_2)
    encoder_output_2 = MaxPooling2D((2, 2), padding='same')(encoder_output_2)
    encoder_output_2 = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output_2)
    encoder_output_2 = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output_2)
    encoder_output_2 = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output_2)
    
    #Fusion
    fusion_output_2 = Conv2D(512, (1, 1), activation='relu', padding='same')(encoder_output_2)
    
    #Decoder
    decoder_output_2 = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output_2)
    decoder_output_2 = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output_2)
    decoder_output_2 = UpSampling2D((2, 2))(decoder_output_2)
    decoder_output_2 = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output_2)
    decoder_output_2 = UpSampling2D((2, 2))(decoder_output_2)
    decoder_output_2 = Conv2D(64, (4,4), activation='relu', padding='same')(decoder_output_2)
    decoder_output_2 = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output_2)
    decoder_output_2 = Conv2D(32, (2,2), activation='relu', padding='same')(decoder_output_2)
    decoder_output_2 = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output_2)
    decoder_output_2 = UpSampling2D((2, 2))(decoder_output_2)

    encoder_input_3 = concatenate([decoder_output, decoder_output_2], axis=3)
    encoder_output_3 = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_input_3)
    encoder_output_3 = MaxPooling2D((2, 2), padding='same')(encoder_output_3)
    encoder_output_3 = Conv2D(128, (4,4), activation='relu', padding='same')(encoder_output_3)
    encoder_output_3 = Conv2D(128, (3,3), activation='relu', padding='same',strides=1)(encoder_output_3)
    encoder_output_3 = MaxPooling2D((2, 2), padding='same')(encoder_output_3)
    encoder_output_3 = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output_3)
    encoder_output_3 = Conv2D(256, (3,3), activation='relu', padding='same',strides=1)(encoder_output_3)
    encoder_output_3 = MaxPooling2D((2, 2), padding='same')(encoder_output_3)
    encoder_output_3 = Conv2D(256, (4,4), activation='relu', padding='same')(encoder_output_3)
    encoder_output_3 = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output_3)
    encoder_output_3 = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output_3)
    
    #Fusion
    fusion_output_3 = Conv2D(512, (1, 1), activation='relu', padding='same')(encoder_output_3)
    
    #Decoder
    decoder_output_3 = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output_3)
    decoder_output_3 = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output_3)
    decoder_output_3 = UpSampling2D((2, 2))(decoder_output_3)
    decoder_output_3 = Conv2D(128, (3,3), activation='relu', padding='same')(decoder_output_3)
    decoder_output_3 = UpSampling2D((2, 2))(decoder_output_3)
    decoder_output_3 = Conv2D(64, (4,4), activation='relu', padding='same')(decoder_output_3)
    decoder_output_3 = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output_3)
    decoder_output_3 = Conv2D(32, (2,2), activation='relu', padding='same')(decoder_output_3)
    decoder_output_3 = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output_3)
    decoder_output_3 = UpSampling2D((2, 2))(decoder_output_3)

    return Model(inputs=[encoder_input, embed_input], outputs=decoder_output_3)
