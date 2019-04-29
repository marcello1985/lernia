"""
train_weekImg:
implementation of different models with keras to predict time series represented like images.
"""

import random, json, datetime, re
import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import keras
import tensorflow
from keras import backend as K
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, LSTM, RepeatVector, Dropout
from keras.models import Model, Sequential
from keras.callbacks import TensorBoard
from keras.wrappers.scikit_learn import KerasRegressor
import geomadi.train_score as t_s
from geomadi.train_keras import trainKeras

class weekImg(trainKeras):
    """train on image representation of weeks per hourly values"""
    def __init__(self):
        trainKeras.__init__(self,X)

    def getX(self):
        return self.X
    
    def plotImg(self,nline=12):
        """plot a sample of weekly representations"""
        n = min(nline*nline,self.X.shape[0])
        N = self.X.shape[0]
        shuffleL = random.sample(range(N),N)
        plt.figure()
        for i in range(n):
            ax = plt.subplot(nline,nline,i+1)
            plt.imshow(self.X[shuffleL[i]])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #plt.tight_layout()
        plt.subplots_adjust(left=0.1,bottom=0.5,right=None,top=None,wspace=None,hspace=None)
        plt.show()

    def plotTimeSeries(self,nline=6):
        """plot a time series along side with a the image representation"""
        glL = []
        N = self.X.shape[0]
        shuffleL = random.sample(range(N),N)
        for i in shuffleL[:nline]:
            glL.append(self.X[i])
            j = [1,5,9,3,7,11]
            ncol = int(nline*2/4)
            ncol = ncol + (ncol %2)
            plt.figure()
            plt.title("image representation of time series")
        for i in range(nline):
            ax = plt.subplot(4,ncol,i*2+1)
            plt.plot(glL[i].ravel())
            ax = plt.subplot(4,ncol,i*2+2)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            plt.imshow(glL[i])

        plt.xlabel("hours")
        plt.ylabel("frequency")
        plt.show()

    def plotMorph(self,nline=4):
        """plot the morphing between original images and autoencoded"""
        n = 10
        rawIm = self.X.reshape((len(self.X),np.prod(self.X.shape[1:])))
        encIm = self.encoder.predict(rawIm)
        decIm = self.decoder.predict(encIm)
        N = self.X.shape[0]
        shuffleL = random.sample(range(N),N)
        plt.figure()
        for i in range(int(nline*n/2)):
            c = (i % n)
            r = int( (i - c)/n )
            ax = plt.subplot(nline,n, c + (r*2)*n + 1)
            plt.imshow(rawIm[shuffleL[i]].reshape(X.shape[1], X.shape[2]))
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #plt.subplots_adjust(bottom=.4)
            ax = plt.subplot(nline,n,c + (r*2+1)*n + 1)
            plt.imshow(decIm[shuffleL[i]].reshape(X.shape[1], X.shape[2]))
            ax.set_title("^")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            #plt.subplots_adjust(bottom=.1)
            #        plt.tight_layout(pad=1.08, h_pad=0.1, w_pad=None, rect=None)
        plt.subplots_adjust(left=None,bottom=0.4,right=None, top=None, wspace=None, hspace=None)
        plt.show()

    def getEncoder(self):
        """return the encoder"""
        if not self.encoder:
            raise Exception("train the model first")
        return self.encoder
        
    def getDecoder(self):
        """return the decoder"""
        if not self.decoder:
            raise Exception("train the model first")
        return self.decoder

    def getaAutoencoder(self):
        """return the autoencoder"""
        if not self.autodecoder:
            raise Exception("train the model first")
        return self.autodecoder

    def defEncoder(self):
        """define encoder"""
        encoding_dim = 64
        pix_dim = self.X.shape[1]*self.X.shape[2]#784
        input_img = Input(shape=(pix_dim,))
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(pix_dim, activation='sigmoid')(encoded)
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img, encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input, decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta',loss='binary_crossentropy')
        self.model = autoencoder
        return encoder, decoder, autoencoder

    def defDeepEncoder(self):
        """define deep encoder"""
        layDim = [128,64,32]
        encoding_dim = 128
        # layDim = [64,32,16]
        # encoding_dim = 32
        pix_dim = self.X.shape[1]*self.X.shape[2]
        input_img = Input(shape=(pix_dim,))
        encoded = Dense(layDim[0], activation='relu')(input_img)
        encoded = Dense(layDim[1], activation='relu')(encoded)
        encoded = Dense(layDim[2], activation='relu')(encoded)
        decoded = Dense(layDim[1], activation='relu')(encoded)
        decoded = Dense(layDim[0], activation='relu')(decoded)
        decoded = Dense(pix_dim, activation='sigmoid')(decoded)
        autoencoder = Model(input_img, decoded)
        encoder = Model(input_img,encoded)
        encoded_input = Input(shape=(encoding_dim,))
        decoder_layer = autoencoder.layers[-1]
        decoder = Model(encoded_input,decoder_layer(encoded_input))
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.model = autoencoder
        return encoder, decoder, autoencoder

    def defConvNet(self):
        """define convolution neural net"""
        input_img = Input(shape=(x_train.shape[1], x_train.shape[2], 1))
        k_size = (3,3) #(5,5)
        convS = [8,16] #[32,64]
        x = Conv2D(convS[0], k_size, activation='relu', padding='same')(input_img)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(convS[1], k_size, activation='relu', padding='same')(x)
        x = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(convS[1], k_size, activation='relu', padding='same')(x)
        encoded = MaxPooling2D((2, 2), padding='same')(x)
        x = Conv2D(convS[1], k_size, activation='relu', padding='same')(encoded)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(convS[1], k_size, activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)
        x = Conv2D(convS[0], k_size, activation='relu')(x)
        x = UpSampling2D((2, 2))(x)
        decoded = Conv2D(1,k_size,activation='sigmoid', padding='same')(x)
        autoencoder = Model(input_img, decoded)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.model = autoencoder
        return encoder, decoder, autoencoder
        
    def runEncoder(self,epoch=150,isEven=True):
        """perform a simple autoencoder"""
        if not isEven:
            X = np.reshape(self.X,(self.X.shape[0],14,12))
        else:
            X = self.X
        N = X.shape[0]
        X_train, X_test, y_train, y_test = self.splitSet(X,list(range(N)))
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
        encoder, decoder, autoencoder = self.defEncoder()
        #encoder, decoder, autoencoder = self.defDeepEncoder()
        #encoder, decoder, autoencoder = self.defConvNet()
        self.history = autoencoder.fit(x_train,x_train,epochs=epoch,batch_size=128,shuffle=True,validation_data=(x_test,x_test),callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
        self.encoder = encoder
        self.decoder = decoder
        self.model = autoencoder
        self.autoencoder = autoencoder
        
    def varAutoenc(self):
        i = 1
        
