import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import regularizers
class ConvBlock(tf.keras.Model):
    def __init__(self, kernel_size, filters, reg):
        super(ConvBlock, self).__init__()
        filters1, filters2, filters3 = filters

        self.conv2a = tf.keras.layers.Conv2D(filters1, (1, 1), kernel_regularizer=regularizers.l2(reg))
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same', kernel_regularizer=regularizers.l2(reg))
        self.bn2b = tf.keras.layers.BatchNormalization()

        self.conv2c = tf.keras.layers.Conv2D(filters3, (1, 1), kernel_regularizer=regularizers.l2(reg))
        self.bn2c = tf.keras.layers.BatchNormalization()
    
    @tf.function
    def call(self, input_tensor, training=False):
        #print(f"1: {input_tensor.shape}")
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)
        #print(f"2: {x.shape}")
        x = self.conv2b(x)
        x = self.bn2b(x, training=training)
        x = tf.nn.relu(x)
        #print(f"3: {x.shape}")
        x = self.conv2c(x)
        x = self.bn2c(x, training=training)
        #print(f"4: {x.shape}")
        x += input_tensor
        return tf.nn.relu(x)

class PolicyNet(tf.keras.Model):
    def __init__(self, boardSize, nb_convBlocks, hidden_size, reg):
        super(PolicyNet, self).__init__()
        tf.keras.backend.set_floatx('float32')
        self.board_input = tf.keras.layers.InputLayer(input_shape=(boardSize,boardSize))
        self.indicator_input = tf.keras.layers.InputLayer(input_shape=(1,))

        # Covolution model for board embbeding
        self.Conv_blocks = []
        for _ in range(nb_convBlocks):
            self.Conv_blocks.append(ConvBlock(kernel_size=3, filters=[1,2,3], reg=reg))
        
        self.Conv_blocks.append(tf.keras.layers.Flatten())

        # Dense layers for considering the indicator and to create the global flattened embedding

        self.dance = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=regularizers.l2(reg)),
            tf.keras.layers.Dense(boardSize**2, activation='relu', kernel_regularizer=regularizers.l2(reg))
            #tf.keras.layers.Reshape((boardSize,boardSize, 1))
        ])
        self.con = tf.keras.layers.Concatenate()
        # Policy head
        self.pieHead = tf.keras.Sequential([
            tf.keras.layers.Conv2D(2, 3, (1, 1), kernel_regularizer=regularizers.l2(reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Conv2D(2, 3, (1, 1), kernel_regularizer=regularizers.l2(reg)),
            tf.keras.layers.ReLU(),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(boardSize**2, activation='softmax', kernel_regularizer=regularizers.l2(reg))
        ])
        self.pieHead0 = tf.keras.layers.Dense(boardSize**2+1, activation='softmax', kernel_regularizer=regularizers.l2(reg))
        # Value head
        self.vHead = tf.keras.Sequential([
            tf.keras.layers.Conv2D(1, 1, (1, 1), kernel_regularizer=regularizers.l2(reg)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(reg)),
            tf.keras.layers.Dense(1, activation='tanh', kernel_regularizer=regularizers.l2(reg))
        ])

    @tf.function
    def call(self, board_tensor, ind_tensor, training=False):
        z = self.board_input(board_tensor)

        for block in self.Conv_blocks:
            z = block(z, training=training)
        # Here z should be the flattened embedding of size board_size^2

        y = self.indicator_input(ind_tensor)
        #print(z.shape, y)
        z = tf.keras.layers.concatenate([z, y], axis=1)
        z = self.dance(z, training=training) 
        z = tf.reshape(z, board_tensor.shape)
        pie = self.pieHead(z, training=training)
        l = tf.keras.layers.concatenate([pie, y], axis=1) 
        pie = self.pieHead0(l, training=training)
        v = self.vHead(z, training=training)
        #print(f"call pie shape {pie.shape}")
        return pie, v
        