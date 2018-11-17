import tensorflow as tf
import numpy as np
import time
import utils

path = r'data/'

x, y = utils.reload_data(path)
x = np.array(x).reshape(-1, 1000, 1)


# prepared for testing and evaluating. try other combinations of architecture
dense_layers = [1]
conv_sizes = [64]
conv_layers = [2]
dense_layer_sizes = [256]
kernel = 10
pool_size = 4

_batchs = 5
_epochs = 10

for dense_layer in dense_layers:
    for conv_layer in conv_layers:
        for dense_size in dense_layer_sizes:
            for conv_size in conv_sizes:

                NAME = '{}-conv_layers-{}-dense_layers-{}-conv_size-{}-dense_size-{}-kernel-{}'.format(conv_layer,dense_layer,conv_size, dense_size,kernel, int(time.time()))
                model = tf.keras.Sequential()

                model.add(tf.keras.layers.Conv1D(conv_size, kernel, activation='relu', input_shape = (1000, 1)))
                model.add(tf.keras.layers.MaxPooling1D(4))

                for i in range(conv_layer-1):
                    model.add(tf.keras.layers.Conv1D(conv_size, kernel, activation='relu'))
                    model.add(tf.keras.layers.MaxPooling1D(4))

                model.add(tf.keras.layers.Flatten())

                for _ in range(dense_layer):
                    model.add(tf.keras.layers.Dense(dense_size, activation='relu'))

                model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
                model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
                tensorboard = tf.keras.callbacks.TensorBoard(log_dir='model_evaluate/{}'.format(NAME))
                print(NAME)
                model.fit(x,y, batch_size = _batchs, epochs=_epochs, validation_split = 0.2, callbacks=[tensorboard])
                model.save('trained_models/{}.h5'.format(NAME))