import tensorflow as tf
from training.audio import Audio

RECORD_SECONDS = 45

model = tf.keras.models.load_model('training/trained_models/2-conv_layers-1-dense_layers-64-conv_size-256-dense_size-10-kernel-1542471350.h5')
model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])


class Predict(object):
    '''class to be called in audio.start_stream'''
    def __init__(self, rec_sec):
        self.RECORD_SEC = rec_sec
        self.audio_data = Audio()
        self.data = []

    def stream(self):
        '''start audio streaming'''
        self.audio_data.start_stream(self, self.RECORD_SEC)

    def __call__(self, sample):
        '''new sample from Audio processing'''
        result = model.predict(sample.reshape(-1, 1000, 1))
        print(round(result[0][0], 1))


if __name__=="__main__":
    p = Predict(RECORD_SECONDS)
    p.stream()