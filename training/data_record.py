import numpy as np
from time import time
from audio import Audio


#LABEL = 1 if recording normal voice, LABEL = 0 if recording mumble voice (stuffy nose)
# set before recording !!!
LABEL = 1
# LABEL = int(input(':'))

RECORD_SECONDS = 45

class Record(object):
    '''class to be called in audio.start_stream'''
    def __init__(self, rec_sec):
        self.RECORD_SEC = rec_sec
        self.audio_data = Audio()
        self.data = []

    def stream(self):
        '''start audio streaming and return collected data'''
        self.audio_data.start_stream(self, self.RECORD_SEC)
        return self.data

    def __call__(self, sample):
        '''collect data'''
        self.data.append(sample)


if __name__=="__main__":

    rec = Record(RECORD_SECONDS)
    data = rec.stream()
    np.save('data/data_x-{}'.format(time()),[np.array(data), LABEL]) #if no data folder, please create one