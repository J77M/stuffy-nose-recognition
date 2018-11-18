# stuffy-nose-recognition
Trained neural network for recognizing speaking with stuffy nose. <br />
Extension : real time recognition on 2 seconds frames.<br />
Model was trained only on data from one person (me), so there is possibility to record data and train model. <br />
Also record new data and train model if demand is smaller frame (more real time recognition).<br />
(There was extension in training process, instead of training on recording data, to train on analysed wav files, but results on real time were about 60% accuracy, so feel free to try and improve)<br />

<br />
|<br />
|__ training <br />
|          |_ data		# data recorded by data_record<br />
|          |_ model_evaluate	# folder for tensorboard evaluations<br />
|          |_ trained_models	# trained models<br />
|          |_ data_record.py	# record and saves<br />
|          |_ model.py		# neural network model <br />
|          |_ utils.py		# useful functions<br />
|          |_ audio.py		# class for audio processing<br />
|<br />
|__ real_time_analysis.py	# real time use of trained neural network <br />
	   
<br />
<br />
Testing showed, that the best architecture is :<br />
_________________________________________________________________<br />
Layer (type)                 Output Shape              Param #   <br />
=================================================================<br />
conv1d (Conv1D)              (None, 991, 64)           704       <br />
_________________________________________________________________<br />
max_pooling1d (MaxPooling1D) (None, 247, 64)           0         <br />
_________________________________________________________________<br />
conv1d_1 (Conv1D)            (None, 238, 64)           41024     <br />
_________________________________________________________________<br />
max_pooling1d_1 (MaxPooling1 (None, 59, 64)            0         <br />
_________________________________________________________________<br />
flatten (Flatten)            (None, 3776)              0         <br />
_________________________________________________________________<br />
dense (Dense)                (None, 256)               966912    <br />
_________________________________________________________________<br />
dense_1 (Dense)              (None, 1)                 257       <br />
=================================================================<br />
Total params: 1,008,897<br />
Trainable params: 1,008,897<br />
Non-trainable params: 0<br />
_________________________________________________________________<br />
None<br />
<br />

batch_size = 5, epochs=10 <br />
100% validation accuracy on my data (i wont share them, because of my privacy :) )<br />
data - total: 4.5 min of recording (2.25 min talking with stuffy nose - nose stuffed with fingers, 2.25 min talking with clear nose)<br />
<br />
If you are training model, please test and evaluate your combinations of architecture.<br />
Training code is adapted for tensorboard evaluation. <br />
<br />
J.M.
