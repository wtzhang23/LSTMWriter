import tensorflow as tf
from model import Model
import numpy as np
import json
import os

lib_path = 'lib.json'
output_path = 'out.txt'
weights_path = 'weights'
analysis_level = 12
out_length = 1000

#load in library
with open(lib_path) as f:
	library = json.load(f)
	library = {value: key for key, value in library.items()}
	
# load in model
model_names = {'lib_size': len(library)}
model = Model('config.json', model_names)
model.auto_compile()
output = np.zeros(out_length)
if os.path.isfile(weights_path + '.index'):
	seed = np.zeros([1, analysis_level, len(library)])
	model.predict(seed)
	model.load_weights(weights_path)
	for i in range(out_length):
		print('Progress: {}/{}'.format(i, out_length), end = '\r')
		prediction = np.random.choice(len(library), p = model.predict(seed)[0])
		#prediction = np.argmax(model.predict(seed)[0])
		output[i] = prediction
		one_hot = np.zeros(len(library))
		one_hot[prediction] = 1
		seed[0] = np.concatenate([seed[0][1:], [one_hot]], axis = 0) 
	print('Progress: {}/{}'.format(out_length, out_length))
	
	print('Transcribing and writing predictions')
	output_str = ''
	upper = False
	for id in output:
		word = library[int(id)]
		if word == '\\upper':
			upper = True
		else:
			if upper:
				word = word[0].upper() + word[1:]
				upper = False
			output_str += word
		
	with open(output_path, 'w') as out_file:
		out_file.write(output_str)
		