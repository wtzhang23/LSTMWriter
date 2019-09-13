from model import Model
import tensorflow as tf
import numpy as np
import os
import json
import re
import unidecode

# convert file to numpy array
def file_to_word_array(in_path):
	library = dict()
	arr = []
	with open(in_path, errors = 'ignore') as f:
		def push_word(word, arr):
			val = library.setdefault(word, len(library))
			arr += [val]
		
		for line_num, line in enumerate(f):
			line = unidecode.unidecode(line)
			word = ''
			for char_num, c in enumerate(line):
				if 'a' <= c <= 'z' or c == '\'':
					word += c
				elif len(word) > 0:
					push_word(word, arr)
					word = ''
				if c in (' ', '\t', '!', '?', ',', '\n', '.', '"', '-', '(', ')', ';', ':'):
					push_word(c, arr)
				elif 'A' <= c <= 'Z':
					push_word('\\upper', arr)
					word = c.lower()
	return (library, np.array(arr))
	
# convert text into training set
def dataset_generator(arr, library, batch_size, n_batches, n_epochs):
	n_samples = batch_size * n_batches
	def sample_generator():
		for idx in range(n_samples):
			seq = np.ones([analysis_level], dtype = np.int32) * -1
			seq[max(0, (analysis_level - idx)):] = arr[max(0, (idx - analysis_level)):idx]
			yield (seq, arr[idx])
	for epoch_idx in range(n_epochs):
		sample_gen = sample_generator()
		for batch_idx in range(n_batches):
			inputs = []
			targets = []
			for i in range(batch_size):
				seq, target = next(sample_gen)
				input = []
				for c in seq:
					one_hot_in = np.zeros(len(library))
					if c != -1:
						one_hot_in[c] = 1
					input += [one_hot_in]
				inputs += [input]
				targets += [target]
			inputs = np.array(inputs, dtype = np.float32)
			targets = np.array(targets, dtype = np.float32)
			yield (inputs, targets)
		

in_path = 'in.txt'
lib_path = 'lib.json'
weights_path = 'weights'
analysis_level = 12
n_epochs = 100
n_workers = 1
library, arr = file_to_word_array(in_path)
batch_size = 20
n_batches = (len(arr) - analysis_level) // batch_size
		
# save library
with open(lib_path, 'w') as lib_file:
	json.dump(library, lib_file, sort_keys = True, indent = 4)
		
# load in model
model_names = {'lib_size': len(library)}
model = Model('config.json', model_names)
model.auto_compile()
if os.path.isfile(weights_path + '.index'):
	model.predict(np.zeros([1, analysis_level, len(library)]))
	model.load_weights(weights_path)
model.fit_generator(dataset_generator(arr, library, batch_size, n_batches, n_epochs),
		epochs = n_epochs, workers = n_workers, steps_per_epoch = n_batches)
model.save_weights(weights_path)
