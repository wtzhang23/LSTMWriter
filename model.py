import tensorflow as tf
import json
from tensorflow.keras.models import save_model, load_model
class Model(tf.keras.Model):
	__names = {'conv2d': tf.keras.layers.Conv2D,
			'dense': tf.keras.layers.Dense,
			'lstm': tf.keras.layers.LSTM,
			'transpose': tf.transpose,
			'gather': tf.gather,
			'gather_nd': tf.gather_nd,
			'reshape': tf.reshape,
			'relu': tf.keras.activations.relu,
			'cross_entropy': tf.keras.losses.SparseCategoricalCrossentropy(),
			'adam': tf.keras.optimizers.Adam}
	
	__reserved_keywords = ['outputs']
	
	def __init__(self, config_path, custom_names = None):
		super(Model, self).__init__()
		with open(config_path) as config:
			self.data = json.load(config)
		self.__init_outputs = dict()
		for key, block in self.data.items():
			if 'init' in block:
				assert key not in Model.__reserved_keywords
				args = block['init']
				self.__init_outputs[key] = self.__parse_args(args, self.__init_outputs, custom_names, Model.__names)
	
	def call(self, inputs, training = False):
		self.__call_outputs = dict()
		custom_names = {'training': training, 'inputs': inputs}
		for key, block in self.data.items():
			if 'call' in block:
				assert key not in Model.__reserved_keywords
				args = block['call']
				self.__call_outputs[key] = self.__parse_args(args,
					self.__call_outputs, custom_names, self.__init_outputs, Model.__names)
		model_outputs = self.data['outputs']
		model_outputs = self.__replace_names(model_outputs, self.__call_outputs, custom_names, self.__init_outputs, Model.__names)
		return model_outputs
	
	def get_call_outputs():
		return self.__call_outputs
	
	def __parse_args(self, args, *custom_names):
		if 'func' in args:
			func = self.__replace_names(args['func'], *custom_names)
			if 'kwargs' in args:
				kwargs = args['kwargs']
				kwargs = self.__replace_names(kwargs, *custom_names)
				value = func(**kwargs)
			else:
				value = func();
		else:
			value = self.__replace_names(args, *custom_names)
		return value
	
	def __replace_names(self, name, *name_dicts):
		if isinstance(name, str):
			for name_dict in name_dicts:
				if name in name_dict:
					return name_dict[name]
			return name
		elif isinstance(name, list):
			return [self.__replace_names(value, *name_dicts) for value in name]
		elif isinstance(name, dict):
			return {key: self.__replace_names(value, *name_dicts) for key, value in name.items()}
		else:
			return name
	
	def auto_compile(self):
		compile_args = self.data['compile_args']
		compile_args = self.__replace_names(compile_args, self.__init_outputs, Model.__names)
		self.compile(**compile_args)
		