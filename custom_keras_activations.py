import keras.backend as K

def swish(x, beta):
	return x * K.sigmoid(beta * x)
	
def gelu(x):
	return swish(x, 1.702)