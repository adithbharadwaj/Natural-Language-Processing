
# character level rnn to sample characters from whatever it is trained on. 
# (in other words, we can generate new words based on the training data.)

import numpy as np

from utils import *

# activation functions.

def tanh(x):

	return np.tanh(x)

def softmax(x):

	ex = np.exp(x - np.max(x))

	return ex/ex.sum(axis = 0)

def sigmoid(x):

	return 1/(1 + np.exp(-x))


# a single step of forward propagation.
def single_forward_pass(x, a_prev, param):

	wx = param["Wax"]
	wa = param["Waa"]
	wy = param["Wya"]

	ba = param["b"]
	by = param["by"]

	a1 = tanh(np.dot(wx, x)+ np.dot(wa, a_prev) + ba) 

	y_pred = softmax(np.dot(wy, a1) + by)

	return a1, y_pred


# carries forward propagation for 'tx' times. in other words, the length of each word in the data.
def forward_propagation(X, Y, a0, param, vocab_size = 27):

	x, a, y_hat = {}, {}, {}

	a[-1] = np.copy(a0)

	loss = 0 

	# for each letter in the word (we loop tx times).
	for t in range(len(X)):

		
		# Set x[t] to be the one-hot vector representation of the t'th character in X.
		x[t] = np.zeros((vocab_size, 1))
		x[t][X[t]] = 1

		#perform single forward pass.

		a[t], y_hat[t] = single_forward_pass(x[t], a[t - 1],param)


		loss -= np.log(y_hat[t][Y[t],0])


	cache = (y_hat, a, x)

	return loss, cache


# gradient clipping function. this eliminates the problem of exploding gradients.
def clip(gradients, maxValue):

	dwx = gradients["dWax"]
	dwa = gradients["dWaa"]
	dwy = gradients["dWya"]

	dba = gradients["db"]
	dby = gradients["dby"]

	for gradient in [dwx, dwa, dwy, dba, dby]:

		np.clip(gradient, -maxValue, maxValue, out=gradient)


	gradients = {'dWax': dwx, 'dWaa': dwa, 'dWya': dwy, 'db': dba, 'dby': dby}
	
	return gradients


# function to sample a value from the model. since we are using character level recognition,
# each sample is a character.
def sample(param, char_to_ix, seed):


	wx = param["Wax"]
	wa = param["Waa"]
	wy = param["Wya"]

	ba = param["b"]
	by = param["by"]

	vocab_size = by.shape[0]

	na = wa.shape[1]

	# we are using a one hot vector to represent which character is selected. in this case, 
	# we have 26 characters and therefore, we have a vector of shape (26, 1) with zeros.
	# if we select a particular character, we make its index equal to 1. 
	# ex: [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]. here, we select 'd' (since its index is 1)
	x = np.zeros((vocab_size, 1))

	a_prev = np.zeros((na, 1))

	indices = []

	ind = -1 # flag to detect new-line character.

	count = 0 # counter to count the number of characters.
	new_line = char_to_ix['\n'] # index of the new line character in our enumeration. 

	max_length_of_word = 50 # defines the max length of words.

	while(ind != new_line and count != max_length_of_word):

		# perform forward prop.
		a = tanh(np.dot(wx, x) + np.dot(wa, a_prev) + ba) 
		z = np.dot(wy, a) + by
		y = softmax(z)

		np.random.seed(count+seed) 

		# np.random.choice is a function that chooses a value from the first argument according to
		# the probability distribution given in the second argument. the value it chooses from the first argument
		# corresponds to the maximum value (probability distribution) in the second argument.

		# ex: 
		# a = np.array([0.1, 0.0, 0.7, 0.2])
		# index = np.random.choice([0, 1, 2, 3], p = a.ravel())
		# output : index = 2 (because 2 corresponds to the index of the max value in p)

		ind = np.random.choice(list(range(vocab_size)), p = y.ravel())

		indices.append(ind)

		# converting x into a one got vector by overwriting it with zeros again.
		x = np.zeros((vocab_size, 1))
		x[ind] = 1 # the index of the character that has been chosen is set to one (one hot representation)

		# now, the current value of a is set to a_prev so that it can be used in the next iteration. 
		a_prev = a

		count += 1 
		seed += 1

	
	if(count == max_length_of_word):
		indices.append(char_to_ix['\n'])


	return indices			


# function to update the parameters with the learning rate. 
def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']

    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    
    return parameters


def optimize(x, y, a_prev, parameters, learning_rate = 0.01):

	loss, cache = forward_propagation(x, y, a_prev, parameters)

	gradients, a = rnn_backward(x, y, parameters, cache)

	gradients = clip(gradients, maxValue = 5)

	parameters = update_parameters(parameters, gradients, learning_rate)

	return loss, gradients, a[len(x) - 1]


def initialize_params(n_a, n_x, n_y):

    # Initialize parameters with small random values	

    np.random.seed(1)

    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    
    parameters = {"wx": Wax, "wa": Waa, "wy": Wya, "ba": b,"by": by}

    return parameters


def train(data, ix_to_char, char_to_ix, num_iterations = 350000, n_a = 100, sample_size = 7, vocab_size = 27):


	n_x, n_y = vocab_size, vocab_size

	parameters = initialize_parameters(n_a, n_x, n_y)

	loss = get_initial_loss(vocab_size, sample_size)

	with open("english_clean.txt") as f:

		examples = f.readlines()

	examples = [x.lower().strip('') for x in examples]

	np.random.seed(0)
	np.random.shuffle(examples)
    
	# Initialize the hidden state of your LSTM
	a_prev = np.zeros((n_a, 1))
 

	for j in range(num_iterations):

		index = j % len(examples)
		X = [None] + [char_to_ix[ch] for ch in examples[index]] 
		Y = X[1:] + [char_to_ix["\n"]]

        #optimize function. 
		curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)			

		loss = smooth(loss, curr_loss)

		if j % 2000 == 0:
            
			print('Iteration: %d, Loss: %f' % (j, loss) + '\n')

			seed = 0

			for name in range(sample_size):
                
				# Sample indices and print them
				sampled_indices = sample(parameters, char_to_ix, seed)
				print_sample(sampled_indices, ix_to_char)
                
				seed += 1  # To get the same result, increment the seed by one. 
      
			print('\n')
	
	return parameters            


if __name__ == '__main__':


	data = open('pokemon_clean.txt', 'r').read()
	data= data.lower()
	chars = list(set(data))

	chars = sorted(chars)

	print(chars)

	# storing each character along with its index using enumerate.
	# this is our character set. i.e, all the characters that we will use.

	char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) } # maps characters to index
	ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) } # maps index to characters. 

	vocab_size = len(chars) # 26 letters in the english alphabet. 

	parameters = train(data, ix_to_char, char_to_ix)

'''

	np.random.seed(1)
	
	vocab_size, n_a = 27, 100
	a_prev = np.random.randn(n_a, 1)
	
	Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
	b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
	
	parameters = {"wx": Wax, "wa": Waa, "wy": Wya, "ba": b, "by": by}

	X = [12,3,5,11,22,3]
	Y = [4,14,11,22,25, 26]

	loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)

	print("Loss =", loss)
	print("gradients[\"dWaa\"][1][2] =", gradients["dwa"][1][2])
	print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dwx"]))
	print("gradients[\"dWya\"][1][2] =", gradients["dwy"][1][2])
	print("gradients[\"db\"][4] =", gradients["dba"][4])
	print("gradients[\"dby\"][1] =", gradients["dby"][1])
	print("a_last[4] =", a_last[4])
'''


'''
	np.random.seed(2)
	_, n_a = 20, 100
	Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
	b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
	parameters = {"wx": Wax, "wa": Waa, "wy": Wya, "ba": b, "by": by}

	indices = sample(parameters, char_to_ix, 0)
	print("Sampling:")
	print("list of sampled indices:", indices, '\n')
	print("list of sampled characters:", [ix_to_char[i] for i in indices])

	#print(char_to_ix)	

	test data to test the functions. 

	np.random.seed(1)
	x = np.random.randn(3,10,4)
	a0 = np.random.randn(5,10)
	Waa = np.random.randn(5,5)
	Wax = np.random.randn(5,3)
	Wya = np.random.randn(2,5)
	ba = np.random.randn(5,1)
	by = np.random.randn(2,1)
	parameters = {"wa": Waa, "wx": Wax, "wy": Wya, "ba": ba, "by": by}

	a, y_pred, caches = forward_propagation(x, a0, parameters)
	print("a[4][1] = ", a[4][1])
	print("a.shape = ", a.shape)
	print("y_pred[1][3] =", y_pred[1][3])
	print("y_pred.shape = ", y_pred.shape)
	print("caches[1][1][3] =", caches[1][1][3])
	print("len(caches) = ", len(caches))

	np.random.seed(1)
	xt = np.random.randn(3,10)
	a_prev = np.random.randn(5,10)
	Waa = np.random.randn(5,5)
	Wax = np.random.randn(5,3)
	Wya = np.random.randn(2,5)
	ba = np.random.randn(5,1)
	by = np.random.randn(2,1)
	parameters = {"wa": Waa, "wx": Wax, "wy": Wya, "ba": ba, "by": by}

	a_next, yt_pred, a_prev, x, param = forward_prop(xt, a_prev, parameters)
	print("a_next[4] = ", a_next[4])
	print("a_next.shape = ", a_next.shape)
	print("yt_pred[1] =", yt_pred[1])
	print("yt_pred.shape = ", yt_pred.shape)

'''