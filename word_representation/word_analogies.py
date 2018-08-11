import numpy as np

# function to calculate the cosine similarity of the vectors.
def cosine_similarity(u, v):

	# the dot product of two vectors is : u.v = |u|.|v|.cos(theta), where theta is the angle between them.
	# therefore, cos(theta) = (u.v)/(|u|.|v|)

	numerator = np.dot(u, v)

	d1 = np.linalg.norm(u)
	d2 = np.linalg.norm(v)

	cos = numerator/np.dot(d1, d2)

	return cos

# function to calculate the analogies
def analogy(a, b, c, word_to_vec_map):

	a, b, c = a.lower(), b.lower(), c.lower()
	# getting the vector for that particular word.
	e_a, e_b, e_c = word_to_vec_map[a], word_to_vec_map[b], word_to_vec_map[c]

	words = word_to_vec_map.keys()
	max_cosine_similarity = -1000
	best_word = None

	# iterate through all the words to find the one which has the closest cosine similarity.
	for w in words:

		vec = word_to_vec_map[w] # obtain the word embedding vector for that word.

		# if it is the words that we already have, continue.
		if w in [a, b, c]:
			continue

		sim = cosine_similarity(e_b - e_a, vec - e_c)

		# to calculate the word that is most similar to our analogy. 
		if(sim > max_cosine_similarity):
			max_cosine_similarity = sim
			best_word = w

	
	return best_word			

# function to read the pre-trainedd word embedding vectors using glove.
def read_glove_vecs(file):

	with open(file, 'r', encoding = 'utf8') as f:

		words = set()
		word_to_vec_map = {}

		for line in f:

			line = line.strip().split()
			word = line[0]
			words.add(word)
			word_to_vec_map[word] = np.array(line[1:], dtype=np.float64)

	return words, word_to_vec_map		



if __name__ == '__main__':

	words, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')

	triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
	
	for triad in triads_to_try:	
		print('{} -> {} :: {} -> {}'.format( *triad, analogy(*triad,word_to_vec_map)))




