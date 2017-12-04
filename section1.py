'''
Currently concatenates the title and body together to get the text
'''

import random

# -------------------------- DATA INPUT + PROCESSING ----------------------------------

TEXT_FILEPATH = "askubuntu/text_tokenized.txt"
TRAIN_FILEPATH = "askubuntu/train_random.txt"
EMBEDDINGS = "askubuntu/vector/vectors_pruned.200.txt"

BATCH_SIZE = 

id_to_text = {}
word_to_embedding = {}

# Sets the dictionary id_to_text, where id_to_text[id] of some given question id returns the text
# Is the concatenation of the title and the body, where the body is capped at 100 words
def get_id_to_text():
	with open(TEXT_FILEPATH, 'r') as f:
		for line in f.readlines():
			id, title, body = line.split("\t")
			id_to_text[id] = title + " " + body

# Sets the dictionary word_to_embedding, where word_to_embedding[word] of some given word
# returns the 200-length word embedding as provided
def get_word_to_embedding():
	with open(EMBEDDINGS, 'r') as f:
		for line in f.readlines():
			splits = line.split()
			word_to_embedding[splits[0]] = map(float, splits[1:])

# Returns all samples of training data
# The data is provided in the format (original_question), (LIST of positive_matches), (100 RANDOM negative_matches)
# For EACH original_question <-> positive_match pair, get 20 random negative samples
# Returns a list of training samples, where each training sample is of the format (q_id, pos_match, neg_match1, neg_match2, ..., neg_match20)
# Thus, each training sample is a tuple of length 22 (and these are all question IDs)
def get_training_data():
	samples = []
	with open(TRAIN_FILEPATH, 'r') as f:
		for line in f.readlines():
			id, pos, neg = line.split("\t")
			all_pos_matches = pos.split()
			all_neg_matches = neg.split()
			for pos_match in all_pos_matches:
				samples.append([id, pos_match] + random.sample(all_neg_matches, 20))
	return samples

# Look at this doc for more info on pack_padded_sequence: http://pytorch.org/docs/master/_modules/torch/nn/utils/rnn.html
# 
def set_batches(samples):


if __name__ == '__main__':
	pass




