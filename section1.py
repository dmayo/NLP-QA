import random
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import defaultdict
import numpy as np
import math
from time import time

torch.manual_seed(1)
random.seed(1)

TEXT_FILEPATH = "askubuntu/text_tokenized.txt"
TRAIN_FILEPATH = "askubuntu/train_random.txt"
EMBEDDINGS = "askubuntu/vector/vectors_pruned.200.txt"
DEV_FILEPATH = "askubuntu/dev.txt"

BATCH_SIZE = 1
EMBEDDING_DIM = 200
LSTM_HIDDEN_DIM = 240
NUM_TRAINING_SAMPLES = 22853
NUM_DEV_SAMPLES = 189

LEARNING_RATE = 6e-4 # might change later
WEIGHT_DECAY = 1e-5 # are we supposed to use this?
NUM_EPOCHS = 1

# -------------------------- DATA INPUT + PROCESSING ----------------------------------

# GLOBAL DICTIONARIES FOR DATA PROCESSING
id_to_title = {'0': ""} # keep a bogus mapping qid 0 -> empty string
id_to_body = {'0': ""}
word_to_index = defaultdict(lambda: 0) # by default, return 0, which corresponds to UNK or PAD and has embedding the zero vector

# Sets the dictionary id_to_title and id_to_body
def get_id_to_text():
    with open(TEXT_FILEPATH, 'r') as f:
        for line in f.readlines():
            id, title, body = line.split("\t")
            id_to_title[id] = title
            id_to_body[id] = " ".join(body.split()[:100])

# Returns the numpy array embeddings, which is of shape (num_embeddings, EMBEDDING_DIM)
# Sets the dictinoary word_to_index, where word_to_index[word] of some word returns the index within the embeddings numpy array
def get_word_embeddings():
    embedding_list = [[0] * 200] # set the zero vector for UNK or PADDING
    index = 1
    with open(EMBEDDINGS, 'r') as f:
        for line in f.readlines():
            splits = line.split()
            word_to_index[splits[0]] = index
            embedding_list.append(map(float, splits[1:]))
            index += 1
    return np.array(embedding_list)

# Returns all samples of training data
# The data is provided in the format (original_question), (LIST of positive_matches), (100 RANDOM negative_matches)
# For EACH original_question <-> positive_match pair, get 20 random negative samples
# RETURNS: a numpy array of training samples, where each training sample is of the format (q_id, pos_match, neg_match1, neg_match2, ..., neg_match20)
# This numpy array has shape (num_training_samples, 22)
def get_training_data():
    samples = []
    with open(TRAIN_FILEPATH, 'r') as f:
        for line in f.readlines():
            id, pos, neg = line.split("\t")
            all_pos_matches = pos.split()
            all_neg_matches = neg.split()
            for pos_match in all_pos_matches:
                samples.append([id, pos_match] + random.sample(all_neg_matches, 20))
    return np.array(samples)

# Flatten the list of questions (main_q, +, -), (main_q, +, -), ... into a single list
# We use BATCH_SIZE samples in a single batch, so this equates ot BATCH_SIZE * 22 questions in a single batch
# Inputs to the neural net are of the shape (max_question_length x batch_size), where batch_size = BATCH_SIZE * 22
# Pad missing questions with the 0 index (which corresponds to PAD or UNK)
# PARAMETERS: a list of BATCH_SIZE training samples, where each training sample is of the format (q_id, pos_match, neg_match1, neg_match2, ...)
# RETURNS: a Variable that we feed into the neural net, AND returns the numpy array of all question lengths (of length BATCH_SIZE * 22)
def get_tensor_from_batch(samples, use_title=True):
    d = id_to_title if use_title else id_to_body
    all_question_lengths = np.vectorize(lambda x: len(d[x].split()))(samples.flatten())
    max_question_length = np.amax(all_question_lengths)
    tensor = torch.zeros([max_question_length, BATCH_SIZE * len(samples[0])])
    for q_index, q_id in enumerate(samples.flatten()):
        for word_index, word in enumerate(d[q_id].split()):
            tensor[word_index][q_index] = word_to_index[word]
    return Variable(tensor.long()), all_question_lengths

# Returns the dev data in a numpy array, where each dev sample is of the format (q_id, candidate_1, candidate_2, ..., candidate_20, 0)
# Add an extra bogus question to keep this shape similar to the training data shape
# This numpy array has shape (num_dev_samples, 22)
# Look at https://github.com/taolei87/askubuntu for some more info
# Also returns a numpy array is_correct of the shape (num_dev_samples, 20), where is_correct[i][j] indicates whether the j'th candidate
# question for sample i is actually a similar question or not
def get_dev_data():
    dev_data = []
    is_correct = []
    with open(DEV_FILEPATH, 'r') as f:
        for line in f.readlines():
            id, similar_ids, candidate_qids, _ = line.split('\t')
            similar_ids = similar_ids.split()
            candidate_qids = candidate_qids.split()
            if len(similar_ids) != 0:
                dev_data.append([id] + candidate_qids + ['0'])
                is_correct.append([True if cand_qid in similar_ids else False for cand_qid in candidate_qids])
    return np.array(dev_data), np.array(is_correct)

# -------------------------- LSTM ----------------------------------

class LSTMQA(nn.Module):
    def __init__(self, hidden_dim, embedding_dim, pretrained_weight):
        super(LSTMQA, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim

        self.embed = nn.Embedding(len(pretrained_weight), self.embedding_dim, padding_idx=-1)
        self.embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
        # self.embed.weight.requires_grad = False # may make this better, not really sure. Using this would require parameters = filter(lambda p: p.requires_grad, net.parameters())

        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (Variable(torch.zeros(1, BATCH_SIZE * 22, self.hidden_dim)),
                Variable(torch.zeros(1, BATCH_SIZE * 22, self.hidden_dim)))

    def forward(self, sentence):
        # sentence is a Variable of a LongVector of shape (max_sentence_length, BATCH_SIZE)
        # returns a list of all the hidden states
        embeds = self.embed(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        return lstm_out

# Given a set of hidden states in the LSTM, and given a list of the question lengths, calculates
# the mean hidden state for every question.
def get_encodings(lstm_out, question_lengths):
    mean_hidden_state = Variable(torch.zeros(BATCH_SIZE * 22, LSTM_HIDDEN_DIM))
    # mean_hidden_state.requires_grad = True
    for word_index in range(len(lstm_out)):
        for q_index in range(len(lstm_out[0])):
            if word_index < question_lengths[q_index]:
                mean_hidden_state[q_index] = mean_hidden_state[q_index] + lstm_out[word_index][q_index]
    for q_index in range(len(mean_hidden_state)):
        mean_hidden_state[q_index] = mean_hidden_state[q_index] / question_lengths[q_index]
    return mean_hidden_state

# Generates the matrix X of shape (BATCH_SIZE, 21), where X[i][j] gives the cosine similarity
# between the main question i and the j-th question candidate, i.e. j=0 is the positive match and j=1~20 are the negative matches
# Takes the average of the title encoding and the body encoding
# The parameters are of shape (BATCH_SIZE * 22, LSTM_HIDDEN_DIM)
# y is just an all-zero vector of size (BATCH_SIZE)
def generate_score_matrix(title_encoding, body_encoding):
    mean_hidden_state = (title_encoding + body_encoding) / 2.
    # mean_hidden_state.requires_grad = True

    cos = nn.CosineSimilarity(dim=0)
    X = Variable(torch.zeros(BATCH_SIZE, 21))
    # X.requires_grad = True
    for i in range(BATCH_SIZE):
        for j in range(21):
            X[i, j] = cos(mean_hidden_state[22 * i], mean_hidden_state[22 * i + j + 1])
    y = Variable(torch.zeros(BATCH_SIZE).long())
    # y.requires_grad = True
    return X, y

# score_matrix is a shape (NUM_DEV_SAMPLES, 20) matrix that contains the cosine similarity scores
# score_matrix[i][j] contains the similarity score between the i'th sample's main question and its j'th candidate question
def evaluate_score_matrix(score_matrix):
    map_total = 0
    mrr_total = 0
    p_at_1_total = 0
    p_at_5_total = 0
    # TODO later

def train_LSTM():
    get_id_to_text()
    embeddings = get_word_embeddings()
    model = LSTMQA(LSTM_HIDDEN_DIM, EMBEDDING_DIM, embeddings)
    loss_function = nn.MultiMarginLoss(margin=0.2) # TODO: what about size_average?
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    orig_time = time()

    num_batches = int(math.ceil(1. * NUM_TRAINING_SAMPLES / BATCH_SIZE)) # TODO: what about the last batch where it's not full?
    for epoch in range(NUM_EPOCHS):
        samples = get_training_data() # recalculate this every epoch to get new random selections
        
        for i in range(num_batches):
            # Get the samples ready
            batch = samples[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
            title_tensor, title_lengths = get_tensor_from_batch(batch, use_title=True)
            body_tensor, body_lengths = get_tensor_from_batch(batch, use_title=False)

            # Reset the model
            optimizer.zero_grad()

            # Run our forward pass and get the entire sequence of hidden states
            model.hidden = model.init_hidden()
            title_lstm = model(title_tensor)
            title_encoding = get_encodings(title_lstm, title_lengths)
            model.hidden = model.init_hidden()
            body_lstm = model(body_tensor)
            body_encoding = get_encodings(body_lstm, body_lengths)

            # Compute loss, gradients, update parameters
            X, y = generate_score_matrix(title_encoding, body_encoding)
            loss = loss_function(X, y)
            loss.backward()
            optimizer.step()

            print "For batch number " + str(i) + " it has taken " + str(time() - orig_time) + " seconds"
    return model

def evaluate_LSTM(model):
    # samples has shape (num_dev_samples, 22), and is_correct has shape (num_dev_samples, 20)
    samples, is_correct = get_dev_data()

    num_batches = int(math.ceil(1. * NUM_DEV_SAMPLES / BATCH_SIZE)) # TODO: what about the last batch where it's not full?
    score_matrix = torch.Tensor()
    for i in range(num_batches):
        # Get the samples ready
        batch = samples[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
        title_tensor, title_lengths = get_tensor_from_batch(batch, use_title=True)
        body_tensor, body_lengths = get_tensor_from_batch(batch, use_title=False)

        # Run the model
        model.hidden = model.init_hidden()
        title_lstm = model(title_tensor)
        title_encoding = get_encodings(title_lstm, title_lengths)
        model.hidden = model.init_hidden()
        body_lstm = model(body_tensor)
        body_encoding = get_encodings(body_lstm, body_lengths)

        # Compute evaluation
        X, _ = generate_score_matrix(title_encoding, body_encoding)
        X = torch.index_select(X.data, 1, torch.arange(0, 20).long()) # convert to tensor, throw out last bogus question
        score_matrix = torch.cat([score_matrix, X])

    # score_matrix is a shape (num_dev_samples, 20) matrix that contains the cosine similarity scores
    evaluate_score_matrix(score_matrix)

if __name__ == '__main__':
    # Train LSTM
    model = train_LSTM()
    # evaluate_LSTM(model)
    # get_dev_data()

