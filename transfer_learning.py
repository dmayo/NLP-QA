# Uses unsupervised methods on the Android data set in order to rank the pos/neg pairs
# Uses TfIdF weighted BoW vectors (all with cosine similarity) to rank
# Measures success with the AUC metric (also with MAP, MRR, P@1?)

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
from meter import AUCMeter

torch.manual_seed(1)
random.seed(1)

USE_GPU = True

TEXT_FILEPATH = "Android/corpus.tsv"
DEV_FILEPATH_POS = "Android/dev.pos.txt"
DEV_FILEPATH_NEG = "Android/dev.neg.txt"
TEST_FILEPATH_POS = "Android/test.pos.txt"
TEST_FILEPATH_NEG = "Android/test.neg.txt"
EMBEDDINGS = "pruned_glove.txt"
CHECKPOINT_FILENAME = "glove_lstm/epoch_1.txt"
OUTPUT = "transfer_learning.txt"

BATCH_SIZE = 20
EMBEDDING_DIM = 300
LSTM_HIDDEN_DIM = 240
CNN_HIDDEN_DIM = 667
CNN_KERNEL_SIZE = 3
DROPOUT = 0.1

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
    embedding_list = [[0] * EMBEDDING_DIM] # set the zero vector for UNK or PADDING
    index = 1
    with open(EMBEDDINGS, 'r') as f:
        for line in f.readlines():
            splits = line.split()
            word_to_index[splits[0]] = index
            embedding_list.append(map(float, splits[1:]))
            index += 1
    return np.array(embedding_list)

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def print_and_write(text):
    print text
    with open(OUTPUT, "a") as f:
        f.write(text)
        f.write("\n")

# Flatten the list of questions (main_q, +, -), (main_q, +, -), ... into a single list
# We use BATCH_SIZE samples in a single batch, so this equates ot BATCH_SIZE * 22 questions in a single batch
# Inputs to the neural net are of the shape (max_question_length x BATCH_SIZE * 22)
# Pad missing questions with the 0 index (which corresponds to PAD or UNK)
# PARAMETERS: a list of BATCH_SIZE training samples, where each training sample is of the format (q_id, pos_match, neg_match1, neg_match2, ...)
# RETURNS: a Variable that we feed into the neural net, AND returns the numpy array of all question lengths (of length BATCH_SIZE * 22)
def get_tensor_from_batch(samples, use_title=True):
    d = id_to_title if use_title else id_to_body
    all_question_lengths = np.vectorize(lambda x: len(d[x].split()))(samples.flatten())
    max_question_length = np.amax(all_question_lengths)
    tensor = torch.zeros([max_question_length, BATCH_SIZE * len(samples[0])])
    if USE_GPU:
        tensor = tensor.cuda()
    for q_index, q_id in enumerate(samples.flatten()):
        for word_index, word in enumerate(d[q_id].split()):
            tensor[word_index][q_index] = word_to_index[word]
    return Variable(tensor.long()), all_question_lengths

# Returns the dev data in a numpy array, where each dev sample is of the format (q_id, pos_match, neg_match1, neg_match2, ..., neg_match20)
# This numpy array has shape (num_dev_samples, 22)
# Can use the flag use_test_data to get data from the test filepath instead
def get_dev_data_android(use_test_data=False):
    filepath_pos = TEST_FILEPATH_POS if use_test_data else DEV_FILEPATH_POS
    filepath_neg = TEST_FILEPATH_NEG if use_test_data else DEV_FILEPATH_NEG
    pos = defaultdict(lambda: set())
    neg = defaultdict(lambda: set())
    with open(filepath_pos, 'r') as f:
        for line in f.readlines():
            main_qid, candidate_qid = line.split()
            pos[main_qid].add(candidate_qid)
    with open(filepath_neg, 'r') as f:
        for line in f.readlines():
            main_qid, candidate_qid = line.split()
            neg[main_qid].add(candidate_qid)
    samples = []
    for main_q in pos:
        for pos_match in pos[main_q]:
            samples.append([main_q, pos_match] + random.sample(neg[main_q], 20))
    return np.array(samples)

# Generates the matrix X of shape (BATCH_SIZE, 21), where X[i][j] gives the cosine similarity
# between the main question i and the j-th question candidate, i.e. j=0 is the positive match and j=1~20 are the negative matches
# Takes the average of the title encoding and the body encoding
# The parameters are of shape (BATCH_SIZE * 22, HIDDEN_DIM)
# y is just an all-zero vector of size (BATCH_SIZE)
def generate_score_matrix(title_encoding, body_encoding):
    mean_hidden_state = (title_encoding + body_encoding) / 2.
    # mean_hidden_state.requires_grad = True

    cos = nn.CosineSimilarity(dim=0)
    X = Variable(torch.zeros(BATCH_SIZE, 21).cuda()) if USE_GPU else Variable(torch.zeros(BATCH_SIZE, 21))
    # X.requires_grad = True
    for i in range(BATCH_SIZE):
        for j in range(21):
            X[i, j] = cos(mean_hidden_state[22 * i], mean_hidden_state[22 * i + j + 1])
    y = Variable(torch.zeros(BATCH_SIZE).long().cuda()) if USE_GPU else Variable(torch.zeros(BATCH_SIZE).long())
    # y.requires_grad = True
    return X, y

class LSTMQA(nn.Module):
    def __init__(self, pretrained_weight):
        super(LSTMQA, self).__init__()

        self.embed = nn.Embedding(len(pretrained_weight), EMBEDDING_DIM)
        pretrained_weight = torch.from_numpy(pretrained_weight).cuda() if USE_GPU else torch.from_numpy(pretrained_weight)
        self.embed.weight.data.copy_(pretrained_weight)
        self.embed.weight.requires_grad = False # may make this better, not really sure. Using this would require parameters = filter(lambda p: p.requires_grad, net.parameters())

        # Use LSTM_HIDDEN_DIM/2 because this is bidirectional
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_HIDDEN_DIM / 2, bidirectional=True)
        self.dropout = nn.Dropout(p=DROPOUT) 
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if USE_GPU:
            return (Variable(torch.zeros(2, BATCH_SIZE * 22, LSTM_HIDDEN_DIM / 2).cuda()), 
                    Variable(torch.zeros(2, BATCH_SIZE * 22, LSTM_HIDDEN_DIM / 2).cuda()))
        else:
            return (Variable(torch.zeros(2, BATCH_SIZE * 22, LSTM_HIDDEN_DIM / 2)), 
                    Variable(torch.zeros(2, BATCH_SIZE * 22, LSTM_HIDDEN_DIM / 2)))

    def forward(self, sentence):
        # sentence is a Variable of a LongVector of shape (max_sentence_length, BATCH_SIZE * 22)
        # returns a list of all the hidden states, is of shape (max_question_length, BATCH_SIZE * 22, LSTM_HIDDEN_DIM)
        embeds = self.embed(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        return self.dropout(lstm_out)

# Evaluates the model on the dev set data
def evaluate_model(model, use_test_data=False, use_lstm=True):
    if use_test_data:
        print_and_write("Running evaluate on the TEST data:")
    else:
        print_and_write("Running evaluate on the DEV data:")
    # Set the model to eval mode
    model.eval()

    # samples has shape (num_dev_samples, 22)
    samples = get_dev_data_android(use_test_data=use_test_data)
    num_samples = len(samples)

    num_batches = int(math.ceil(1. * num_samples / BATCH_SIZE))
    for i in range(num_batches):
        # Get the samples ready
        batch = samples[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
        # If this is the last batch, then need to pad the batch to get the same shape as expected
        if i == num_batches - 1 and num_samples % BATCH_SIZE != 0:
            batch = np.concatenate((batch, np.full(((i+1) * BATCH_SIZE - num_samples, 22), "0")), axis=0)

        # Convert from numpy arrays to tensors
        title_tensor, title_lengths = get_tensor_from_batch(batch, use_title=True)
        body_tensor, body_lengths = get_tensor_from_batch(batch, use_title=False)

        # Run the model
        model.hidden = model.init_hidden()
        title_lstm = model(title_tensor)
        title_encoding = get_encodings(title_lstm, title_lengths, use_lstm=use_lstm)
        model.hidden = model.init_hidden()
        body_lstm = model(body_tensor)
        body_encoding = get_encodings(body_lstm, body_lengths, use_lstm=use_lstm)

        # Compute evaluation
        X, _ = generate_score_matrix(title_encoding, body_encoding)
        X = X.data
        if i == num_batches - 1 and num_samples % BATCH_SIZE != 0:
            score_matrix = torch.cat([score_matrix, X[:num_samples - i * BATCH_SIZE]])
        else:
            score_matrix = torch.cat([score_matrix, X])

    # score_matrix is a shape (num_dev_samples, 21) matrix that contains the cosine similarity scores
    meter = AUCMeter()
    similarities, targets = [], []
    for i in range(len(score_matrix)):
        similarities.append(score_matrix[i][0])
        targets.append(1)
        for j in range(1, 21):
            similarities.append(score_matrix[i][j])
            targets.append(0)
    meter.add(similarities, targets)
    print "The AUC(0.05) value is " + str(meter.value(0.05))

    # Set the model back to train mode
    model.train()

if __name__ == '__main__':
    get_id_to_text()
    embeddings = get_word_embeddings()
    model = LSTMQA(embeddings)
    optim = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()))
    load_checkpoint(CHECKPOINT_FILENAME, model, optim)
    print "hello"
    evaluate_model(model)

