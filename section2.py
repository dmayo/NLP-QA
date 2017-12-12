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

USE_GPU = True
SAVE_MODELS = True # stores the models in lstm_models/epoch_0.txt
GPU_NUM=0 # sets which gpu to use

TEXT_FILEPATH = "android/text_tokenized.txt"
TRAIN_FILEPATH = "askubuntu/train_random.txt"
EMBEDDINGS = "word_vectors.txt"
DEV_FILEPATH = "android/dev.txt"
DEV_FILEPATH_POS = "android/dev.pos.txt"
DEV_FILEPATH_NEG = "android/dev.neg.txt"
TEST_FILEPATH = "askubuntu/test.txt"
TEST_FILEPATH_POS = "android/test.pos.txt"
TEST_FILEPATH_NEG = "android/test.neg.txt"
OUTPUT = "output.txt"

BATCH_SIZE = 20
EMBEDDING_DIM = 200
LSTM_HIDDEN_DIM = 240
CNN_HIDDEN_DIM = 667
CNN_KERNEL_SIZE = 3
DROPOUT = 0.2

LEARNING_RATE = 6e-4 # might change later
WEIGHT_DECAY = 1e-5 # are we supposed to use this?
NUM_EPOCHS = 15

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
        tensor = tensor.cuda(GPU_NUM)
    for q_index, q_id in enumerate(samples.flatten()):
        for word_index, word in enumerate(d[q_id].split()):
            tensor[word_index][q_index] = word_to_index[word]
    return Variable(tensor.long()), all_question_lengths

# Given a set of hidden states in the neural net, and given a list of the question lengths, calculates
# the mean hidden state for every question.
# nn_out is of shape (max_question_length, BATCH_SIZE * 22, HIDDEN_DIM) for LSTMs
# and is (BATCH_SIZE * 22, HIDDEN_DIM, max_question_length) for CNNs
# question_lengths is a simple numpy array of length (BATCH_SIZE * 22)
def get_encodings(nn_out, question_lengths, use_lstm=True):
    # changes the dimensions to shape (BATCH_SIZE * 22, max_question_length, HIDDEN_DIM)
    nn_out = nn_out.permute(1, 0, 2) if use_lstm else nn_out.permute(0, 2, 1)
    HIDDEN_DIM = len(nn_out[0][0])
    # Generate a mask of shape (BATCH_SIZE * 22, max_question_length, HIDDEN_DIM)
    mask = torch.zeros(BATCH_SIZE * 22, len(nn_out[0]), HIDDEN_DIM).cuda(GPU_NUM) if USE_GPU else torch.zeros(BATCH_SIZE * 22, len(nn_out[0]), HIDDEN_DIM)
    for q_index in range(len(question_lengths)):
        length = question_lengths[q_index]
        if length != 0:
            mask[q_index][:length] = torch.ones(length, HIDDEN_DIM)
    nn_out = nn_out * Variable(mask)

    mean_hidden_state = Variable(torch.zeros(BATCH_SIZE * 22, HIDDEN_DIM).cuda(GPU_NUM)) if USE_GPU else Variable(torch.zeros(BATCH_SIZE * 22, HIDDEN_DIM))
    # mean_hidden_state.requires_grad = True
    for q_index in range(len(nn_out)):
        if question_lengths[q_index] != 0:
            mean_hidden_state[q_index] = torch.sum(nn_out[q_index], 0) / question_lengths[q_index]
    return mean_hidden_state

# Saves the model in a file called lstm_models/epoch_0.txt
def save_checkpoint(epoch, model, optimizer, use_lstm):
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    filename = ('lstm' if use_lstm else "cnn") + '_models/epoch_' + str(epoch) + ".txt"
    torch.save(state, filename)

def load_checkpoint(filename, model, optimizer):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def print_and_write(text):
    print text
    with open(OUTPUT, "a") as f:
        f.write(text)
        f.write("\n")

# -------------------------- EVALUATION RELATED CODE ----------------------------------

# Returns the dev data in a numpy array, where each dev sample is of the format (q_id, candidate_1, candidate_2, ..., candidate_20, 0)
# Add an extra bogus question to keep this shape similar to the training data shape
# This numpy array has shape (num_dev_samples, 22)
# Look at https://github.com/taolei87/askubuntu for some more info
# Also returns a numpy array is_correct of the shape (num_dev_samples, 20), where is_correct[i][j] indicates whether the j'th candidate
# question for sample i is actually a similar question or not
# Can use the flag use_test_data to get data from the test filepath instead
def get_dev_data(use_test_data=False):
    dev_data = []
    is_correct = []
    filepath = TEST_FILEPATH if use_test_data else DEV_FILEPATH
    with open(filepath, 'r') as f:
        for line in f.readlines():
            id, similar_ids, candidate_qids, _ = line.split('\t')
            similar_ids = similar_ids.split()
            candidate_qids = candidate_qids.split()
            if len(similar_ids) != 0:
                dev_data.append([id] + candidate_qids + ['0'])
                is_correct.append([True if cand_qid in similar_ids else False for cand_qid in candidate_qids])
    return np.array(dev_data), np.array(is_correct)


# Returns the dev data in a numpy array, where each dev sample is of the format (q_id, candidate_1, candidate_2, ..., candidate_20, 0)
# Add an extra bogus question to keep this shape similar to the training data shape
# This numpy array has shape (num_dev_samples, 22)
# Look at https://github.com/taolei87/askubuntu for some more info
# Also returns a numpy array is_correct of the shape (num_dev_samples, 20), where is_correct[i][j] indicates whether the j'th candidate
# question for sample i is actually a similar question or not
# Can use the flag use_test_data to get data from the test filepath instead
def get_dev_data_android(use_test_data=False):
    dev_data = []
    is_correct = []
    filepath_pos = TEST_FILEPATH_POS if use_test_data else DEV_FILEPATH_POS
    filepath_neg = TEST_FILEPATH_NEG if use_test_data else DEV_FILEPATH_NEG
    pos={}
    #neg={}
    total={}
    with open(filepath_pos, 'r') as f:
        for line in f.readlines():
            similar_ids, candidate_qids, _ = line.split('\t')
            pos.setdefault(similar_ids, [candidate_qids]).append(candidate_qids)
            total.setdefault(similar_ids, [candidate_qids]).append(candidate_qids)
    with open(filepath_neg, 'r') as f:
        for line in f.readlines():
            similar_ids, candidate_qids, _ = line.split('\t')
            #neg.setdefault(similar_ids, [candidate_qids]).append(candidate_qids)
            total.setdefault(similar_ids, [candidate_qids]).append(candidate_qids)
    for key in total:
        dev_data.append([key] + total[key] + ['0'])
        is_correct.append([True if cand_qid in pos else False for cand_qid in total[key]])
    return np.array(dev_data), np.array(is_correct)

# Generates the matrix X of shape (BATCH_SIZE, 21), where X[i][j] gives the cosine similarity
# between the main question i and the j-th question candidate, i.e. j=0 is the positive match and j=1~20 are the negative matches
# Takes the average of the title encoding and the body encoding
# The parameters are of shape (BATCH_SIZE * 22, HIDDEN_DIM)
# y is just an all-zero vector of size (BATCH_SIZE)
def generate_score_matrix(title_encoding, body_encoding):
    mean_hidden_state = (title_encoding + body_encoding) / 2.
    # mean_hidden_state.requires_grad = True

    cos = nn.CosineSimilarity(dim=0)
    X = Variable(torch.zeros(BATCH_SIZE, 21).cuda(GPU_NUM)) if USE_GPU else Variable(torch.zeros(BATCH_SIZE, 21))
    # X.requires_grad = True
    for i in range(BATCH_SIZE):
        for j in range(21):
            X[i, j] = cos(mean_hidden_state[22 * i], mean_hidden_state[22 * i + j + 1])
    y = Variable(torch.zeros(BATCH_SIZE).long().cuda(GPU_NUM)) if USE_GPU else Variable(torch.zeros(BATCH_SIZE).long())
    # y.requires_grad = True
    return X, y

# In these functions, sorted_args and is_correct are for ONE SAMPLE ONLY - so both are 1D arrays of length 20
def calculate_precision_at(sorted_args, is_correct, precision_value):
    return 1. * sum([is_correct[index] for index in sorted_args[:precision_value]]) / precision_value
def calculate_map(sorted_args, is_correct):
    num_total_correct = sum(is_correct)
    num_found, total = 0, 0
    for i in range(20):
        if is_correct[sorted_args[i]]:
            num_found += 1
            total += 1. * num_found / (i + 1)
            if num_found == num_total_correct:
                return total / num_total_correct
def calculate_mrr(sorted_args, is_correct):
    for i in range(20):
        if is_correct[sorted_args[i]]:
            return 1. / (i+1)

# score_matrix is a numpy array shape (num_dev_samples, 20) matrix that contains the cosine similarity scores
# score_matrix[i][j] contains the similarity score between the i'th sample's main question and its j'th candidate question
# is_correct of the numpy array shape (num_dev_samples, 20), where is_correct[i][j] indicates whether the j'th candidate
# question for sample i is actually a similar question or not
def evaluate_score_matrix_and_print(score_matrix, is_correct):
    map_total, mrr_total, p_at_1_total, p_at_5_total = 0, 0, 0, 0
    num_samples = len(score_matrix)
    sorted_args = np.argsort(-score_matrix)
    # sorted_args = np.repeat(np.arange(0, 20)[np.newaxis, :], num_samples, axis=0) # uncomment this to find BM25 values
    for i in range(num_samples):
        map_total += calculate_map(sorted_args[i], is_correct[i])
        mrr_total += calculate_mrr(sorted_args[i], is_correct[i])
        p_at_1_total += calculate_precision_at(sorted_args[i], is_correct[i], 1)
        p_at_5_total += calculate_precision_at(sorted_args[i], is_correct[i], 5)
    print_and_write("MAP score is " + str(map_total / num_samples))
    print_and_write("MRR score is " + str(mrr_total / num_samples))
    print_and_write("P@1 score is " + str(p_at_1_total / num_samples))
    print_and_write("P@5 score is " + str(p_at_5_total / num_samples))

# -------------------------- MODEL DEFINITIONS ----------------------------------

class LSTMQA(nn.Module):
    def __init__(self, pretrained_weight):
        super(LSTMQA, self).__init__()

        self.embed = nn.Embedding(len(pretrained_weight), EMBEDDING_DIM)
        pretrained_weight = torch.from_numpy(pretrained_weight).cuda(GPU_NUM) if USE_GPU else torch.from_numpy(pretrained_weight)
        self.embed.weight.data.copy_(pretrained_weight)
        self.embed.weight.requires_grad = False # may make this better, not really sure. Using this would require parameters = filter(lambda p: p.requires_grad, net.parameters())

        # Use LSTM_HIDDEN_DIM/2 because this is bidirectional
        self.lstm = nn.LSTM(EMBEDDING_DIM, LSTM_HIDDEN_DIM / 2, bidirectional=True)
        self.dropout = nn.Dropout(p=DROPOUT) 
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        if USE_GPU:
            return (Variable(torch.zeros(2, BATCH_SIZE * 22, LSTM_HIDDEN_DIM / 2).cuda(GPU_NUM)), 
                    Variable(torch.zeros(2, BATCH_SIZE * 22, LSTM_HIDDEN_DIM / 2).cuda(GPU_NUM)))
        else:
            return (Variable(torch.zeros(2, BATCH_SIZE * 22, LSTM_HIDDEN_DIM / 2)), 
                    Variable(torch.zeros(2, BATCH_SIZE * 22, LSTM_HIDDEN_DIM / 2)))

    def forward(self, sentence):
        # sentence is a Variable of a LongVector of shape (max_sentence_length, BATCH_SIZE * 22)
        # returns a list of all the hidden states, is of shape (max_question_length, BATCH_SIZE * 22, LSTM_HIDDEN_DIM)
        embeds = self.embed(sentence)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        return self.dropout(lstm_out)

class CNNQA(nn.Module):
    def __init__(self, pretrained_weight):
        super(CNNQA, self).__init__()

        self.embed = nn.Embedding(len(pretrained_weight), EMBEDDING_DIM)
        pretrained_weight = torch.from_numpy(pretrained_weight).cuda(GPU_NUM) if USE_GPU else torch.from_numpy(pretrained_weight)
        self.embed.weight.data.copy_(pretrained_weight)
        self.embed.weight.requires_grad = False # may make this better, not really sure. Using this would require parameters = filter(lambda p: p.requires_grad, net.parameters())
        
        self.cnn = nn.Conv1d(EMBEDDING_DIM, CNN_HIDDEN_DIM, CNN_KERNEL_SIZE, padding=(CNN_KERNEL_SIZE - 1) / 2)
        self.dropout = nn.Dropout(p=DROPOUT)
        self.hidden = None # doesn't actually matter, used for consistentcy between the two models

    def init_hidden(self):
        pass

    def forward(self, sentence):
        # sentence is a Variable of a LongVector of shape (max_sentence_length, BATCH_SIZE * 22)
        # returns a list of all the hidden states, is of shape ()
        embeds = self.embed(sentence) # currently shape (max_question_length, BATCH_SIZE * 22, EMBEDDING_DIM)
        embeds = embeds.permute(1, 2, 0) # now (BATCH_SIZE * 22, EMBEDDING_DIM, max_question_length)

        cnn_out = self.cnn(embeds) # shape (BATCH_SIZE * 22, CNN_HIDDEN_DIM, max_question_length)
        return self.dropout(cnn_out)

# Actually trains this thing
def train_model(use_lstm=True):
    if use_lstm:
        print_and_write("Training the LSTM model with the GPU:" if USE_GPU else "Training the LSTM model:")
    else:
        print_and_write("Training the CNN model with the GPU:" if USE_GPU else "Training the CNN model")

    get_id_to_text()
    embeddings = get_word_embeddings()
    model = LSTMQA(embeddings) if use_lstm else CNNQA(embeddings)
    if USE_GPU:
        model.cuda(GPU_NUM)
    loss_function = nn.MultiMarginLoss(margin=0.2) # TODO: what about size_average?
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    orig_time = time()

    for epoch in range(NUM_EPOCHS):
        samples = get_training_data() # recalculate this every epoch to get new random selections
        num_samples = len(samples)

        num_batches = int(math.ceil(1. * num_samples / BATCH_SIZE))
        total_loss = 0 # used for debugging
        for i in range(num_batches):
            # Get the samples ready
            batch = samples[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
            # If this is the last batch, then need to pad the batch to get the same shape as expected
            if i == num_batches - 1 and num_samples % BATCH_SIZE != 0:
                batch = np.concatenate((batch, np.full(((i+1) * BATCH_SIZE - num_samples, 22), "0")), axis=0)

            # Convert from numpy arrays to tensors
            title_tensor, title_lengths = get_tensor_from_batch(batch, use_title=True)
            body_tensor, body_lengths = get_tensor_from_batch(batch, use_title=False)

            # Reset the model
            optimizer.zero_grad()

            # Run our forward pass and get the entire sequence of hidden states
            model.hidden = model.init_hidden()
            title_hidden = model(title_tensor)
            title_encoding = get_encodings(title_hidden, title_lengths, use_lstm=use_lstm)
            model.hidden = model.init_hidden()
            body_hidden = model(body_tensor)
            body_encoding = get_encodings(body_hidden, body_lengths, use_lstm=use_lstm)
            # Compute loss, gradients, update parameters
            # Could potentially do something about the last batch, but prolly won't affect training that much
            X, y = generate_score_matrix(title_encoding, body_encoding)
            loss = loss_function(X, y)
            total_loss += loss.data[0]
            loss.backward()
            optimizer.step()

            # every so while, check the dev accuracy
            # if i % 10 == 0:
            #     print_and_write("For batch number " + str(i) + " it has taken " + str(time() - orig_time) + " seconds and has loss " + str(total_loss))
            # if i > 0 and i % 100 == 0:
            #     evaluate_model(model, use_lstm=use_lstm)
        print_and_write("For epoch number " + str(epoch) + " it has taken " + str(time() - orig_time) + " seconds and has loss " + str(total_loss))
        evaluate_model(model, use_lstm=use_lstm)
        evaluate_model(model, use_test_data=True, use_lstm=use_lstm)
        if SAVE_MODELS:
            save_checkpoint(epoch, model, optimizer, use_lstm)
    return model

# Evaluates the model on the dev set data
def evaluate_model(model, use_test_data=False, use_lstm=True):
    if use_test_data:
        print_and_write("Running evaluate on the TEST data:")
    else:
        print_and_write("Running evaluate on the DEV data:")
    # Set the model to eval mode
    model.eval()

    # samples has shape (num_dev_samples, 22), and is_correct has shape (num_dev_samples, 20)
    samples, is_correct = get_dev_data_android(use_test_data=use_test_data)
    num_samples = len(samples)

    num_batches = int(math.ceil(1. * num_samples / BATCH_SIZE))
    score_matrix = torch.Tensor().cuda(GPU_NUM) if USE_GPU else torch.Tensor()
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
        X = torch.index_select(X.data, 1, torch.arange(0, 20).long().cuda(GPU_NUM) if USE_GPU else torch.arange(0,20).long()) # convert to tensor, throw out last bogus question
        if i == num_batches - 1 and num_samples % BATCH_SIZE != 0:
            score_matrix = torch.cat([score_matrix, X[:num_samples - i * BATCH_SIZE]])
        else:
            score_matrix = torch.cat([score_matrix, X])

    # score_matrix is a shape (num_dev_samples, 20) matrix that contains the cosine similarity scores
    evaluate_score_matrix_and_print(score_matrix.cpu().numpy(), is_correct)

    # Set the model back to train mode
    model.train()

if __name__ == '__main__':
    # Train our two models
    train_model(use_lstm=True)
    print_and_write("\n\n\n\n\n\n")
    train_model(use_lstm=False)
