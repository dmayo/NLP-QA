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
import random

torch.manual_seed(1)
random.seed(1)

USE_GPU = False
SAVE_MODELS = True # stores the models in lstm_models/epoch_0.txt
GPU_NUM=0 # sets which gpu to use

TEXT_FILEPATH_UBUNTU = "askubuntu/text_tokenized.txt"
TEXT_FILEPATH_ANDROID = "askubuntu/text_tokenized.txt"
TRAIN_FILEPATH = "askubuntu/train_random.txt"
EMBEDDINGS = "pruned_glove.txt"
DEV_FILEPATH = "askubuntu/dev.txt"
TEST_FILEPATH = "askubuntu/test.txt"
OUTPUT = "output.txt"

BATCH_SIZE = 20
EMBEDDING_DIM = 300
LSTM_HIDDEN_DIM = 240
CNN_HIDDEN_DIM = 667
CNN_KERNEL_SIZE = 3
DROPOUT = 0.2

LEARNING_RATE = 6e-4 # might change later
WEIGHT_DECAY = 1e-5 # are we supposed to use this?
NUM_EPOCHS = 15

# -------------------------- DATA INPUT + PROCESSING ----------------------------------

# GLOBAL DICTIONARIES FOR DATA PROCESSING
id_to_title_ubuntu = {'0': ""} # keep a bogus mapping qid 0 -> empty string
id_to_body_ubuntu = {'0': ""}
id_to_title_android = {'0': ""} # keep a bogus mapping qid 0 -> empty string
id_to_body_android = {'0': ""}
word_to_index = defaultdict(lambda: 0) # by default, return 0, which corresponds to UNK or PAD and has embedding the zero vector

id_to_index = {} # maps question ID to the index in the sparse Tf-idf weighted matrix
X = None # the sparse Tf-idf weighted matrix of shape (num_samples, num_features)

# Sets the dictionary id_to_title_ubuntu and id_to_body_ubuntu
def get_id_to_text_ubuntu():
    with open(TEXT_FILEPATH_UBUNTU, 'r') as f:
        for line in f.readlines():
            id, title, body = line.split("\t")
            id_to_title_ubuntu[id] = title
            id_to_body_ubuntu[id] = " ".join(body.split()[:100])

# Sets the dictionary id_to_title and id_to_body
def get_id_to_text_android():
    with open(TEXT_FILEPATH_ANDROID, 'r') as f:
        for line in f.readlines():
            id, title, body = line.split("\t")
            id_to_title_android[id] = title.lower()
            id_to_body_android[id] = (" ".join(body.split()[:100])).lower()


# Returns the numpy array embeddings, which is of shape (num_embeddings, EMBEDDING_DIM)
# Sets the dictinoary word_to_index, where word_to_index[word] of some word returns the index within the embeddings numpy array
def get_word_embeddings():
    embedding_list = [[0] * 300] # set the zero vector for UNK or PADDING
    index = 1
    with open(EMBEDDINGS, 'r') as f:
        for line in f.readlines():
            splits = line.split()
            word_to_index[splits[0]] = index
            embedding_list.append(map(float, splits[1:]))
            index += 1
    return np.array(embedding_list)

# Returns all samples of ubuntu training data
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

def get_android_samples(numSamples):
    samples = []
    for i in range(numSamples):
        samples.append(random.sample(id_to_title_android.keys(),22))
    return np.array(samples)


# Maps a question ID to the title concatenated with the body
# no limits on body length (as of yet)
def get_id_to_vector():
    global X
    all_questions = []
    index = 0
    with open(TEXT_FILEPATH, 'r') as f:
        for line in f:
            id, title, body = line[:-1].split("\t")
            all_questions.append(title + ' ' + body)
            id_to_index[id] = index
            index += 1
    vec = TfidfVectorizer()
    X = vec.fit_transform(all_questions)

# Creates the two dictionaries pos and neg
# pos[main_qid] maps to a set containing all positive matches to the main question ID
# neg does the same thing for negative matches
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
    return pos, neg


# Flatten the list of questions (main_q, +, -), (main_q, +, -), ... into a single list
# We use BATCH_SIZE samples in a single batch, so this equates ot BATCH_SIZE * 22 questions in a single batch
# Inputs to the neural net are of the shape (max_question_length x BATCH_SIZE * 22)
# Pad missing questions with the 0 index (which corresponds to PAD or UNK)
# PARAMETERS: a list of BATCH_SIZE training samples, where each training sample is of the format (q_id, pos_match, neg_match1, neg_match2, ...)
# RETURNS: a Variable that we feed into the neural net, AND returns the numpy array of all question lengths (of length BATCH_SIZE * 22)
def get_tensor_from_batch(samples_ubuntu, samples_android, use_title=True):
    d_ubuntu = id_to_title_ubuntu if use_title else id_to_body_ubuntu
    d_android = id_to_title_android if use_title else id_to_body_android
    all_question_lengths_ubuntu = np.vectorize(lambda x: len(d_ubuntu[x].split()))(samples_ubuntu.flatten())
    all_question_lengths_android = np.vectorize(lambda x: len(d_android[x].split()))(samples_android.flatten())
    max_question_length = max(np.amax(all_question_lengths_ubuntu),np.amax(all_question_lengths_android))
    tensor = torch.zeros([max_question_length, BATCH_SIZE * len(samples_ubuntu[0])*2])
    if USE_GPU:
        tensor = tensor.cuda(GPU_NUM)
    for q_index, q_id in enumerate(samples_ubuntu.flatten()):
        for word_index, word in enumerate(d_ubuntu[q_id].split()):
            tensor[word_index][q_index] = word_to_index[word]
    for q_index, q_id in enumerate(samples_android.flatten()):
        for word_index, word in enumerate(d_android[q_id].split()):
            tensor[word_index][BATCH_SIZE * len(samples_ubuntu[0])+q_index] = word_to_index[word]
    return Variable(tensor.long()), all_question_lengths_ubuntu.concatenate(all_question_lengths_android)

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

def get_training_part(mean_hidden_state,y_d):
    out=[]
    for i in range(y_d):
        if(y_d[i]==0):
            out.append(mean_hidden_state[i])
    return torch.FloatTensor(out)

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

class GRL(torch.autograd.Function):
    def forward(self, x):
        return x.view_as(x)
    def backward(self, grad_output):
        return (grad_output * -LAMDA) # need tune

class CNN_Feature_Extractor(nn.Module):
    def __init__(self, pretrained_weight):
        super(CNN_Feature_Extractor, self).__init__()

        self.embed = nn.Embedding(len(pretrained_weight), EMBEDDING_DIM)
        pretrained_weight = torch.Tensor(pretrained_weight).cuda(GPU_NUM) if USE_GPU else torch.Tensor(pretrained_weight)
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

# three fully connected layers
# x->1024->1024->2
class NN_Domain_Classifier(nn.Module):
    def __init__(self,):
        super(NN_Domain_Classifier, self).__init__()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def init_hidden(self):
        pass

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Actually trains this thing
def train_model(use_lstm=True):
    if use_lstm:
        print_and_write("Training the LSTM model with the GPU:" if USE_GPU else "Training the LSTM model:")
    else:
        print_and_write("Training the CNN model with the GPU:" if USE_GPU else "Training the CNN model")

    get_id_to_text_ubuntu()
    get_id_to_text_android()

    embeddings = get_word_embeddings()
    '''
    model_Feature_Extractor = LSTMQA(embeddings) if use_lstm else CNN_Feature_Extractor(embeddings)
    if USE_GPU:
        model.cuda(GPU_NUM)
    '''
    model_Feature_Extractor = CNN_Feature_Extractor(embeddings)
    model_Domain_Classifier = NN_Domain_Classifier()

    #domain classifier loss
    L_d_function = nn.MultiMarginLoss(margin=0.2) #binomial cross entropy loss
    L_y_function = nn.MultiMarginLoss(margin=0.2) #logistic regression loss
    
    optimizer_L_d = optim.Adam(filter(lambda x: x.requires_grad, model_Domain_Classifier.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    optimizer_L_f = optim.Adam(filter(lambda x: x.requires_grad, model_Feature_Extractor.parameters()), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    orig_time = time()

    for epoch in range(NUM_EPOCHS):
        ubuntu_samples = get_training_data() # recalculate this every epoch to get new random selections
        android_samples = get_android_samples(len(ubuntu_samples))
        num_samples = 2*len(ubuntu_samples)

        num_batches = int(math.ceil(1. * num_samples / BATCH_SIZE))
        total_loss = 0 # used for debugging
        for i in range(num_batches):
            # Get the samples ready, 50% of samples from ubuntu and 50% from android
            batch_ubuntu = ubuntu_samples[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
            batch_android = android_samples[i * BATCH_SIZE: (i+1) * BATCH_SIZE]
            # If this is the last batch, then need to pad the batch to get the same shape as expected
            if i == num_batches - 1 and num_samples % BATCH_SIZE != 0:
                batch_ubuntu = np.concatenate((batch_ubuntu, np.full(((i+1) * BATCH_SIZE - num_samples, 22), "0")), axis=0)
                batch_android = np.concatenate((batch_android, np.full(((i+1) * BATCH_SIZE - num_samples, 22), "0")), axis=0)

            ########### Set up the feature extractor network ###########
            # Convert from numpy arrays to tensors
            title_tensor, title_lengths = get_tensor_from_batch(batch_ubuntu, batch_android, use_title=True)
            body_tensor, body_lengths = get_tensor_from_batch(batch_ubuntu, batch_android, use_title=False)

            # Reset the model
            optimizer_L_d.zero_grad() # where should these be reset?
            optimizer_L_f.zero_grad()

            model_Feature_Extractor.hidden = model_Feature_Extractor.init_hidden()
            # Run our forward pass and get the entire sequence of hidden states
            title_hidden = model_Feature_Extractor(title_tensor)
            title_encoding = get_encodings(title_hidden, title_lengths, use_lstm=use_lstm)
            model_Feature_Extractor.hidden = model_Feature_Extractor.init_hidden()
            body_hidden = model_Feature_Extractor(body_tensor)
            body_encoding = get_encodings(body_hidden, body_lengths, use_lstm=use_lstm)
            mean_hidden_state = (title_encoding + body_encoding) / 2.

            ########### Set up the domain classifier network ###########

            model_Domain_Classifier.hidden = model_Domain_classifier.init_hidden()
            Domain_Classifier_hidden=model_Domain_Classifier(mean_hidden_state)

            y_d = torch.zeros(BATCH_SIZE/2).concat(torch.ones(BATCH_SIZE/2))

            L_d_loss = L_d_function(Domain_Classifier_hidden,y_d)
            total_L_d_loss += L_d_loss.data[0]



            ########### Set up the label predictor network ###########
            mean_hidden_state_training = get_training_part(mean_hidden_state,y_d)
            # Compute loss, gradients, update parameters
            X_y, y_y = generate_score_matrix(mean_hidden_state_training)

            L_y_loss = L_y_function(X_y,y_y)
            total_L_y_loss += L_y_loss.data[0]


            #L_d_loss.backward()
            #L_y_loss.backward()
            (L_y-lambda1*L_d).backward()
            optimizer_L_d.step()
            optimizer_L_f.step()

            total_loss += loss.data[0]


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
    samples, is_correct = get_dev_data(use_test_data=use_test_data)
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
    #train_model(use_lstm=True)
    #print_and_write("\n\n\n\n\n\n")
    train_model(use_lstm=False)
