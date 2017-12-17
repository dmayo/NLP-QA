# Uses unsupervised methods on the Android data set in order to rank the pos/neg pairs
# Uses TfIdF weighted BoW vectors (all with cosine similarity) to rank
# Measures success with the AUC metric (also with MAP, MRR, P@1?)

import numpy as np
from meter import AUCMeter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from collections import defaultdict
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

TEXT_FILEPATH = "Android/corpus.tsv"
DEV_FILEPATH_POS = "Android/dev.pos.txt"
DEV_FILEPATH_NEG = "Android/dev.neg.txt"
TEST_FILEPATH_POS = "Android/test.pos.txt"
TEST_FILEPATH_NEG = "Android/test.neg.txt"

# GLOBAL DICTIONARIES FOR DATA PROCESSING
id_to_index = {} # maps question ID to the index in the sparse Tf-idf weighted matrix
X = None # the sparse Tf-idf weighted matrix of shape (num_samples, num_features)

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

# Gets the cosine distance between two vectors x and y and maps the result to [0, 1]
# 0 means that the two vectors are opposite, 1 means that they are the same
# x and y are sparse vectors
def cosine(x, y):
    return (1 + cosine_similarity(sparse.vstack([x, y]))[0][1]) / 2

if __name__ == '__main__':
    get_id_to_vector()
    meter = AUCMeter()

    pos, neg = get_dev_data_android(use_test_data=False)
    # Only use questions that have at least one positive match
    for main_qid in pos:
        main_vector = X[id_to_index[main_qid]]
        similiarities, targets = [], []
        # For all positive matches, append similarity score + a 1 on the targets
        for pos_match_qid in pos[main_qid]:
            similiarities.append(cosine(main_vector, X[id_to_index[pos_match_qid]]))
            targets.append(1)
        # For all negative matches, append similarity score + a 0 on the targets
        for neg_match_qid in neg[main_qid]:
            similiarities.append(cosine(main_vector, X[id_to_index[neg_match_qid]]))
            targets.append(0)
        meter.add(np.array(similiarities), np.array(targets))
    print "The AUC(0.05) value on the TfIdF weighted vectors are " + str(meter.value(0.05))
