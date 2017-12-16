# Used to prune the standard glove embeddings to only include words found in the two datasets

ANDROID_TEXT_FILEPATH = "Android/corpus.tsv"
UBUNTU_TEXT_FILEPATH = "askubuntu/text_tokenized.txt"
GLOVE_FILEPATH = "glove.840B.300d.txt"
PRUNED_FILEPATH = "pruned_glove.txt"
ASKUBUNTU_VECTOR_FILEPATH = "askubuntu/vector/vectors_pruned.200.txt"

def add_words(filepath, all_words):
    with open(filepath, 'r') as f:
        for line in f:
            id, title, body = line[:-1].split('\t')
            for word in title.split():
                all_words.add(word.lower())
            for word in body.split():
                all_words.add(word.lower()) 

all_words = set()
add_words(ANDROID_TEXT_FILEPATH, all_words)
add_words(UBUNTU_TEXT_FILEPATH, all_words)
print len(all_words)

output = ""
num_words_found = 0
with open(GLOVE_FILEPATH, 'r') as f:
    for line in f:
        word = line[:-1].split()[0]
        if word in all_words:
            output += line
            num_words_found += 1
            if num_words_found % 10000 == 0:
                print "currently at " + str(num_words_found)
with open(PRUNED_FILEPATH, 'w') as f:
    f.write(output)
print "Found " + str(num_words_found) + " words in total"

# words_in_pruned = set()
# with open(PRUNED_FILEPATH, 'r') as f:
#     for line in f:
#         word = line[:-1].split()[0]
#         words_in_pruned.add(word)

# words_in_askubuntu = set()
# with open(ASKUBUNTU_VECTOR_FILEPATH, 'r') as f:
#     for line in f:
#         word = line[:-1].split()[0]
#         words_in_askubuntu.add(word)

# words_found = 0
# for word in all_words:
#     if word not in words_in_pruned and word in words_in_askubuntu:
#         print "found word " + word
#         words_found += 1
# print "found this many words: " + str(words_found)