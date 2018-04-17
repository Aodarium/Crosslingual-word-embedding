from fasttext import FastVector
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.models.wrappers import FastText

import numpy as np


def normalized(a, axis=-1, order=2):
    """Utility function to normalize the rows of a numpy array."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2==0] = 1
    return a / np.expand_dims(l2, axis)

def make_training_matrices(source_dictionary, target_dictionary, bilingual_dictionary):
    """
    Source and target dictionaries are the FastVector objects of
    source/target languages. bilingual_dictionary is a list of 
    translation pair tuples [(source_word, target_word), ...].
    """
    source_matrix = []
    target_matrix = []
    long = len(bilingual_dictionary)
    i = 0
    for (source, target) in bilingual_dictionary:
        i = i + 1	
        if i % long == 0:
            print (i/long)		
        if source in source_dictionary and target in target_dictionary:
            source_matrix.append(source_dictionary[source])
            target_matrix.append(target_dictionary[target])

    # return training matrices
    return np.array(source_matrix), np.array(target_matrix)

def learn_transformation(source_matrix, target_matrix, normalize_vectors=True):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalized(source_matrix)
        target_matrix = normalized(target_matrix)

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V)
	
	
al_dictionary = FastVector(vector_file='cc.de.300.vec')
ru_dictionary = FastVector(vector_file='wiki.ru.vec')

print('[Ok] Loading done')
ru_words = set(ru_dictionary.word2id.keys())
al_words = set(al_dictionary.word2id.keys())
overlap = list(ru_words & al_words)
bilingual_dictionary = [(entry, entry) for entry in overlap]
print('[Ok] Dico ready')

print('[-] Matrix in process')

# form the training matrices
source_matrix, target_matrix = make_training_matrices(
    al_dictionary, ru_dictionary, bilingual_dictionary)
print('[Ok] Matrix ready')

# learn and apply the transformation
print('[-] Transformation in process')
transform = learn_transformation(source_matrix, target_matrix)
print('[Ok] Transformation completed')

print('[-] Applying transformation')
al_dictionary.apply_transform(transform)
print('[Ok] Done')

print('[-] Export')
al_dictionary.export('alltransform.bin')
print('[Ok] Export done')
