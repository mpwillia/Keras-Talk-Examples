
import string
from collections import defaultdict, namedtuple
import itertools
import math

from .line_util import clean_line

from keras.preprocessing.text import text_to_word_sequence

Dataset = namedtuple("Dataset", ['inputs', 'labels'])
WordStats = namedtuple("VocabInfo", ['count', 'max_freq', 'idf'])

def apply_word_count_threshold(dataset, min_words = 2, max_words = None):
    trimmed_dataset = Dataset([], [])
    for line, label in zip(*dataset):
        seq = text_to_word_sequence(line)
        if min_words is not None and len(seq) < min_words:
            continue
        elif max_words is not None and len(seq) > max_words:
            continue
        else:
            trimmed_dataset.inputs.append(line)
            trimmed_dataset.labels.append(label)
    
    return trimmed_dataset

def compute_vocab_with_stats(*inputs):
    vocab_doc_counts = defaultdict(list)
    total_docs = 0
    for line in itertools.chain.from_iterable(inputs):
        total_docs += 1
        seq = text_to_word_sequence(line)
        doc_stats = defaultdict(lambda:0)
        for word in seq:
            doc_stats[word] += 1
        
        for word, count in doc_stats.items():
            vocab_doc_counts[word].append(count)
    
    vocab_stats = dict() 
    for word, doc_counts in vocab_doc_counts.items():
        count = sum(doc_counts)
        max_freq = max(doc_counts)
        idf = math.log(total_docs / count)
        
        vocab_stats[word] = WordStats(count, max_freq, idf)

    return vocab_stats


def trim_vocab_by_freq(vocab, min_freq = 3, max_freq = None):
 
    def within_thresh(value):
        if min_freq is not None and value < min_freq:
            return False
        elif max_freq is not None and value > max_freq:
            return False
        else:
            return True   

    return {k:v for k,v in vocab.items() if within_thresh(v.count)}


def trim_dataset_by_tfidf(dataset, vocab, 
                          min_thresh = 0.5,
                          max_thresh = None, 
                          replacement = ""):
    trimmed_dataset = Dataset([], [])

    def within_thresh(value):
        if min_thresh is not None and value < min_thresh:
            return False
        elif max_thresh is not None and value > max_thresh:
            return False
        else:
            return True

    for line, label in zip(*dataset):
        seq = text_to_word_sequence(line)
        
        tf = defaultdict(lambda:0)

        for word in seq:
            tf[word] += 1
        
        for word, freq in tf.items():
            tf[word] = freq / vocab[word].max_freq
        
        trimmed_line = []
        for word in seq:
            tfidf = tf[word] * vocab[word].idf
            #print("  TF = {:7.3f}  IDF = {:7.3f} | {:7.3f} : '{}' in '{}'".format(tf[word], vocab[word].idf, tfidf, word, line))
            
            if within_thresh(tfidf):
                trimmed_line.append(word)
            else:
                trimmed_line.append(replacement)

        line = clean_line(" ".join(trimmed_line))

        if len(line) > 0:
            trimmed_dataset.inputs.append(line)
            trimmed_dataset.labels.append(label)
    
    return trimmed_dataset



def trim_dataset_by_vocab(dataset, vocab, replacement = ""):
    trimmed_dataset = Dataset([], [])
    
    for line, label in zip(*dataset):
        seq = text_to_word_sequence(line)
        line = clean_line(" ".join(word if word in vocab else replacement for word in seq))
        if len(line) > 0:
            trimmed_dataset.inputs.append(line)
            trimmed_dataset.labels.append(label)
    
    return trimmed_dataset


def dump_vocab(vocab):
    alpha = set(string.ascii_letters)
    alphanumeric = set(string.ascii_letters + string.digits)

    def is_non_alpha(word):
        return any(c not in alpha for c in word)
    
    def is_non_alphanumeric(word):
        return any(c not in alpha for c in word)

    def possible_contraction(word):
        return "'" in word

    fmt = "{:9d} : \"{}\"\n"
    with open("vocab.txt", 'w') as vocab_f, \
         open("non_alpha_vocab.txt", 'w') as non_alpha_f, \
         open("possible_contractions.txt", 'w') as contr_f:

        for word, stats in sorted(vocab.items(), key = lambda x : x[1], reverse = True):
            count = stats.count
            vocab_f.write(fmt.format(count, word))

            if is_non_alpha(word):
                non_alpha_f.write(fmt.format(count, word))
            
            if possible_contraction(word):
                contr_f.write(fmt.format(count, word))



