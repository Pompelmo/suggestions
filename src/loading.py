# -------------------------------------------------------------
# script to load model needed to show ranks
# -------------------------------------------------------------

import pickle
import gensim
from datetime import datetime


def loading():
    print datetime.now(), "loading corpus"
    corpus = gensim.corpora.MmCorpus('source/web_text_mm_last.mm')

    print datetime.now(), "loading model for tf-idf"
    tfidf = gensim.models.TfidfModel.load('source/web_text_tifidf_last.tfidf_model')  # tfidf model
    index = gensim.similarities.Similarity.load('source/tfidfSim_last_100.index')         # tfidf similarity matrix
    tfidf_dict = gensim.corpora.Dictionary.load('source/web_text_dict_last.dict')     # dictionary for word <-> id

    print datetime.now(), "loading dictionary for tf-idf"
    tfidf_dict_file = open('source/web_text_doc_last.pkl', 'r')
    tfidf_web = pickle.load(tfidf_dict_file)                                          # dictionary for doc n <-> website
    tfidf_dict_file.close()

    print datetime.now(), "loading dictionary for w2v keywords n_similarity"
    mean_dict_file = open('source/mean_dict_key_scan.pkl', 'r')
    mean_dict = pickle.load(mean_dict_file)
    mean_dict_file.close()

    print datetime.now(), "loading ball tree for nearest neighbor"
    input_file = open("source/ball_tree", "r")
    ball_tree = pickle.load(input_file)
    input_file.close()

    print datetime.now(), "loading d2v description model"
    d2v_model = gensim.models.Doc2Vec.load('source/d2vdescription')

    print datetime.now(), "loading description dictionary"
    input_file = open("source/des_dict", "r")
    des_dict = pickle.load(input_file)
    input_file.close()

    print datetime.now(), "loading word2vector model"
    w2v_model = gensim.models.Word2Vec.load('source/w2vmodel_keywords_scan')

    print datetime.now(), "loading keywords dictionary"
    input_file = open("source/key_dict", "r")
    key_dict = pickle.load(input_file)
    input_file.close()

    print datetime.now(), "load finish"

    return corpus, tfidf, index, tfidf_dict, tfidf_web, mean_dict, ball_tree, d2v_model, des_dict, w2v_model, key_dict
