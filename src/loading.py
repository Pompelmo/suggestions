# -------------------------------------------------------------
# script to load model needed to show ranks
# -------------------------------------------------------------

import pickle
import gensim
from datetime import datetime
import shelve


def loading():
    print datetime.now(), "loading corpus"
    corpus = gensim.corpora.MmCorpus('source/web_text_mm_last.mm')     # corpus for tfidf MemoryUsage(MU): 13Mib

    print datetime.now(), "loading model for tf-idf"
    tfidf = gensim.models.TfidfModel.load('source/web_text_tifidf_last.tfidf_model')  # tfidf model MU: 95Mib
    index = gensim.similarities.Similarity.load('source/tfidfSim_last_100.index')     # tfidf similarity matrix MU:0
    tfidf_dict = gensim.corpora.Dictionary.load('source/web_text_dict_last.dict')  # dict for word <-> id MU:118Mib

    print datetime.now(), "loading dictionary for tf-idf"
    tfidf_dict_file = open('source/web_text_doc_last.pkl', 'r')
    tfidf_web = pickle.load(tfidf_dict_file)                      # dictionary for doc n <-> website MU:103Mib
    tfidf_dict_file.close()

    print datetime.now(), "open db connection for w2v keywords mean value n_similarity"
    # just 1 usage per request: db instead of pickle dict = gain MU:500Mib
    db_mean_value = shelve.open('source/mean_dict_key_scan.db')

    print datetime.now(), "loading ball tree for nearest neighbor"
    input_file = open("source/ball_tree", "r")                      # ball tree structure to perform nearest neighbour
    ball_tree = pickle.load(input_file)                             # query. MU:629Mib
    input_file.close()

    print datetime.now(), "loading dict to translate nearest neighbor results"
    input_file = open("source/id_to_web.pkl", "r")
    id_to_web = pickle.load(input_file)
    input_file.close()

    print datetime.now(), "loading d2v description model"
    d2v_model = gensim.models.Doc2Vec.load('source/d2vdescription')     # doc 2 vector model. MU:241Mib

    print datetime.now(), "open db connection for description dictionary"
    # just 1 usage per request: db instead of pickle dict = gain MU:3759Mib
    db_des = shelve.open('source/des_dict_db.db')

    print datetime.now(), "loading word2vector model"
    w2v_model = gensim.models.Word2Vec.load('source/w2vmodel_keywords_scan')    # word 2 vec model, MU:678Mib

    print datetime.now(), "open db connection for keywords dictionary"
    # just 1 usage per request: db instead of pickle dict = gain MU:162Mib
    db_key = shelve.open("source/key_dict_db.db")

    print datetime.now(), "load dictionary for metadata"
    inp_file = open("source/key_des_len.pkl", "r")
    len_dict = pickle.load(inp_file)
    inp_file.close()

    print datetime.now(), "model loading finished"

    return corpus, tfidf, index, tfidf_dict, tfidf_web, db_mean_value, ball_tree,\
        id_to_web, d2v_model, db_des, w2v_model, db_key, len_dict
