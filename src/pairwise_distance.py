# ------------------------------------------------------------
# script to compute the distance between two websites
# using the different models
# ------------------------------------------------------------

import numpy as np
from gensim import matutils
from math import sqrt


def tfidf_distance(corpora, tfidf, tfidf_web, mean_vec, web_2, loss_weight):
    """compute the distance (as a function of cosine similarity) between two websites using tfidf model"""
    try:
        indx_2 = tfidf_web.values().index(web_2)
    except ValueError:
        return loss_weight

    doc_num_2 = tfidf_web.keys()[indx_2]

    bow_2 = corpora[doc_num_2]

    tf_rap_2 = matutils.unitvec(tfidf[bow_2])       # get its tfidf representation

    cosine_sim = min(matutils.cossim(mean_vec, tf_rap_2), 1.0)

    return sqrt(2.0 * (1.0 - cosine_sim)) / 2.0               # return the distance of the two vectors


def w2v_distance(mean_dict, mean_vec, web_2, loss_weight):
    """compute the distance (as a function of cosine similarity) between two websites using w2v model"""
    try:
        vector_2 = np.array(mean_dict[web_2])        # already unit vectors by construction
    except KeyError:
        return loss_weight                                  # return max distance if not found

    return float(np.linalg.norm(mean_vec - vector_2)) / 2.0      # return the distance of the two vectors


def d2v_distance(d2v_model, mean_vec, web_2, loss_weight):
    """compute the distance(as a function of cosine similarity) between two websites using d2v model"""

    vector_2 = np.array(d2v_model.docvecs[web_2])

    if len(vector_2) == 100:

        vec_unit_2 = vector_2 / np.linalg.norm(vector_2)

        dist = np.linalg.norm(mean_vec - vec_unit_2)

        return dist / 2.0

    else:
        return loss_weight
