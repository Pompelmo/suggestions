# ------------------------------------------------------------
# script to compute the distance between two websites
# using the different models
# ------------------------------------------------------------

import numpy
from gensim import matutils
from math import sqrt


def tfidf_distance(corpora, tfidf, tfidf_web, web_1, web_2, loss_weight):
    """compute the distance (as a function of cosine similarity) between two websites using tfidf model"""
    try:
        indx_1 = tfidf_web.values().index(web_1)            # try to get the index of the website
        indx_2 = tfidf_web.values().index(web_2)
    except ValueError:
        return loss_weight

    doc_num_1 = tfidf_web.keys()[indx_1]                    # now get its id (same index)
    doc_num_2 = tfidf_web.keys()[indx_2]

    bow_1 = corpora[doc_num_1]                               # transform it in bow
    bow_2 = corpora[doc_num_2]

    tf_rap_1 = matutils.unitvec(tfidf[bow_1])                                 # get its tfidf representation
    tf_rap_2 = matutils.unitvec(tfidf[bow_2])

    cosine_sim = matutils.cossim(tf_rap_1, tf_rap_2)

    return sqrt(2.0 * (1.0 - cosine_sim)) / 2.0               # return the distance of the two vectors


def w2v_distance(mean_dict, web_1, web_2, loss_weight):
    """compute the distance (as a function of cosine similarity) between two websites using w2v model"""
    try:
        vector_1 = numpy.array(mean_dict[web_1])        # already unit vectors by construction
        vector_2 = numpy.array(mean_dict[web_2])
    except KeyError:
        return loss_weight                                  # return max distance if not found

    return float(numpy.linalg.norm(vector_1 - vector_2)) / 2.0      # return the distance of the two vectors


def d2v_distance(d2v_model, web_1, web_2, loss_weight):
    """compute the distance(as a function of cosine similarity) between two websites using d2v model"""
    try:
        cosine_sim = d2v_model.docvecs.similarity(web_1, web_2)
    except ValueError:
        return loss_weight                                  # if not present, return max distance

    if not isinstance(cosine_sim, float):
        return loss_weight

    return sqrt(2.0 * (1.0 - cosine_sim)) / 2.0     # return distance of unit vectors
