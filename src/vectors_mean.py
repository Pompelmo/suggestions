from gensim import matutils
import numpy as np


def sparse_mean(sparse_vectors_list):
    dense_vectors_list = []

    for vec in sparse_vectors_list:
        dense_vectors_list.append(matutils.sparse2full(vec, length=1013243))

    mean = np.mean(dense_vectors_list, axis=0)

    return matutils.unitvec(matutils.full2sparse(mean))


def mean_tfidf(tfidf_web, corpus, tfidf, lsi, weblist):
    web_vec_rep = list()  # collect all the vectorial representation in one list

    for item in weblist:
        try:
            indx = tfidf_web.values().index(item)        # try to get the index of the website
        except ValueError:
            continue
        # indx = doc_num actually
        doc_num = tfidf_web.keys()[indx]                 # now get its id (same index)

        bow = corpus[doc_num]                            # transform it in bow
        web_vec_rep.append(lsi[tfidf[bow]])                   # get its tfidf representation

    if web_vec_rep:
        number = len(web_vec_rep)
        # transform all the vectors to dense vectors, then compute their mean with numpy,
        # transform it in unit vec and return to sparse vector representation (to be able to query for similarity)
        if len(web_vec_rep) > 1:
            mean_vector = sparse_mean(web_vec_rep)
        else:
            mean_vector = web_vec_rep[0]

        return mean_vector, number

    else:
        return [], 0


def mean_w2v(db_mean_value, weblist):
    web_vec_rep = []

    for url in weblist:
        try:                                            # try to find a website in the dictionary
            value = db_mean_value[str(url)]             # that associates name with mean vector value
        except KeyError:
            continue

        web_vec_rep.append(value)

    if web_vec_rep:
        number = len(web_vec_rep)
        mean_vec = np.sum(web_vec_rep, axis=0)         # get the mean vector of the websites vector rep
        dim = np.linalg.norm(mean_vec)
        if dim:
            mean_vec /= dim             # if the vector is different from zero, normalize it

        return mean_vec, number

    else:
        return [], 0


def mean_d2v(d2v_model, weblist):
    web_vec_rep = []

    for url in weblist:
        ms = d2v_model.docvecs[url]              # compute most similar with d2v

        if len(ms) == 100:
            web_vec_rep.append(ms)

    if web_vec_rep:
        number = len(web_vec_rep)
        mean_vec = np.mean(web_vec_rep, axis=0)
        dim = np.linalg.norm(mean_vec)
        if dim:
            mean_vec /= dim             # if the vector is different from zero, normalize it

        return mean_vec, number

    else:
        return [], 0
