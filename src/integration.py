# -------------------------------------------------------------
# script to compute the top n similar or n nearest
# websites to a given list of websites
# -------------------------------------------------------------

from math import sqrt
from vectors_mean import *


class Integration(object):
    def __init__(self, corpus, tfidf, lsi, lsi_index, tfidf_web, db_mean_value, ball_tree, id_to_web, d2v_model):
        self.corpus = corpus                    # bow corpus
        self.tfidf = tfidf                      # tfidf model
        self.lsi = lsi
        self.lsi_index = lsi_index                      # tfidf similarity matrix
        self.tfidf_web = tfidf_web              # dictionary for doc n <-> website
        self.db_mean_value = db_mean_value      # mean vector <-> website database
        self.ball_tree = ball_tree              # nearest neighbors ball tree structure
        self.id_to_web = id_to_web              # associate an id to a website (less memory usage)
        self.d2v_model = d2v_model              # description doc2vec model

    def ms_tfidf(self, weblist, n):
        """compute most similar websites using tfidf text model"""
        # retrieve mean value of the vector representation of the input
        mean_vector, number = mean_tfidf(self.tfidf_web, self.corpus, self.tfidf, self.lsi, weblist)

        if number == 0:     # if there is no vector representation (websites don't exist in the models...)
            return [], []

        number += n                 # retrieve also n(=website in input) to avoid repeating input in the output
        sims = self.lsi_index[mean_vector][:number]                      # query for similarity

        rank = []
        scores = []

        for ite in sims:
            url = self.tfidf_web[ite[0]]                    # find document website name
            if url not in weblist:                  # this check is needed because SimMatrix return also input doc
                rank.append(url)                                # append website name
                cosine_sim = float(ite[1])                      # cosine similarity
                dist = sqrt(2.0 * abs(1.0 - cosine_sim)) / 2.0     # transform cosine similarity in euclidean distance
                scores.append(dist)                             # append score (the smaller the better)

        return scores, rank

    def ms_w2v_key(self, weblist, n):
        """compute most similar websites using w2v keywords model"""

        mean_vec_w2v, number = mean_w2v(self.db_mean_value, weblist)        # mean vector representation of the input

        if number == 0:     # if there is no vector representation (websites don't exist in the models...)
            return [], []
        # compute the nearest neighbors with the constructed ball_tree
        number += n         # n(=length of input) more to avoid repeating input in the output
        distance, index = self.ball_tree.query([mean_vec_w2v], k=number, return_distance=True, sort_results=True)

        # transform the result in lists
        dist = distance.tolist()[0]             # list of distances of the output from the input
        ind = index.tolist()[0]                 # list of indexes (from how ball tree was created) of the output

        # have the list of websites names:
        # recall that dict.keys()[i] = key and dict.values()[i] = value are such that dict[key] = value

        rank = []
        scores = []

        for i in range(0, len(dist)):           # scan the two lists
            if self.id_to_web[ind[i]] not in weblist:              # do not return the same website
                url = self.id_to_web[ind[i]]
                rank.append(url)                    # append website name
                scores.append(dist[i] / 2.0)          # append normalized distance

        return scores, rank

    def ms_d2v(self, weblist, n):
        """compute the most similar websites using d2v descriptions model"""

        mean_vec, number = mean_d2v(self.d2v_model, weblist)    # mean vector representation of the input

        if number == 0:     # if there is no vector representation (websites don't exist in the models...)
            return [], []

        rank = []
        scores = []

        number += n         # to avoid repeating input in the output
        similar = self.d2v_model.docvecs.most_similar([mean_vec], topn=number)      # retrieve similar according to d2v

        for item in similar:
            if item[0] not in weblist:          # che ck if they are not in the input
                rank.append(item[0])                              # compute rank and scores list
                cosine_sim = item[1]
                dist = sqrt(2.0 * abs(1.0 - cosine_sim)) / 2.0       # transform cosine similarity in euclidean distance
                scores.append(dist)

        return scores, rank
