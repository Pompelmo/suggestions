# -------------------------------------------------------------
# script to compute the top n similar or n nearest
# websites to a given list of websites
# -------------------------------------------------------------

from math import sqrt
from vectors_mean import *


class Integration(object):
    def __init__(self, corpus, tfidf, index, tfidf_web, db_mean_value, ball_tree, id_to_web, d2v_model):
        self.corpus = corpus                    # bow corpus
        self.tfidf = tfidf                      # tfidf model
        self.index = index                      # tfidf similarity matrix
        self.tfidf_web = tfidf_web              # dictionary for doc n <-> website
        self.db_mean_value = db_mean_value      # mean vector <-> website database
        self.ball_tree = ball_tree              # nearest neighbors ball tree structure
        self.id_to_web = id_to_web
        self.d2v_model = d2v_model              # description doc2vec model

    def ms_tfidf(self, weblist, n):
        """compute most similar websites using tfidf text model"""
        mean_vector, number = mean_tfidf(self.tfidf_web, self.corpus, self.tfidf, weblist)

        if number == 0:
            return [], []

        number += n
        sims = self.index[mean_vector][:number]                      # query for similarity

        rank = []
        scores = []

        for ite in sims:
            url = self.tfidf_web[ite[0]]                    # find document website name
            if url not in weblist:
                rank.append(url)                                # append website name
                cosine_sim = float(ite[1])                      # cosine similarity
                dist = sqrt(2.0 * (1.0 - cosine_sim)) / 2.0     # transform cosine similarity in euclidean distance
                scores.append(dist)                             # append score (the smaller the better)

        return scores, rank

    def ms_w2v_key(self, weblist, n):
        """compute most similar websites using w2v keywords model"""

        mean_vec_w2v, number = mean_w2v(self.db_mean_value, weblist)

        if number == 0:
            return [], []
        # compute the nearest neighbors with the constructed ball_tree
        number += n
        distance, index = self.ball_tree.query([mean_vec_w2v], k=number, return_distance=True, sort_results=True)

        # transform the result in lists
        dist = distance.tolist()[0]
        ind = index.tolist()[0]

        # have the list of websites names:
        # recall that dict.keys()[i] = key and dict.values()[i] = value are such that dict[key] = value

        rank = []
        scores = []

        for i in range(0, len(dist)):
            if self.id_to_web[ind[i]] not in weblist:              # do not return the same website
                url = self.id_to_web[ind[i]]
                rank.append(url)                    # append website name
                scores.append(dist[i] / 2.0)          # append normalized distance

        return scores, rank

    def ms_d2v(self, weblist, n):
        """compute the most similar websites using d2v descriptions model"""

        mean_vec, number = mean_d2v(self.d2v_model, weblist)

        if number == 0:
            return [], []

        rank = []
        scores = []

        number += n
        similar = self.d2v_model.docvecs.most_similar([mean_vec], topn=number)

        for item in similar:
            if item[0] not in weblist:
                rank.append(item[0])                              # compute rank and scores list
                cosine_sim = item[1]
                dist = sqrt(2.0 * (1.0 - cosine_sim)) / 2.0       # transform cosine similarity in euclidean distance
                scores.append(dist)

        return scores, rank
