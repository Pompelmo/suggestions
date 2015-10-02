# -------------------------------------------------------------
# script to compute the top n similar or n nearest
# websites to a given website
# -------------------------------------------------------------

from math import sqrt


class Integration(object):
    def __init__(self, corpus, tfidf, index, tfidf_web, mean_dict, ball_tree, d2v_model):
        self.corpus = corpus                    # bow corpus
        self.tfidf = tfidf                      # tfidf model
        self.index = index                      # tfidf similarity matrix
        self.tfidf_web = tfidf_web              # dictionary for doc n <-> website
        self.mean_dict = mean_dict              # mean vector <-> website
        self.ball_tree = ball_tree              # nearest neighbors ball tree structure
        self.d2v_model = d2v_model              # description doc2vec model
        # counter for keywords/token

    def ms_tfidf(self, url_id, n):
        """compute most similar websites using tfidf text model"""
        try:
            indx = self.tfidf_web.values().index(url_id)        # try to get the index of the website
        except ValueError:
            return [], []
        # indx = doc_num actually
        doc_num = self.tfidf_web.keys()[indx]                           # now get its id (same index)

        bow = self.corpus[doc_num]                                      # transform it in bow
        tf_rep = self.tfidf[bow]                                        # get its tfidf representation

        sims = self.index[tf_rep][:n+1]                                        # query for similarity

        rank = []
        scores = []

        for ite in sims:
            url = self.tfidf_web[ite[0]]                    # find document website name
            if url != url_id:
                rank.append(url)                                # append website name
                cosine_sim = float(ite[1])                      # cosine similarity
                dist = sqrt(2.0 * (1.0 - cosine_sim)) / 2.0     # transform cosine similarity in euclidean distance
                scores.append(dist)                             # append score (the smaller the better)

        return scores, rank

    def ms_w2v_key(self, url_id, n):
        """compute most similar websites using w2v keywords model"""
        try:                                            # try to find a website in the dictionary
            value = self.mean_dict[url_id]              # that associates name with mean vector value
        except KeyError:
            return [], []

        # compute the nearest neighbors with the constructed ball_tree
        distance, index = self.ball_tree.query([value], k=n+1, return_distance=True, sort_results=True)

        # transform the result in lists
        dist = distance.tolist()[0]
        ind = index.tolist()[0]

        keys = self.mean_dict.keys()
        # have the list of websites names:
        # recall that dict.keys()[i] = key and dict.values()[i] = value are such that dict[key] = value

        rank = []
        scores = []

        for i in range(0, len(dist)):
            if keys[ind[i]] != url_id:              # do not return the same website
                url = keys[ind[i]]
                rank.append(url)                    # append website name
                scores.append(dist[i] / 2.0)          # append normalized distance

        return scores, rank

    def ms_d2v(self, url_id, n):
        """compute the most similar websites using d2v descriptions model"""
        try:
            ms = self.d2v_model.docvecs.most_similar(url_id, topn=n)        # compute most similar with d2v
        except KeyError:
            return [], []

        rank = []
        scores = []

        for item in ms:
            rank.append(item[0])                              # compute rank and scores list
            cosine_sim = item[1]
            dist = sqrt(2.0 * (1.0 - cosine_sim)) / 2.0       # transform cosine similarity in euclidean distance
            scores.append(dist)

        return scores, rank
