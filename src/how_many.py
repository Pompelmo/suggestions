# -------------------------------------------------------------
# script used to count number of keywords or tokens used
# by the model for a given website
# -------------------------------------------------------------


class Counter(object):                                           # class with all the methods for counting

    def __init__(self, corpus, des_dict, key_dict, tfidf_dict, tfidf, tfidf_web):
        self.corpus = corpus            # bow corpus
        self.des_dict = des_dict
        self.key_dict = key_dict
        self.tfidf_dict = tfidf_dict    # tfidf text model dictionary
        self.tfidf = tfidf              # tfidf text model
        self.tfidf_web = tfidf_web      # dictionary for doc n <-> website

    def count_keywords(self, url):
        """function to count keywords present in the model keywords word2vec"""
        try:
            key = self.key_dict[url]
        except KeyError:
            key = []

        return key                        # return how many keywords are present in the model

    def count_description(self, url):
        """function to count description tokens present in the model description doc2vec"""
        try:
            des = self.des_dict[url]
        except KeyError:
            des = []

        return des                        # return how many text are present in the model

    def count_text(self, url):
        """function to count text tokens present in tfidf model"""
        try:
            indx = self.tfidf_web.values().index(url)       # try to get the index of the website
        except ValueError:
            return None

        doc_num = self.tfidf_web.keys()[indx]               # now get its id (same index)

        bow = self.corpus[doc_num]                          # transform it in bow
        tf_rap = self.tfidf[bow]                            # get its tfidf representation

        return len(tf_rap)           # tfidf representation is format (index, value!=0)
