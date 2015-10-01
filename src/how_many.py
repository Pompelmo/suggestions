# -------------------------------------------------------------
# script used to count number of keywords or tokens used
# by the model for a given website
# -------------------------------------------------------------


from elasticsearch import Elasticsearch
import json
from TokenStem import TokenStem

es = Elasticsearch(['http://es-idg:9200'])  # client for the elasticsearh index

index = 'idg-20150723'

ts = TokenStem()        # class for tokenizing and stemming


class Counter(object):                                           # class with all the methods for counting

    def __init__(self, corpus, w2v_model, d2v_model, tfidf_dict, tfidf, tfidf_web):
        self.corpus = corpus            # bow corpus
        self.w2v_model = w2v_model      # keywords word2vector model
        self.d2v_model = d2v_model      # description doc2vector model
        self.tfidf_dict = tfidf_dict    # tfidf text model dictionary
        self.tfidf = tfidf              # tfidf text model
        self.tfidf_web = tfidf_web      # dictionary for doc n <-> website

    def q_tokenize(self, url_id, field):
        """query + function to retrieve keywords or tokens in description"""
        url_id = [url_id]
        query = json.dumps({
            '_source': ['_id', field],
            'query': {
                'filtered': {
                    'filter': {
                        'terms': {'_id': url_id}
                    }
                }
            }
        })
        try:
            response = es.search(index=index, body=query)[u'hits'][u'hits'][0]
        except IndexError:
            return []

        if field == "keywords":
            if u'keywords' in response[u'_source'].keys():
                return ts.tokenize_keywords(response[u'_source'][u'keywords'])
            else:
                return []

        elif field == "description":
            if u'description' in response[u'_source'].keys():
                return ts.tokenize_description(response[u'_source'][u'description'])
            else:
                return []

        else:
            return []

    def count_keywords(self, url):
        """function to count keywords present in the model keywords word2vec"""
        keywords = self.q_tokenize(url, "keywords")  # retrieve keywords
        key = []
        for item in keywords:
            try:
                a = self.w2v_model[item]            # this a is useless, but PyCharm complain that self.w2v[item]
                key.append(item)                    # has no effect
            except KeyError:
                pass

        return key                        # return how many keywords are present in the model

    def count_description(self, url):
        """function to count description tokens present in the model description doc2vec"""
        description = self.q_tokenize(url, "description")
        des = []
        for item in description:
            try:
                a = self.d2v_model[item]            # this a is useless, but PyCharm complain that self.w2v[item]
                des.append(item)                    # has no effect
            except KeyError:
                pass

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

        return len([x for x in tf_rap if x != 0])           # count non zero entries
