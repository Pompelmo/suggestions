# ------------------------------------------------------------
# script to generate the json object with websites similar
# to inout website and their scores
# ------------------------------------------------------------

from pairwise_distance import *
from vectors_mean import *
from integration import Integration


class CreateJson(object):
    def __init__(self, corpus, tfidf, index, tfidf_dict, tfidf_web, db_mean_value,
                 ball_tree, id_to_web, d2v_model, db_des, w2v_model, db_key, len_dict):
        self.corpus = corpus                    # bow corpus
        self.tfidf = tfidf                      # tfidf model
        self.index = index                      # tfidf similarity matrix
        self.tfidf_dict = tfidf_dict            # dictionary for word <-> id
        self.tfidf_web = tfidf_web              # dictionary for doc n <-> website
        self.db_mean_value = db_mean_value      # mean vector <-> website database
        self.ball_tree = ball_tree              # nearest neighbors ball tree structure
        self.id_to_web = id_to_web              # dictioanry to translate nearest neighbor
        self.d2v_model = d2v_model              # description doc2vec model
        self.db_des = db_des                    # website <-> description db connection
        self.w2v_model = w2v_model              # keywords word2vec model
        self.db_key = db_key                    # website <-> keywords db connection
        self.len_dict = len_dict                # website <-> metadata (len) dict
        self.loss = 1.0
        self.key_len_in = 0.0                           # metadata about input website
        self.des_len_in = 0.0                           # they are changed when asking for input web metadata
        self.txt_len_in = 0.0                           # in self.inp_web_info with explicit = True

        # to have the integration functions
        self.integrate = Integration(self.corpus, self.tfidf, self.index, self.tfidf_web,
                                     self.db_mean_value, self.ball_tree, self.id_to_web, self.d2v_model)

    def count_text(self, url):
        """function to count text tokens present in tfidf model"""
        try:
            indx = self.tfidf_web.values().index(url)       # try to get the index of the website
        except ValueError:
            return 0

        doc_num = self.tfidf_web.keys()[indx]               # now get its id (same index)

        bow = self.corpus[doc_num]                          # transform it in bow
        tf_rap = self.tfidf[bow]                            # get its tfidf representation

        return len(tf_rap)           # tfidf representation is format (index, value!=0)

    def inp_web_info(self, url, explicit=False):
        """information on the input website"""
        # if explicit = True, keywords and description tokens are explicitly written. Use it just for input data!

        if explicit:                        # enriched metadata, used only for input website

            text_tokens = self.count_text(url)

            try:
                keywords = self.db_key[str(url)]
            except KeyError:
                keywords = []

            try:
                description = self.db_des[str(url)]
            except KeyError:
                description = []

            self.key_len_in = len(keywords)
            self.des_len_in = len(description)
            self.txt_len_in = text_tokens

            if self.key_len_in == 0 and self.des_len_in == 0 and self.txt_len_in == 0:
                return {}

            input_dict = {'metadata': {'keywords': keywords, 'description': description,
                                       'keywords_number': self.key_len_in, 'desc_tokens': self.des_len_in,
                                       'text_tokens': self.txt_len_in},
                          'link': 'http://' + url}

        else:
            key_len, des_len, text_tokens = self.get_weight(url)

            input_dict = {'metadata': {'keywords_number': key_len, 'desc_tokens': des_len,
                                       'text_tokens': text_tokens},
                          'link': 'http://' + url}

        return input_dict

    def get_weight(self, url):

        try:
            key_len = self.len_dict[url][0]
        except KeyError:
            key_len = 0

        try:
            des_len = self.len_dict[url][1]
        except KeyError:
            des_len = 0

        text_tokens = self.count_text(url)              # get count of text tokens

        return key_len, des_len, text_tokens

    def text_websites(self, weblist, sf, n, only_web=False):
        """compute the 20 websites most similar according to tfidf, and compute their value also in the other models"""

        # get 20 most similar web according to tfidf
        tfidf_score, tfidf_rank = self.integrate.ms_tfidf(weblist, n)

        text_dict = dict()              # empty dict for json obj creation

        w2v_mean, num = mean_w2v(self.db_mean_value, weblist)
        d2v_mean, num = mean_d2v(self.d2v_model, weblist)

        if not only_web:            # if we want the entire dictionary with metadata and partial score

            for i in range(0, len(tfidf_rank)):         # for every similar website

                item = tfidf_rank[i]                    # get its name
                text_dict[item] = {}

                w2v_s = w2v_distance(self.db_mean_value, w2v_mean, item, self.loss)    # distance according to w2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)      # distance according to d2v model

                metadata = self.inp_web_info(item)      # get its metadata
                text_dict[item].update(metadata)              # json obj I part: metadata

                scores = {'w2v': w2v_s,                 # json obj II part: scores according to the three models
                          'd2v': d2v_s,
                          'tfidf': tfidf_score[i]}

                text_dict[item].update({'scores': scores})

                if sf.meta_len:
                    w2v_d = metadata['metadata']['keywords_number']  # retrieve single metadata in order to use them for
                    d2v_d = metadata['metadata']['desc_tokens']         # the score function
                    tfidf_d = metadata['metadata']['text_tokens']
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i],
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)
                else:
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i])

                text_dict[item].update({'total_score': total_score})

        else:                # if we want just a dictionary item: total score
            for i in range(0, len(tfidf_rank)):
                item = tfidf_rank[i]                    # get its name

                w2v_s = w2v_distance(self.db_mean_value, w2v_mean, item, self.loss)    # distance according to w2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)      # distance according to d2v model

                text_dict[item] = {}

                if sf.meta_len:
                    w2v_d, d2v_d, tfidf_d = self.get_weight(item)
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i],
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)
                else:
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i])

                text_dict[item].update({'total_score': total_score})
                text_dict[item].update({'link': 'http://' + item})

        return text_dict

    def d2v_websites(self, weblist, sf, n, only_web=False):
        """compute the 20 websites most similar according to tfidf, and compute their value also in the other models"""
        # get 20 most similar websites according to d2v
        d2v_score, d2v_rank = self.integrate.ms_d2v(weblist, n)
        d2v_dict = dict()           # empty dict for json obj creation

        w2v_mean, num = mean_w2v(self.db_mean_value, weblist)
        tfidf_mean, num = mean_tfidf(self.tfidf_web, self.corpus, self.tfidf, weblist)

        if not only_web:

            for i in range(0, len(d2v_rank)):               # for every similar website

                item = d2v_rank[i]                  # get its name
                d2v_dict[item] = {}

                w2v_s = w2v_distance(self.db_mean_value, w2v_mean, item, self.loss)   # distance according to w2v model
                # and according to tfidf
                tfidf_s = tfidf_distance(self.corpus, self.tfidf, self.tfidf_web, tfidf_mean, item, self.loss)

                metadata = self.inp_web_info(item)  # and retrieve its metadata
                d2v_dict[item].update(metadata)  # json obj I part: metadata

                scores = {'w2v': w2v_s,                 # json obj II part: scores according to the three models
                          'd2v': d2v_score[i],
                          'tfidf': tfidf_s}

                d2v_dict[item].update({'scores': scores})

                if sf.meta_len:
                    w2v_d = metadata['metadata']['keywords_number']  # retrieve single metadata in order to use them for
                    d2v_d = metadata['metadata']['desc_tokens']         # the score function
                    tfidf_d = metadata['metadata']['text_tokens']

                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_score[i], tfidf_score=tfidf_s,
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)

                else:
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_score[i], tfidf_score=tfidf_s)

                d2v_dict[item].update({'total_score': total_score})

        else:
            for i in range(0, len(d2v_rank)):
                item = d2v_rank[i]
                d2v_dict[item] = {}

                w2v_s = w2v_distance(self.db_mean_value, w2v_mean, item, self.loss)   # distance according to w2v model
                # and according to tfidf
                tfidf_s = tfidf_distance(self.corpus, self.tfidf, self.tfidf_web, tfidf_mean, item, self.loss)

                # compute the total score
                if sf.meta_len:
                    w2v_d, d2v_d, tfidf_d = self.get_weight(item)
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_score[i], tfidf_score=tfidf_s,
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)
                else:
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_score[i], tfidf_score=tfidf_s)

                d2v_dict[item].update({'total_score': total_score})
                d2v_dict[item].update({'link': 'http://' + item})

        return d2v_dict

    def w2v_websites(self, weblist, sf, n, only_web=False):
        """compute the 20 websites most similar according to tfidf, and compute their value also in the other models"""
        # 20 most similar according to w2v
        w2v_score, w2v_rank = self.integrate.ms_w2v_key(weblist, n)
        w2v_dict = dict()             # empty dict for json obj creation

        # weblist is the input list of websites
        d2v_mean, num = mean_d2v(self.d2v_model, weblist)
        tfidf_mean, num = mean_tfidf(self.tfidf_web, self.corpus, self.tfidf, weblist)

        if not only_web:
            for i in range(0, len(w2v_rank)):               # for every similar website

                item = w2v_rank[i]                          # get its name
                w2v_dict[item] = {}

                # compute the distance according to d2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)
                # and according to tfidf
                tfidf_s = tfidf_distance(self.corpus, self.tfidf, self.tfidf_web, tfidf_mean, item, self.loss)

                metadata = self.inp_web_info(item)
                w2v_dict[item].update(metadata)                   # json obj I part: metadata

                scores = {'w2v': w2v_score[i],              # json obj II part: scores according to the three models
                          'd2v': d2v_s,
                          'tfidf': tfidf_s}

                w2v_dict[item].update({'scores': scores})

                if sf.meta_len:
                    w2v_d = metadata['metadata']['keywords_number']  # retrieve single metadata in order to use them for
                    d2v_d = metadata['metadata']['desc_tokens']         # the score function
                    tfidf_d = metadata['metadata']['text_tokens']
                    total_score = sf.score_func(w2v_score=w2v_score[i], d2v_score=d2v_s, tfidf_score=tfidf_s,
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)

                else:
                    total_score = sf.score_func(w2v_score=w2v_score[i], d2v_score=d2v_s, tfidf_score=tfidf_s)

                w2v_dict[item].update({'total_score': total_score})

        else:
            for i in range(0, len(w2v_rank)):
                item = w2v_rank[i]
                w2v_dict[item] = {}

                # compute the distance according to d2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)

                # and according to tfidf
                tfidf_s = tfidf_distance(self.corpus, self.tfidf, self.tfidf_web, tfidf_mean, item, self.loss)

                if sf.meta_len:
                    w2v_d, d2v_d, tfidf_d = self.get_weight(item)

                    # compute the total score
                    total_score = sf.score_func(w2v_score=w2v_score[i], d2v_score=d2v_s, tfidf_score=tfidf_s,
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)
                else:
                    total_score = sf.score_func(w2v_score=w2v_score[i], d2v_score=d2v_s, tfidf_score=tfidf_s)

                w2v_dict[item].update({'total_score': total_score})
                w2v_dict[item].update({'link': 'http://' + item})

        return w2v_dict

    def get_json(self, weblist, sf, n, only_web=False):                 # weblist must be a list
        """generate the json object with the wanted information"""

        # putting inp_data as first operation because it changes some class parameters then used in others

        d2v_web = self.d2v_websites(weblist, sf, n, only_web)       # construct dictionary doc2vec similar websites
        txt_web = self.text_websites(weblist, sf, n, only_web)      # construct dictionary with tfidf similar websites
        w2v_web = self.w2v_websites(weblist, sf, n, only_web)       # construct dictionary with word2v similar websites

        d2v_web.update(w2v_web)                     # update first dictionary with the second, to avoid repetitions
        d2v_web.update(txt_web)                     # and update also with the third one.

        # now a json obj is created: metadata of the input website, with the output given by the three models
        if d2v_web:
            input_metadata = dict()

            for website in weblist:                                 # input_website_metadata
                inp_web = self.inp_web_info(website, explicit=True)
                if inp_web:
                    input_metadata[website] = inp_web

                else:
                    input_metadata[website] = 'website not present in the models'

            # it has be ordered according to the total score
            json_obj = {'input_website_metadata': input_metadata, 'output': d2v_web}
        else:
            json_obj = {}

        return json_obj