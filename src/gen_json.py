# ------------------------------------------------------------
# script to generate the json object with websites similar
# to inout website and their scores
# ------------------------------------------------------------

from how_many import Counter
from pairwise_distance import *
from vectors_mean import *
from integration import Integration


class CreateJson(object):
    def __init__(self, corpus, tfidf, index, tfidf_dict, tfidf_web,
                 mean_dict, ball_tree, d2v_model, des_dict, w2v_model, key_dict):
        self.corpus = corpus                    # bow corpus
        self.tfidf = tfidf                      # tfidf model
        self.index = index                      # tfidf similarity matrix
        self.tfidf_dict = tfidf_dict            # dictionary for word <-> id
        self.tfidf_web = tfidf_web              # dictionary for doc n <-> website
        self.mean_dict = mean_dict              # mean vector <-> website
        self.ball_tree = ball_tree              # nearest neighbors ball tree structure
        self.d2v_model = d2v_model              # description doc2vec model
        self.des_dict = des_dict
        self.w2v_model = w2v_model              # keywords word2vec model
        self.key_dict = key_dict
        self.loss = 1.0
        self.key_len_in = 0.0                           # metadata about input website
        self.des_len_in = 0.0                           # they are changed when asking for input web metadata
        self.txt_len_in = 0.0                           # in self.inp_web_info with explicit = True
        self.counter = Counter(self.corpus, self.des_dict, self.key_dict,
                               self.tfidf_dict, self.tfidf, self.tfidf_web)   # counter for keywords/token
        # to have the integration functions
        self.integrate = Integration(self.corpus, self.tfidf, self.index, self.tfidf_web,
                                     self.mean_dict, self.ball_tree, self.d2v_model)

    def inp_web_info(self, url, explicit=False):
        """information on the input website"""

        keywords = self.counter.count_keywords(url)             # get keywords
        description = self.counter.count_description(url)       # get description tokens
        text_tokens = self.counter.count_text(url)              # get count of text tokens

        # if explicit = True, keywords and description tokens are explicitly written. Use it just for input data!

        if explicit:                        # enriched metadata, used only for input website

            self.key_len_in = len(keywords)
            self.des_len_in = len(description)
            self.txt_len_in = text_tokens

            if self.key_len_in == 0 and self.des_len_in == 0 and self.txt_len_in == 0:
                return {}

            input_dict = {'metadata': {'keywords': keywords, 'description': description,
                                       'keywords_number': self.key_len_in, 'desc_tokens': self.des_len_in,
                                       'text_tokens': self.txt_len_in}}

        else:
            input_dict = {'metadata': {'keywords_number': len(keywords), 'desc_tokens': len(description),
                                       'text_tokens': text_tokens}}

        return input_dict

    def get_weight(self, url):
        keywords = self.counter.count_keywords(url)            # get keywords
        description = self.counter.count_description(url)       # get description tokens
        text_tokens = self.counter.count_text(url)              # get count of text tokens

        return len(keywords), len(description), text_tokens

    def text_websites(self, weblist, sf, n, only_web=False):
        """compute the 20 websites most similar according to tfidf, and compute their value also in the other models"""

        # get 20 most similar web according to tfidf
        tfidf_score, tfidf_rank = self.integrate.ms_tfidf(weblist, n)

        text_dict = dict()              # empty dict for json obj creation

        w2v_mean, num = mean_w2v(self.mean_dict, weblist)
        d2v_mean, num = mean_d2v(self.d2v_model, weblist)

        if not only_web:            # if we want the entire dictionary with metadata and partial score

            for i in range(0, len(tfidf_rank)):         # for every similar website

                item = tfidf_rank[i]                    # get its name
                text_dict[item] = {}

                w2v_s = w2v_distance(self.mean_dict, w2v_mean, item, self.loss)      # distance according to w2v model
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

                w2v_s = w2v_distance(self.mean_dict, w2v_mean, item, self.loss)      # distance according to w2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)      # distance according to d2v model

                text_dict[item] = {}

                if sf.meta_len:
                    w2v_d, d2v_d, tfidf_d = self.get_weight(item)
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i],
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)
                else:
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i])

                text_dict[item].update({'total_score': total_score})

        return text_dict

    def d2v_websites(self, weblist, sf, n, only_web=False):
        """compute the 20 websites most similar according to tfidf, and compute their value also in the other models"""
        # get 20 most similar websites according to d2v
        d2v_score, d2v_rank = self.integrate.ms_d2v(weblist, n)
        d2v_dict = dict()           # empty dict for json obj creation

        w2v_mean, num = mean_w2v(self.mean_dict, weblist)
        tfidf_mean, num = mean_tfidf(self.tfidf_web, self.corpus, self.tfidf, weblist)

        if not only_web:

            for i in range(0, len(d2v_rank)):               # for every similar website

                item = d2v_rank[i]                  # get its name
                d2v_dict[item] = {}

                w2v_s = w2v_distance(self.mean_dict, w2v_mean, item, self.loss)   # distance according to w2v model
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

                w2v_s = w2v_distance(self.mean_dict, w2v_mean, item, self.loss)   # distance according to w2v model
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

        return d2v_dict

    def w2v_websites(self, weblist, sf, n, only_web=False):
        """compute the 20 websites most similar according to tfidf, and compute their value also in the other models"""
        # 20 most similar according to w2v
        w2v_score, w2v_rank = self.integrate.ms_w2v_key(weblist, n)
        w2v_dict = dict()             # empty dict for json obj creation

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

        return w2v_dict

    def get_json(self, weblist, sf, n, only_web=False):
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
            for website in weblist:
                inp_web = self.inp_web_info(website, explicit=True)
                if inp_web:
                    input_metadata[website] = inp_web
                else:
                    input_metadata[website] = {'website': 'not present in the models'}

            # it has be ordered according to the total score
            json_obj = {'input_website_metadata': input_metadata, 'output': d2v_web}
        else:
            json_obj = {}

        return json_obj

