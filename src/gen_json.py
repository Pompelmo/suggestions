# ------------------------------------------------------------
# script to generate the json object with websites similar
# to inout website and their scores
# ------------------------------------------------------------

from pairwise_distance import *
from vectors_mean import *
from integration import Integration


class CreateJson(object):
    def __init__(self, corpus, tfidf, lsi, lsi_index, tfidf_dict, tfidf_web, db_mean_value,
                 ball_tree, id_to_web, d2v_model, db_des, w2v_model, db_key, len_dict):
        self.corpus = corpus                    # bow corpus
        self.tfidf = tfidf                      # tfidf model
        self.lsi = lsi                          # lsi model
        self.lsi_index = lsi_index              # index for lsi similarity computation
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
        self.integrate = Integration(self.corpus, self.tfidf, self.lsi, self.lsi_index, self.tfidf_web,
                                     self.db_mean_value, self.ball_tree, self.id_to_web, self.d2v_model)

    def count_text(self, url):
        """function to count text tokens present in tfidf model"""
        # it doesn't really make much sense now that lsi is used...but anyway, returns the number of words in tf-idf
        try:
            indx = self.tfidf_web.values().index(url)       # try to get the index of the website
        except ValueError:
            return 0

        # the following procedure to get tf-idf representation is described in gensim tutorial
        doc_num = self.tfidf_web.keys()[indx]               # now get its id (same index)

        bow = self.corpus[doc_num]                          # transform it in bag of words
        tf_rap = self.tfidf[bow]                            # get its tfidf representation

        # since it is a sparse vector representation, just need to return the length (length = number of words present)
        return len(tf_rap)           # tfidf representation format is [(index_1, value_1),...,(index_n, value_n)]

    def get_weight(self, url):
        # count the keywords/description token/text tokens. We use len_dict[url] = (#keywords, #description tokens)
        try:
            key_len = self.len_dict[url][0]     # retrieve #keywords.
        except KeyError:
            key_len = 0

        try:
            des_len = self.len_dict[url][1]     # retrieve #description tokens
        except KeyError:
            des_len = 0

        text_tokens = self.count_text(url)      # get count of text tokens

        return key_len, des_len, text_tokens

    def inp_web_info(self, url, explicit=False):
        """information on the input website"""
        # if explicit = True, keywords and description tokens are explicitly written. Use it just for input data!

        if explicit:                        # enriched metadata, used only for input website

            try:
                keywords = self.db_key[str(url)]        # try to retrieve keywords from the keywords shelve db
            except KeyError:
                keywords = []

            try:
                description = self.db_des[str(url)]     # try to retrieve description from the description shelve db
            except KeyError:
                description = []

            self.key_len_in = len(keywords)             # count keywords
            self.des_len_in = len(description)          # count description tokens
            self.txt_len_in = self.count_text(url)      # count text tokens

            if self.key_len_in == 0 and self.des_len_in == 0 and self.txt_len_in == 0:
                # if we have neither description nor keywords nor text (no existent website for example)
                return {}

            # create the dictionary with metadata information about the website
            input_dict = {'metadata': {'keywords': keywords, 'description': description,
                                       'keywords_number': self.key_len_in, 'desc_tokens': self.des_len_in,
                                       'text_tokens': self.txt_len_in},
                          'link': 'http://' + url}      # add the url. It is assumed that http redirects to https

        else:
            key_len, des_len, text_tokens = self.get_weight(url)        # get information

            # create the dictionary with restricted metadata information about the website
            input_dict = {'metadata': {'keywords_number': key_len, 'desc_tokens': des_len,
                                       'text_tokens': text_tokens},
                          'link': 'http://' + url}      # add the url. It is assumed that http redirects to https

        return input_dict

    def text_websites(self, weblist, sf, n, only_web=False):
        """compute the n websites most similar according to tfidf, and compute their value also in the other models"""

        # get 20 most similar web according to tfidf
        tfidf_score, tfidf_rank = self.integrate.ms_tfidf(weblist, n)

        text_dict = dict()              # empty dict for json obj creation

        w2v_mean, num = mean_w2v(self.db_mean_value, weblist)  # mean vectors of the websites in input according to w2v
        d2v_mean, num = mean_d2v(self.d2v_model, weblist)      # same for d2v.
        # Can we avoid computing it in other functions as well?

        if not only_web:         # if we want the entire dictionary with metadata and partial score (only_website=false)

            for i in range(0, len(tfidf_rank)):         # for every similar website retrieved through tf-idf model

                item = tfidf_rank[i]                    # get its name (url without http://)

                w2v_s = w2v_distance(self.db_mean_value, w2v_mean, item, self.loss)    # distance according to w2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)      # distance according to d2v model

                metadata = self.inp_web_info(item)      # get its metadata
                text_dict[item] = metadata              # json obj I part: metadata

                scores = {'w2v': w2v_s,                 # json obj II part: scores according to the three models
                          'd2v': d2v_s,
                          'tfidf': tfidf_score[i]}

                text_dict[item]['scores'] = scores

                if sf.meta_len:     # if the metadata are used to compute the website total_score
                    w2v_d = metadata['metadata']['keywords_number']  # retrieve single metadata in order to use them for
                    d2v_d = metadata['metadata']['desc_tokens']         # the score function
                    tfidf_d = metadata['metadata']['text_tokens']
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i],  # compute
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)  # totalscore
                else:
                    # otherwise don't even retrieve metadata and compute total_score
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i])

                text_dict[item]['total_score'] = total_score  # add total_score to the dict of output websites info

        else:                # if we want just a dictionary item: total score + link, without visible metadata
            for i in range(0, len(tfidf_rank)):         # for every website siggested by tf-idf model
                item = tfidf_rank[i]                    # get its name

                w2v_s = w2v_distance(self.db_mean_value, w2v_mean, item, self.loss)    # distance according to w2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)      # distance according to d2v model

                text_dict[item] = {}    # tell that is a dictionary, or can't create new keys

                if sf.meta_len:             # are metadata needed to compute total score?
                    w2v_d, d2v_d, tfidf_d = self.get_weight(item)
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i],
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)
                else:
                    # if not, do not retrieve metadata information
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_s, tfidf_score=tfidf_score[i])

                text_dict[item]['total_score'] = total_score        # add total score
                text_dict[item]['link'] = 'http://' + item    # since we are not using self.inp_web_info (for meta&link)

        return text_dict

    def d2v_websites(self, weblist, sf, n, only_web=False):
        """compute the n websites most similar according to d2v, and compute their value also in the other models"""
        # get 20 most similar websites according to d2v
        d2v_score, d2v_rank = self.integrate.ms_d2v(weblist, n)     # retrieve n similar websites suggested by d2v
        d2v_dict = dict()           # empty dict for json obj creation

        w2v_mean, num = mean_w2v(self.db_mean_value, weblist)       # mean value of input websites according to w2v
        tfidf_mean, num = mean_tfidf(self.tfidf_web, self.corpus, self.tfidf, self.lsi, weblist)  # mean value according to tf-idf

        if not only_web:        # if in the query only_website=false

            for i in range(0, len(d2v_rank)):               # for every similar website suggested by d2v

                item = d2v_rank[i]                  # get its name (url without http://)

                w2v_s = w2v_distance(self.db_mean_value, w2v_mean, item, self.loss)   # distance according to w2v model
                # and according to tfidf :
                tfidf_s = tfidf_distance(self.corpus, self.tfidf, self.tfidf_web, tfidf_mean, item, self.loss)

                metadata = self.inp_web_info(item)    # retrieve its metadata and link
                d2v_dict[item] = metadata             # json obj I part: metadata attached

                scores = {'w2v': w2v_s,               # json obj II part: scores according to the three models
                          'd2v': d2v_score[i],
                          'tfidf': tfidf_s}

                d2v_dict[item]['scores'] = scores       # attach scores

                if sf.meta_len:                 # if metadata are needed to compute total_score
                    w2v_d = metadata['metadata']['keywords_number']  # retrieve single metadata in order to use them for
                    d2v_d = metadata['metadata']['desc_tokens']         # the score function
                    tfidf_d = metadata['metadata']['text_tokens']

                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_score[i], tfidf_score=tfidf_s,
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)

                else:       # if they are not needed, don't even retrieve them
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_score[i], tfidf_score=tfidf_s)

                d2v_dict[item].update({'total_score': total_score})     # attach total score

        else:           # if only_website=true, so we don't show metadata in the output websites
            for i in range(0, len(d2v_rank)):       # for all the websites suggested by d2v
                item = d2v_rank[i]
                d2v_dict[item] = {}     # create the json object with web info

                w2v_s = w2v_distance(self.db_mean_value, w2v_mean, item, self.loss)   # distance according to w2v model
                # and according to tfidf
                tfidf_s = tfidf_distance(self.corpus, self.tfidf, self.tfidf_web, tfidf_mean, item, self.loss)

                # compute the total score
                if sf.meta_len:         # if metadata are needed to compute the total score
                    w2v_d, d2v_d, tfidf_d = self.get_weight(item)
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_score[i], tfidf_score=tfidf_s,
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)
                else:       # otherwise don't retrieve them
                    total_score = sf.score_func(w2v_score=w2v_s, d2v_score=d2v_score[i], tfidf_score=tfidf_s)

                d2v_dict[item]['total_score'] = total_score     # attach total score and link, since we are not
                d2v_dict[item]['link'] = 'http://' + item       # using self.inp_web_info for meta & link

        return d2v_dict

    def w2v_websites(self, weblist, sf, n, only_web=False):
        """compute the n websites most similar according to w2v, and compute their value also in the other models"""
        # 20 most similar according to w2v
        w2v_score, w2v_rank = self.integrate.ms_w2v_key(weblist, n)
        w2v_dict = dict()             # empty dict for json obj creation

        # weblist is the input list of websites
        d2v_mean, num = mean_d2v(self.d2v_model, weblist)      # input website vector rep mean value according to d2v
        tfidf_mean, num = mean_tfidf(self.tfidf_web, self.corpus, self.tfidf, self.lsi, weblist)      # same for tf-idf

        if not only_web:            # if only_website=false (so we want to see metadata of websites in the output)
            for i in range(0, len(w2v_rank)):               # for every similar website suggested by w2v

                item = w2v_rank[i]                          # get its name (url without http://)

                # compute the distance according to d2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)
                # and according to tfidf
                tfidf_s = tfidf_distance(self.corpus, self.tfidf, self.tfidf_web, tfidf_mean, item, self.loss)

                metadata = self.inp_web_info(item)
                w2v_dict[item] = metadata                   # json obj I part: metadata

                scores = {'w2v': w2v_score[i],              # json obj II part: scores according to the three models
                          'd2v': d2v_s,
                          'tfidf': tfidf_s}

                w2v_dict[item]['scores'] = scores           # append scores

                if sf.meta_len:             # if metadata are needed to compute the total score
                    w2v_d = metadata['metadata']['keywords_number']  # retrieve single metadata in order to use them for
                    d2v_d = metadata['metadata']['desc_tokens']         # the score function
                    tfidf_d = metadata['metadata']['text_tokens']
                    total_score = sf.score_func(w2v_score=w2v_score[i], d2v_score=d2v_s, tfidf_score=tfidf_s,
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)

                else:   # otherwise don't even retrieve them
                    total_score = sf.score_func(w2v_score=w2v_score[i], d2v_score=d2v_s, tfidf_score=tfidf_s)

                w2v_dict[item]['total_score'] = total_score     # append total score

        else:       # if only_website=true => we don't want to se metadata in the output

            for i in range(0, len(w2v_rank)):       # for every website suggested by w2v
                item = w2v_rank[i]                  # retrieve its name (url without http://)
                w2v_dict[item] = {}         # say it is a dictionary to add keys

                # compute the distance according to d2v model
                d2v_s = d2v_distance(self.d2v_model, d2v_mean, item, self.loss)

                # and according to tfidf
                tfidf_s = tfidf_distance(self.corpus, self.tfidf, self.tfidf_web, tfidf_mean, item, self.loss)

                if sf.meta_len:         # if metadata are needed  for computing the total score
                    w2v_d, d2v_d, tfidf_d = self.get_weight(item)

                    # compute the total score
                    total_score = sf.score_func(w2v_score=w2v_score[i], d2v_score=d2v_s, tfidf_score=tfidf_s,
                                                key_len_out=w2v_d, des_len_out=d2v_d, txt_len_out=tfidf_d)
                else:       # otherwise...u got it by now
                    total_score = sf.score_func(w2v_score=w2v_score[i], d2v_score=d2v_s, tfidf_score=tfidf_s)

                w2v_dict[item]['total_score'] = total_score  # attach total score and link. link addes since
                w2v_dict[item]['link'] = 'http://' + item   # we are not using self.inp_web_info

        return w2v_dict

    def get_json(self, weblist, sf, n, only_web=False):                 # weblist must be a list
        """generate the json object with the wanted information"""

        # putting inp_data as first operation because it changes some class parameters then used in others

        d2v_web = self.d2v_websites(weblist, sf, n, only_web)       # construct dictionary doc2vec similar websites
        txt_web = self.text_websites(weblist, sf, n, only_web)      # construct dictionary with tf-idf similar websites
        w2v_web = self.w2v_websites(weblist, sf, n, only_web)       # construct dictionary with word2v similar websites

        d2v_web.update(w2v_web)                     # update first dictionary with the second and the third one
        d2v_web.update(txt_web)                     # to avoid repetitions.

        # now a json obj is created: metadata of the input website, with the output given by the three models
        if d2v_web:     # if the dictionary is not empty
            input_metadata = dict()

            for website in weblist:                                 # input_website_metadata part
                inp_web = self.inp_web_info(website, explicit=True)
                if inp_web:     # if it exists in the models
                    input_metadata[website] = inp_web
                else:
                    input_metadata[website] = 'website not present in the models'

            # it has to be ordered according to the total score (it is done in company_sim.py)
            json_obj = {'input_website_metadata': input_metadata, 'output': d2v_web}
        else:
            json_obj = {}

        return json_obj
