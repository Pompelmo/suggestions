class ScoreFunc(object):
    def __init__(self):
        self.meta_len = True
        self.key_len_in = 0.0                           # metadata about input website
        self.des_len_in = 0.0                           # they are changed when asking for input web metadata
        self.txt_len_in = 0.0                           # in self.inp_web_info with explicit = True

        self.loss = 1.0                             # loss score for the NOT FOUND elements
        self.w2v_weight = 1.0                       # weight used in the total score function
        self.d2v_weight = 1.0                       # ditto
        self.tfidf_weight = 1.0                     # ditto

        self.mu_in_w = 0.0                          # exponent for input website metadata
        self.mu_in_d = 0.0                          # changed in "interactive.py"
        self.mu_in_t = 0.0
        self.mu_out_w = 0.0                             # exponent for output website metadata
        self.mu_out_d = 0.0                             # changed in "interactive.py"
        self.mu_out_t = 0.0

    def parameters_choice(self, method):
        if method == "linear":
            self.meta_len = False
            self.loss = 1.0
            self.w2v_weight = 1/3.0
            self.d2v_weight = 1/3.0
            self.tfidf_weight = 1/3.0

        elif method == "simple_weighted":
            self.meta_len = True
            self.loss = 1.0
            self.w2v_weight = 1/3.0
            self.d2v_weight = 1/3.0
            self.tfidf_weight = 1/3.0
            self.mu_in_w = 1.0
            self.mu_in_d = 1.0
            self.mu_in_t = 1.0
            self.mu_out_w = 1.0
            self.mu_out_d = 1.0
            self.mu_out_t = 1.0

        elif method == "w2v":
            self.meta_len = False
            self.loss = 1.0
            self.w2v_weight = 1.0
            self.d2v_weight = 0.0
            self.tfidf_weight = 0.0

        elif method == "d2v":
            self.meta_len = False
            self.loss = 1.0
            self.w2v_weight = 0.0
            self.d2v_weight = 1.0
            self.tfidf_weight = 0.0

        elif method == "tfidf":
            self.meta_len = False
            self.loss = 1.0
            self.w2v_weight = 0.0
            self.d2v_weight = 0.0
            self.tfidf_weight = 1.0

        else:
            raise KeyError

        return None

    def score_func(self, w2v_score, d2v_score, tfidf_score, key_len_out=0, des_len_out=0, txt_len_out=0):

        if self.meta_len:

            if self.key_len_in > 0:
                key_len_in = min(self.key_len_in / 15.0, 1)
            else:
                key_len_in = self.loss

            if self.des_len_in > 0:
                des_len_in = min(self.des_len_in / 700.0, 1)
            else:
                des_len_in = self.loss

            if self.txt_len_in > 0:
                text_len_in = min(self.txt_len_in / 30000.0, 1)
            else:
                text_len_in = self.loss

            # normalization
            if key_len_out > 0:
                key_len_out = min(key_len_out / 15.0, 1)
            else:
                key_len_out = self.loss

            if des_len_out > 0:
                des_len_out = min(des_len_out / 700.0, 1)
            else:
                des_len_out = self.loss

            if txt_len_out > 0:
                text_len_out = min(txt_len_out / 30000.0, 1)
            else:
                text_len_out = self.loss

            # computing the partial sum with exponentials mu
            w2v_part = self.w2v_weight * w2v_score * key_len_in ** self.mu_in_w * key_len_out ** self.mu_out_w
            d2v_part = self.d2v_weight * d2v_score * des_len_in ** self.mu_in_d * des_len_out ** self.mu_out_d
            tfidf_part = self.tfidf_weight * tfidf_score * text_len_in ** self.mu_in_t * text_len_out ** self.mu_out_t

            return w2v_part + d2v_part + tfidf_part

        else:
            return self.w2v_weight * w2v_score + self.d2v_weight * d2v_score + self.tfidf_weight * tfidf_score
