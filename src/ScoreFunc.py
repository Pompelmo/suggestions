# ---------------------------------------------------------------------------------------------------------
# ScoreFunc is a class used to set the parameters in order to compute the total_score for websites
# ---------------------------------------------------------------------------------------------------------


class ScoreFunc(object):
    """This class is used to set the parameters for the total store calculation and the computation of the total score
    itself. Total score is computed on some websites suggested as similar to a given set of websites by some models.
    The total score is a combination of parameters and scores of the websites given by the models"""
    def __init__(self):
        # are metadata needed to compute total_score?
        self.meta_len = True
        # metadata
        self.key_len_in = 0.0
        self.des_len_in = 0.0
        self.txt_len_in = 0.0

        # loss score for the NOT FOUND elements
        self.loss = 1.0

        # weight used
        self.w2v_weight = 1.0
        self.d2v_weight = 1.0
        self.tfidf_weight = 1.0

        # exponent for input website metadata
        self.mu_in_w = 0.0
        self.mu_in_d = 0.0
        self.mu_in_t = 0.0

        # exponent for output website metadata
        self.mu_out_w = 0.0
        self.mu_out_d = 0.0
        self.mu_out_t = 0.0

        # it is used just in weight_dist.
        self.dist = False

    def parameters_choice(self, method):
        # linear is the mean value of the scores of the three models
        if method == "linear":
            self.meta_len = False
            self.loss = 1.0
            self.w2v_weight = 1/3.0
            self.d2v_weight = 1/3.0
            self.tfidf_weight = 1/3.0
            self.mu_in_w = 0.0
            self.mu_in_d = 0.0
            self.mu_in_t = 0.0
            self.mu_out_w = 0.0
            self.mu_out_d = 0.0
            self.mu_out_t = 0.0
            self.dist = False

        # simply_weighted is the mean value of the scores of the three models, weighted for their metadata
        elif method == "simply_weighted":
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
            self.dist = False

        # almost equal to simply_weighted, but the metadata are weighted in a different way
        elif method == "weight_dist":
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
            self.dist = True

        # consider just the score coming from w2v
        elif method == "w2v":
            self.meta_len = False
            self.loss = 1.0
            self.w2v_weight = 1.0
            self.d2v_weight = 0.0
            self.tfidf_weight = 0.0
            self.mu_in_w = 0.0
            self.mu_in_d = 0.0
            self.mu_in_t = 0.0
            self.mu_out_w = 0.0
            self.mu_out_d = 0.0
            self.mu_out_t = 0.0
            self.dist = False

        # consider just the score coming from d2v
        elif method == "d2v":
            self.meta_len = False
            self.loss = 1.0
            self.w2v_weight = 0.0
            self.d2v_weight = 1.0
            self.tfidf_weight = 0.0
            self.mu_in_w = 0.0
            self.mu_in_d = 0.0
            self.mu_in_t = 0.0
            self.mu_out_w = 0.0
            self.mu_out_d = 0.0
            self.mu_out_t = 0.0
            self.dist = False

        # consider just the score coming from tf-idf
        elif method == "tfidf":
            self.meta_len = False
            self.loss = 1.0
            self.w2v_weight = 0.0
            self.d2v_weight = 0.0
            self.tfidf_weight = 1.0
            self.mu_in_w = 0.0
            self.mu_in_d = 0.0
            self.mu_in_t = 0.0
            self.mu_out_w = 0.0
            self.mu_out_d = 0.0
            self.mu_out_t = 0.0
            self.dist = False

        else:
            raise KeyError

        return None

    def score_func(self, w2v_score, d2v_score, tfidf_score, key_len_out=0, des_len_out=0, txt_len_out=0):
        """This function is used to normalize the metadata. The numbers chosen from the normalization comes
        from a sample analysis on the distribution of metadata"""

        if self.meta_len:  # if we need metadata for computing the total_score, normalized them (both input and output)

            # normalization of the input
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

            # normalization of the output
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

            # computing the partial sum with exponentials (mu)
            if self.dist:       # for dist_weight
                w2v_part = self.w2v_weight * w2v_score * (1.0 - key_len_in) ** self.mu_in_w \
                           * (1.0 - key_len_out) ** self.mu_out_w
                d2v_part = self.d2v_weight * d2v_score * (1.0 - des_len_in) ** self.mu_in_d \
                           * (1.0 - des_len_out) ** self.mu_out_d
                tfidf_part = self.tfidf_weight * tfidf_score * (1.0 - text_len_in) ** self.mu_in_t \
                             * (1.0 - text_len_out) ** self.mu_out_t
            else:
                w2v_part = self.w2v_weight * w2v_score * key_len_in ** self.mu_in_w \
                           * key_len_out ** self.mu_out_w
                d2v_part = self.d2v_weight * d2v_score * des_len_in ** self.mu_in_d \
                           * des_len_out ** self.mu_out_d
                tfidf_part = self.tfidf_weight * tfidf_score * text_len_in ** self.mu_in_t \
                             * text_len_out ** self.mu_out_t

            return w2v_part + d2v_part + tfidf_part

        else:       # compute the score without metadata
            return self.w2v_weight * w2v_score + self.d2v_weight * d2v_score + self.tfidf_weight * tfidf_score
