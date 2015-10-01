# -----------------------------------------------------
# class for tokenize and stem words retrieved from
# website index, and used for the w2v models
# -----------------------------------------------------

import nltk
import re


class TokenStem(object):
    def __init__(self, ):
        self.stop_words = self.get_stop_words("source/stopword.txt")
        self.stemmer = nltk.stem.snowball.ItalianStemmer()

    def get_stop_words(self, stop_words_path):
        """create a set with italian and english stop words"""
        stop_words = []
        with open(stop_words_path, "r") as asd:
            for line in asd:
                stop_words.append(line[:len(line)-1])

        return set(stop_words)

    def tokenize_description(self, sentence):
        """tokenize a sentence, eliminating stopwords and then stemming words. all the so found
        words are then returned in a list"""
        lower_string = sentence.lower()  # make it all lowercase
        token_list = []

        # find all the words, eliminating all punctuation marks
        tok_list = re.findall(r'[\w]+', lower_string)

        for word in tok_list:
            if word not in self.stop_words:
                # if they are not in the stop words set, stem them
                token_list.append(self.stemmer.stem(word))

        return token_list  # it returns a list of tokenized and stemmed word (keeping the order)

    def tokenize_keywords(self, sentence):
        """sentence is a list of keywords, numbers are eliminated and words are stemmed"""
        stemmed = []
        for w in sentence:
            w = w.lower()
            n = re.sub(r'[0-9]+', "", w)

            # since keywords may be composed by one or more words, tokenize every one of them
            if n != "":
                s = ""
                for word in n.split(" "):

                    try:
                        word = unicode(word, "utf-8")
                    except TypeError:
                        word = word

                    s += self.stemmer.stem(word) + " "
                stemmed.append(s[:len(s)-1])

        return stemmed  # it returns a list of tokenized and stemmed word (keeping the original order, not needed here)


if __name__ == '__main__':
    print "class used for tokenize and stem some field"
