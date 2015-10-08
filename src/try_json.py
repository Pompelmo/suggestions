from loading import loading
from gen_json import CreateJson
from ScoreFunc import ScoreFunc
from datetime import datetime
import json

# load the models
corpora, tfidf, index, tfidf_dict, tfidf_web, \
    mean_dict, ball_tree, d2v_model, des_dict, w2v_model, key_dict = loading()

# class for rank, len, score computation
cs = CreateJson(corpora, tfidf, index, tfidf_dict, tfidf_web,
                mean_dict, ball_tree, d2v_model, des_dict, w2v_model, key_dict)


def main(weblist):
    """run the main for a not ordered example :)"""

    a = datetime.now()
    pretty = json.dumps(cs.get_json(weblist, ScoreFunc(), n=10, only_web=False),
                        indent=4, separators=(':', ','))
    print pretty
    print "time taken iteration ", datetime.now() - a

if __name__ == '__main__':
    while True:
        print "new query, 'exit' to exit, anything else to continue"
        boh = raw_input("--> ")
        if boh != 'exit':
            web_list = []
            while True:
                print "insert a website or 'stop' when the list is finished"
                word = raw_input("--> ")
                if word != "stop":
                    web_list.append(word)
                else:
                    break
            main(web_list)
        else:
            break
