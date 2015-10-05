from loading import loading
from gen_json import CreateJson


def main():
    """run the main for a not ordered example :)"""
    from ScoreFunc import ScoreFunc
    from datetime import datetime
    # load the models
    corpora, tfidf, index, tfidf_dict, tfidf_web, \
        mean_dict, ball_tree, d2v_model, des_dict, w2v_model, key_dict = loading()

    # class for rank, len, score computation
    cs = CreateJson(corpora, tfidf, index, tfidf_dict, tfidf_web,
                    mean_dict, ball_tree, d2v_model, des_dict, w2v_model, key_dict)
    while True:
        print "insert website or 'stop' to exit"
        website = raw_input("--> ")
        if website == "stop":
            break
        a = datetime.now()
        print cs.get_json(website, ScoreFunc(), n=10, only_web=False)
        print "time taken", datetime.now() - a

if __name__ == '__main__':
    main()
