from gen_json import CreateJson
from loading import loading
from bottle import Bottle, run, request, error, static_file, response
from ScoreFunc import ScoreFunc
from collections import OrderedDict
import json

# load the models needed
corpus, tfidf, index, tfidf_dict, tfidf_web, mean_dict, ball_tree, w2v_model, d2v_model = loading()

# class for rank, len, score computation
c_json = CreateJson(corpus, tfidf, index, tfidf_dict, tfidf_web, mean_dict, ball_tree, w2v_model, d2v_model)

sf = ScoreFunc()  # class for total_score computation
app = Bottle()  # bottle application


@app.route('/suggest')
def suggestions():
    response.content_type = 'application/json'
    parameters = request.query.decode()     # retrieve query parameters
    website = parameters['website']         # which website?
    model = parameters['model']             # which model?

    try:                                    # modify total_score class parameters
        sf.parameters_choice(model)
    except KeyError:                        # or return a json with an error
        response.body = json.dumps({"error": "wrong model in input, try: 'linear', 'simple weighted', "
                                             "'w2v', 'd2v' or 'tfidf",
                                    "expected": ".../suggest?website=your_url&model=your_model(&only_website=boolean)"})
        return response

    if 'only_website' in parameters.keys():     # do we want metadata or not?
        try:
            only_website = boolean(parameters['only_website'])
        except KeyError:                                            # or wrong input...
            response.body = json.dumps({"error": "wrong input on only_website: try 't', 'T', 'true', "
                                        "'True' or '1' to eliminate metadata",
                                        "expected":
                                            ".../suggest?website=your_url&model=your_model(&only_website=boolean)"})
            return response

    else:
        only_website = False                    # if nothing is provided, we want metadata!!!

    dictionary = c_json.get_json(website, sf, only_website)         # get dictionary from c_json
    # order everything by the total score
    try:
        dictionary_sort = OrderedDict(sorted(dictionary[u'output'].items(), key=lambda x: x[1][u'total_score']))
        # read it as a json object
        json_obj = {website: dictionary[website], 'output': [{'website': website, 'data': data} for website, data in dictionary_sort.iteritems()]}
    except KeyError:
        json_obj = dictionary

    response.body = json.dumps(json_obj)
    return response


def boolean(string):
    if string in ['true', 'True', 't', 'T', '1']:
        return True
    elif string in ['false', 'False', 'f', 'F', '0']:
        return False
    else:
        raise KeyError


@app.route('/')
def index():
    return static_file('index.html', root='/home/user/code/static')

@app.route('/<filename:path>')
def server_static(filename):
    return static_file(filename, root='/home/user/code/static')


@error(500)
def error404(error):            # try to explain how to make it works
    string = "Wrong inputs. Provide something of the kind " \
             ".../suggest?website=your_url&model=your_model(&only_website=boolean) \n" \
             "where url may be any idg-20150723 website \n \n" \
             "model can be\n" \
             "1) 'linear' \n" \
             "   1/3.0 * w2v distance + 1/3.0 * d2v distance + 1/3.0 * tfidf distance\n" \
             "2) 'simple_weighted'\n" \
             "   1/3.0 * w2v distance * # keywords/15 + 1/3.0 * d2v distance * # description tokens / 700 + 1/3.0 * " \
             "   tfidf distance * # text tokens / 30k\n" \
             "3+) 'w2v', 'd2v', 'tfidf' for query the single models \n \n" \
             "only_website False by default, change to True (true, t, T, 1) if you don't want to see metadata"

    return string


run(app, host='0.0.0.0', port=8080)
