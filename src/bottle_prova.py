from gen_json import CreateJson
from loading import loading
from bottle import Bottle, run, request, error, static_file, response
from ScoreFunc import ScoreFunc
from company_sim import *
import json
import pickle
from datetime import datetime

# load the models needed
corpus, tfidf, index, tfidf_dict, tfidf_web, \
    mean_dict, ball_tree, d2v_model, des_dict, w2v_model, key_dict = loading()

print datetime.now(), "loading company->websites dictionary"
inp_file = open('source/id_key.pkl', 'r')
id_key = pickle.load(inp_file)
inp_file.close()

print datetime.now(), "loading websites->company dictionary"
inp_file = open('source/web_key.pkl', 'r')
web_key = pickle.load(inp_file)
inp_file.close()

print datetime.now(), "initialize the classes"

# class for rank, len, score computation
c_json = CreateJson(corpus, tfidf, index, tfidf_dict, tfidf_web,
                    mean_dict, ball_tree, d2v_model, des_dict, w2v_model, key_dict)

sf = ScoreFunc()  # class for total_score computation
app = Bottle()  # bottle application

print datetime.now(), "preparations finished"


@app.route('/suggest')
def suggestions():
    response.content_type = 'application/json'
    parameters = request.query.decode()     # retrieve query parameters

    # check if there are no mispellings or different parameters
    accepted_input = ['website', 'model', 'num_max', 'only_website', 'company']
    for key in parameters.keys():
        if key not in accepted_input:
            response.body = json.dumps({"error": "wrong parameter(s) in input",
                                        "expected": ".../suggest?website=a_website[&model=(default='linear)'"
                                                    "&num_max=(default=60)&only_website=(default=False)]"
                                                    "\n"
                                                    "or .../suggest?company=atoka_company_id[&model=(default='linear)'"
                                                    "&num_max=(default=60)&only_website=(default=False)]"})
            return response

    # check if website value is provided or return an error
    if 'company' in parameters.keys():
        companies = parameters.getall('company')        # get all the &company= in the query

    elif 'website' in parameters.keys():        # if both company and website is present in the query, web is ignored
        weblist = parameters.getall('website')         # get all the &website= in the query

    else:
        response.body = json.dumps({"error": "parameter 'website' or 'company' is missing"})
        return response

    # check if model parameter is provided, otherwise set default
    if 'model' in parameters.keys():
        model = parameters['model']             # which model?
    else:
        model = 'linear'

    # check if num_max parameter is provided, otherwise set default
    if 'num_max' in parameters.keys():
        try:
            num = int(parameters['num_max']) / 3
        except ValueError:
            response.body = json.dumps({"error": "expected an integer in the 'num_max' field"})
            return response
    else:
        num = 20
        parameters['num_max'] = 60

    try:
        sf.parameters_choice(model)
    except KeyError:
        response.body = json.dumps({"error": "wrong model in input, try: 'linear', 'simple weighted', "
                                             "'w2v', 'd2v' or 'tfidf"})
        return response

    if num < 30:
        num_min = 30
    else:
        num_min = num

    if 'only_website' in parameters.keys():
        try:
            only_website = boolean(parameters['only_website'])
        except KeyError:                                            # or wrong input...
            response.body = json.dumps({"error": "wrong input on only_website: try 't', 'T', 'true', "
                                        "'True' or '1' to eliminate metadata"})
            return response

    else:
        only_website = False                    # if nothing is provided, we want metadata!!!

    if 'company' in parameters.keys():

        print 'company'
        a = datetime.now()
        json_obj = company_similarity(c_json, sf, num_min, only_website,
                                      companies, id_key, web_key, parameters['num_max'])

        print 'end company'
        print datetime.now() - a

    if 'website' in parameters.keys():

        print 'website'
        a = datetime.now()

        dictionary = c_json.get_json(weblist, sf, num_min, only_website)         # get dictionary from c_json

        if dictionary:
            n = min(int(parameters['num_max']), len(dictionary[u'output']))

            json_obj = website_similarity(dictionary, n)

        else:
            json_obj = {'error': 'websites not present in the models'}

        print 'website'
        print datetime.now() - a

    response.body = json.dumps(json_obj)

    return response


def boolean(string):
    if string in ['true', 'True', 't', 'T', '1']:
        return True
    elif string in ['false', 'False', 'f', 'F', '0']:
        return False
    else:
        raise KeyError


@app.route('/<filename:path>')
def server_static(filename):
    return static_file(filename, root='/home/user/code/static')


@app.route('/')
def index():
    return static_file('index.html', root='/home/user/code/static')


@app.route('/doc')
def index():
    return static_file('suggest_doc.html', root='/home/user/code/static')


@error(500)
def error500(error):            # try to explain how to make it works
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
