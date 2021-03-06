# ----------------------------------------------------------------------
# script for running a bottle service for website/company similarity
# ----------------------------------------------------------------------

from gen_json import CreateJson
from loading import loading
from bottle import Bottle, run, request, error, static_file, response
from ScoreFunc import ScoreFunc
from company_sim import *
import json
import pickle
from datetime import datetime
import shelve

# load the models needed (from loading.py)
corpus, tfidf, lsi, lsi_index, tfidf_dict, tfidf_web, db_mean_value, \
    ball_tree, id_to_web, d2v_model, db_des, w2v_model, db_key, len_dict = loading()

# load the database (company_id -> company information) that is needed in company similarity
print datetime.now(), "loading company -> websites database"
id_key = shelve.open('source/id_key_db.db')

# load the dictionary (website -> company_id that owns it) that is needed in company similarity
print datetime.now(), "loading websites -> company dictionary"
inp_file = open('source/web_key.pkl', 'r')
web_key = pickle.load(inp_file)
inp_file.close()

print datetime.now(), "finished loading, initialize the classes"

# initialize the class for rank, len, partial score computation
c_json = CreateJson(corpus, tfidf, lsi, lsi_index, tfidf_dict, tfidf_web, db_mean_value,
                    ball_tree, id_to_web, d2v_model, db_des, w2v_model, db_key, len_dict)

sf = ScoreFunc()  # class for total_score computation
app = Bottle()  # bottle application
print datetime.now(), "preparations finished"


@app.route('/suggest')          # endpoint will be at running_port/suggest
def suggestions():
    response.content_type = 'application/json'   # explicit the response type: json.
    # retrieve query parameters (running_port/suggest?..., see http://hetzy2.spaziodati.eu:1234/doc ;) )
    parameters = request.query.decode()

    # ------------------------------------------------------
    # check for correctness of the query
    # ------------------------------------------------------
    # check if there are no misspellings or different parameters (accepted_input = list of parameters that can be used)
    accepted_input = ['website', 'model', 'num_max', 'only_website', 'company', 'location',
                      'ateco', 'ateco_dist', 'size']

    for key in parameters.keys():           # check if every query parameter is in the accepted ones
        if key not in accepted_input:       # or return an error
            response.body = json.dumps({"error": "wrong parameter(s) in input",
                                        "expected": {"websites": ".../suggest?website=a_website"
                                                                 "[&model=(default='linear)'&num_max=(default=60)&"
                                                                 "only_website=(default=False)]",
                                                     "companies": ".../suggest?company=atoka_company_id"
                                                                 "[&model=(default='linear)'&num_max=(default=60)&"
                                                                 "only_website=(default=False)&location=(false)"
                                                                  "&ateco=(false)&size=(false)]"}})
            return response

    # ----------------------------------------------------------------------
    # check if "non necessary" parameters are given, otherwise set default
    # ----------------------------------------------------------------------

    # check if model parameter is provided, otherwise set default
    if 'model' in parameters.keys():      # ...&model=linear&model=w2v is actually allowed, but with this code line
        model = parameters['model']       # only the last model parameter is kept (in this example it would be kept w2v)
    else:
        model = 'linear'                  # if model parameter is not found, set default

    # try to see if the model parameter is correct
    try:    # recall that sf is the class to set parameters for computation of the total score
        sf.parameters_choice(model)         # set some value for the computation of the total_score
    except KeyError:                     # return error if last model given is not in the supported ones
        response.body = json.dumps({"error": "wrong model in input",
                                    "epected": "'linear', 'simply_weighted', 'weight_dist', 'w2v', 'd2v' or 'tfidf"})
        return response

    # check if num_max parameter is provided and it is readable as an integer (otherwise set default or return an error)
    if 'num_max' in parameters.keys():      # allowed more than 1 num_max input, this line => kept just the last given
        try:        # check if it's an integer
            num = int(parameters['num_max']) / 3     # top "num" websites retrieved for every model ("num" for w2v, ...)
        except ValueError:          # if it is given something different than an integer
            response.body = json.dumps({"error": "wrong input for 'num_max' field",
                                        "expected": "an integer"})
            return response
    else:
        num = 30                # set default
        parameters['num_max'] = 60          # where to cut the list of suggested websites

    # set a minimum length. The list is then cut at 'num_max'. This is done in order to avoid excluding companies
    # that have a high score in a single model, but a good total_score
    if num < 30:
        num_min = 30            # 30 chosen by some mysterious experiments  ^^
    else:
        num_min = num

    # check if 'only_website' parameters is present or set it to false
    if 'only_website' in parameters.keys():     # if we want metadata to be shown or not
        # check if 'only_website' parameters have a format that can be read as true/false
        if parameters['only_website'].lower() == 'true' or parameters['only_website'].lower() == 't':
            only_website = True
        elif parameters['only_website'].lower() == 'false' or parameters['only_website'].lower() == 'f':
            only_website = False
        else:  # or wrong input...
            response.body = json.dumps({"error": "wrong only_website in input",
                                        "expected": " 'true' or 'false'"})
            return response
    # set to false if not found
    else:
        only_website = False          # if nothing is provided, we want metadata!!! plenty of metadata for everyone

    # -------------------------------------------
    # check necessary parameters or set default.
    # -------------------------------------------

    # ---------------------------------------------------------------------------------------------
    # check if company is provided.
    # if it is a company, also  check for location, ateco and size parameters

    if 'company' in parameters.keys():
        companies = parameters.getall('company')  # get all the "&company=" values in the query (getall returns a list)
        ateco_dist = 5          # set default. if it is found it is rewritten

        # ----------------------------------------
        # check for location parameter

        if 'location' in parameters.keys():         # do we want to keep companies in the same area?
            # check if the location given is in the possible location choice
            locations = ['macroregion', 'region', 'province', 'municipality']

            if parameters['location'] in locations:
                location = parameters['location']       # check for correctness of input
            else:
                response.body = json.dumps({"error": "wrong location in input",
                                            "expected": "'macroregion', 'region', 'province', 'municipality' "
                                                        "or 'false'[default]"})
                return response          # return an error if wrong input
        else:
            location = 'false'        # if location parameter is not found, set default value

        # ----------------------------------------
        # check for size parameter

        if 'size' in parameters.keys():         # do we want to keep companies with similar size?
            sizes = ['strict', 'false', 'auto']

            if parameters['size'] in sizes:         # check if the input is correct
                size = parameters['size']
            else:
                response.body = json.dumps({"error": "wrong size in input",
                                            "expected": "'strict', 'false'[default] or 'auto'"})
                return response
        else:
            size = 'false'

        # ----------------------------------------
        # if company, check if also ateco is present
        if 'ateco' in parameters.keys():
            # check if it is given one of the possible ateco
            ateco_par = ['strict', 'distance', 'auto', 'false']

            if parameters['ateco'] in ateco_par:
                ateco = parameters['ateco']

                if ateco == 'distance' and 'ateco_dist' in parameters.keys():
                    # if ateco == distance, ateco distance can be chosen freely (should be max = 8, more has no sense)
                    try:
                        ateco_dist = int(parameters['ateco_dist'])  # it needs to be an integer
                    except ValueError:
                        response.body = json.dumps({"error": "wrong ateco_dist input",
                                                    "expected": "an integer"})
                        return response

            # if it is given a parameters not in the accepted ones, raise an error
            else:
                response.body = json.dumps({"error": "wrong ateco in input",
                                            "expected": "'strict', 'distance', 'auto' or 'false'[default]"})
                return response
        else:
            ateco = 'false'         # if ateco parameter is not found, set default value

        json_obj = company_similarity(c_json, sf, num_min, only_website, companies, id_key, web_key,
                                      parameters['num_max'], location, size, ateco, ateco_dist)

    # ----------------------------------------
    # check if website is provided (in this case ateco, locations and size are completely ignored)
    elif 'website' in parameters.keys():        # if both company and website is present in the query, web is ignored
        weblist = parameters.getall('website')         # get all the &website= in the query. (getall returns a list)

        dictionary = c_json.get_json(weblist, sf, num_min, only_website)         # get output dictionary from c_json

        if dictionary:
            n = min(int(parameters['num_max']), len(dictionary['output']))      # cut the list at the lowest number

            json_obj = website_similarity(dictionary, n)                # make it look better :)

        else:           # if no website is present in the models, get_json returns {}
            json_obj = {'error': 'websites not present in the models'}

    # ----------------------------------------
    # if neither company nor website is provided, return an error
    else:
        # we do need company or website parameter!!!
        response.body = json.dumps({"error": "parameter 'website' or 'company' is missing"})

        return response

    response.body = json.dumps(json_obj)            # return suggested websites / companies

    return response


@app.route('/<filename:path>')
def server_static(filename):
    return static_file(filename, root='/home/user/code/static')


@app.route('/')
def index():
    return static_file('index.html', root='/home/user/code/static')


@app.route('/doc')      # main parameter documentation
def index():
    return static_file('main.html', root='/home/user/code/static')


@app.route('/model')    # model parameter documentation
def index():
    return static_file('model.html', root='/home/user/code/static')


@app.route('/ateco')    # ateco parameter documentation
def index():
    return static_file('ateco.html', root='/home/user/code/static')


@app.route('/location')     # location parameter documentation
def index():
    return static_file('location.html', root='/home/user/code/static')


@error(500)             # I think it doesn't work properly....500 cannot be rewritten I guess
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
