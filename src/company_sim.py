# ----------------------------------------------------------------------
# Given a company, get its similar ones. (using websites)
# ----------------------------------------------------------------------

from collections import OrderedDict


def company_similarity(create_json, sf, num_min, only_website, company_ids, id_key, web_key, num_max):

    companies_input = dict()  # here it is going to be stored the metadata
    weblist = []               # here are going to be stored all the websites of the companies asked
    i = 0

    for company_id in company_ids:                  # iterate through all the companies in the query

        if company_id in id_key.keys():             # and try to find their websites

            websites = id_key[company_id]['websites']
            weblist += websites
            # companies_input[company_id] = {}
            # for website in websites:
            #     companies_input[company_id][website] = 0

        else:
            companies_input[company_id] = "company not found"
            i += 1

    if len(company_ids) == i:                           # if no company is found...
        json_obj = {'error': 'no companies found'}
        return json_obj                                 # just return an error

    # with all the websites, apply gen_json--> find similar websites!!!
    dictionary = create_json.get_json(weblist, sf, num_min, only_website)
    n = min(int(num_max), len(dictionary[u'output']))

    if dictionary:          # if we have similar websites, let's create the json object to be returned

        # ------------------------------------------------------------------------------
        # create the input_website_metadata part

        for website in dictionary['input_website_metadata'].keys():                 # for all the websites

            try:            # not sure it is necessary
                com_name = web_key[website]['legalName']    # company name
                com_id = web_key[website]['id']             # company id to create atoka link

            except KeyError:
                continue

            if com_name not in companies_input.keys():  # we need to say that we have a dictionary for every company
                companies_input[com_name] = dict()
                companies_input[com_name] = {'atoka_link': 'https://atoka.io/azienda/-/' + com_id + "/",
                                             'ateco_code': web_key[website]['ateco'],
                                             'websites': {}}

            companies_input[com_name]['websites'][website] = dictionary['input_website_metadata'][website]

        # ------------------------------------------------------------------------------
        # create the output part
        # ------------------------------------------------------------------------------

        companies = dict()

        # create a dictionary company -> list of websites
        for web_name in dictionary['output']:               # for every website in output

            try:                                        # try to find its company
                company_id = web_key[web_name]['id']     # try to find its name
                company_name = web_key[web_name]['legalName']

            except KeyError:
                continue

            if company_name not in companies.keys():     # websites of the same company together
                companies[company_name] = dict()
                companies[company_name]['websites'] = dict()
                companies[company_name]['company_total_score'] = 0

            companies[company_name]['websites'][web_name] = {'metadata': dictionary['output'][web_name]['metadata'],
                                                             'scores': dictionary['output'][web_name]['scores'],
                                                             'total_score': dictionary['output'][web_name]['total_score'],
                                                             'link': dictionary['output'][web_name]['link']}

            companies[company_name]['company_total_score'] += dictionary['output'][web_name]['total_score']
            companies[company_name]['atoka_link'] = 'https://atoka.io/azienda/-/' + company_id + "/"
            companies[company_name]['ateco_code'] = web_key[web_name]['ateco']

        for company in companies:                       # for every company compute the mean score
            companies[company]['company_total_score'] /= float(len(companies[company])-1)     # mean value of the scores

        output = OrderedDict(sorted(companies.items(), key=lambda x: x[1]['company_total_score'])[:n])

        json_obj = {'input_company_metadata': companies_input,
                    'output': [{'company': key, 'websites': value['websites'],
                                'atoka_link': value['atoka_link'], 'ateco_code': value['ateco_code'],
                                'company_total_score': value['company_total_score']}
                               for key, value in output.iteritems()]}

    else:
        json_obj = {'error': 'websites not present in the models'}

    return json_obj


def website_similarity(dictionary, n):

    # order everything by the total score
    if dictionary:
        dictionary_sort = OrderedDict(sorted(dictionary['output'].items(),
                                             key=lambda x: x[1]['total_score'])[:n])
        # read it as a json object
        json_obj = {'input_website_metadata': dictionary['input_website_metadata'],
                    'output': [{'website': website, 'data': data} for website, data in dictionary_sort.iteritems()]}
    else:
        json_obj = {'error': 'websites not present in the models'}

    return json_obj
