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
                com_id = web_key[website]['legalName']

                if com_id not in companies_input.keys():  # we need to say that we have a dictionary for every company
                    companies_input[com_id] = dict()
                    companies_input[com_id][website] = dictionary['input_website_metadata'][website]
                else:
                    companies_input[com_id][website] = dictionary['input_website_metadata'][website]

            except KeyError:
                pass

        # ------------------------------------------------------------------------------
        # create the output part
        companies = dict()

        # create a dictionary company -> list of websites
        for web_name in dictionary['output']:               # for every website in output

            try:                                        # try to find its company
                company = web_key[web_name]['legalName']     # try to find its name

                if company not in companies.keys():     # websites of the same company together
                    companies[company] = dict()
                    companies[company]['websites'] = dict()
                    companies[company]['websites'][web_name] = dictionary['output'][web_name]
                    companies[company]['company_total_score'] = dictionary['output'][web_name]['total_score']
                else:
                    companies[company][web_name] = dictionary['output'][web_name]
                    companies[company]['company_total_score'] += dictionary['output'][web_name]['total_score']
            except KeyError:
                pass

        for company in companies:                       # for every company compute the mean score
            companies[company]['company_total_score'] /= float(len(companies[company])-1)     # mean value of the scores

        output = OrderedDict(sorted(companies.items(), key=lambda x: x[1]['company_total_score'])[:n])

        json_obj = {'input_company_metadata': companies_input,
                    'output': [{'company': company, 'websites': data['websites'],
                                'company_total_score': data['company_total_score']}
                               for company, data in output.iteritems()]}

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
