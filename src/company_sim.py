# ---------------------------------------------------------------------------------------------------------
# Given a company/websites, get its similar ones. (using websites) + eventually some filter for companies
# ---------------------------------------------------------------------------------------------------------

# id_key and web_key are constructed in 'csv_to_pickle.py'. They are such that info are all contained
# in id_key, and web_key is used when we have a website and we want to know which company it belongs to.
# Since they both are created from the same csv, if c_id=web_key[website] exists, it implies id_key[c_id] exists

from collections import OrderedDict
import re


def company_similarity(create_json, sf, num_min, only_website, company_ids,
                       id_key, web_key, num_max, location, ateco, ateco_dist=5):
    """This function is used to construct a company similarity based on website similarity.
    It works in the same way, but filter on ateco and location can be added"""

    companies_input = dict()        # here it is going to be stored the metadata
    weblist = list()                # here are going to be stored all the websites of the companies asked
    ateco_list = list()             # we are going to store the ateco codes and labels of the companies

    for company_id in company_ids:                  # iterate through all the companies in the query

        try:                                        # and try to find their websites
            websites = id_key[str(company_id)]['websites']
            weblist += websites                     # and add the found websites to their websites list
        except KeyError:
            companies_input[company_id] = "company not found"

    if not weblist:                           # if no company is found (<=> weblist is empty)...
        json_obj = {'error': 'no companies found'}
        return json_obj                                 # return an error

    # with all the websites, apply gen_json --> find similar websites!!!
    dictionary = create_json.get_json(weblist, sf, num_min, only_website)

    if dictionary:          # if we have similar websites, let's create the json object to be returned
        # ----------------------------------------
        # create the input_website_metadata part
        # ----------------------------------------

        n = min(int(num_max), len(dictionary['output']))    # maximum number of company selected
        ateco_input = list()                # used for the filters
        location_list = list()           # collect the locations we want to filter of the input

        for website in dictionary['input_website_metadata'].keys():    # for all the websites in input

            try:
                com_id = web_key[website]           # try to find the company id that correspond to that website
            except KeyError:
                continue

            # if a comp_id can be retrieved from web_key, we are sure the other information can be retrieved from
            # id_key, due to the way we created them (see csv_to_pickle.py)
            com_name = id_key[str(com_id)]['legalName']    # company name
            ateco_input.append(id_key[str(com_id)]['ateco'])

            if com_name not in companies_input.keys():
                # for every new company in input create a dictionary with its informations...
                companies_input[com_name] = {'atoka_link': 'https://atoka.io/azienda/-/' + com_id + "/",  # link atoka
                                             'ateco': id_key[str(com_id)]['ateco'],                # its ateco code/label
                                             # locations info, maybe eliminate from final json, atm kept for debugging
                                             # ['location'] contains a dictionary with info on location
                                             'locations': id_key[str(com_id)]['location'],
                                             'websites': {}}      # its websites (added in next line)
                if location != 'false':
                    for loc in id_key[str(com_id)]['location'][location]:
                        location_list.append(loc)

            # ...and add all its websites information
            companies_input[com_name]['websites'][website] = dictionary['input_website_metadata'][website]

        # --------------------------------------------------------------------------------------
        # create the output part (suggested companies with their information) - FILTER BY ATECO
        # --------------------------------------------------------------------------------------

        companies = dict()          # where to store the (filtered) output companies

        if ateco == 'false' or ateco == 'strict' or ateco == 'distance':

            for webs in weblist:                    # for every website in the input list, retrieve its ateco
                try:
                    atk = id_key[str(web_key[webs])]['ateco']    # this is used if ateco!='auto'
                    ateco_list += atk
                except KeyError:
                    pass

            # create a dictionary company -> list of websites
            for web_name in dictionary['output']:               # for every website in output

                try:                                        # try to find its company id
                    company_id = web_key[web_name]
                except KeyError:
                    continue

                company_name = id_key[str(company_id)]['legalName']          # retrieve company name
                ateco_code = id_key[str(company_id)]['ateco']                # and ateco

                if ateco_filter(ateco, ateco_list, ateco_code, ateco_dist):
                    # only for the companies that respect the ateco request
                    if company_name not in companies.keys():     # websites of the same company together
                        companies[company_name] = dict()
                        companies[company_name]['websites'] = dict()
                        companies[company_name]['company_total_score'] = 0

                    # construct output part: contains websites, with their metadata
                    companies[company_name]['websites'][web_name] = dictionary['output'][web_name]

                    # and we add for every company the total score, the link to atoka and the ateco code
                    companies[company_name]['company_total_score'] += dictionary['output'][web_name]['total_score']
                    companies[company_name]['atoka_link'] = 'https://atoka.io/azienda/-/' + company_id + "/"
                    companies[company_name]['ateco'] = ateco_code
                    companies[company_name]['location'] = id_key[str(company_id)]['location']

        elif ateco == 'auto':
            # create a dictionary company -> list of websites
            ateco_output = list()

            for web_name in dictionary['output']:               # for every website in output

                try:                                        # try to find its company
                    company_id = web_key[web_name]     # try to find its id
                except KeyError:
                    continue

                ateco_output.append(id_key[str(company_id)]['ateco'])     # append its ateco

            good_ateco = ateco_auto(ateco_input, ateco_output)      # find more frequent ateco

            for web_name in dictionary['output']:
                try:
                    company_id = web_key[web_name]     # try to find its id
                except KeyError:
                    continue

                company_name = id_key[str(company_id)]['legalName']

                if id_key[str(company_id)]['ateco'][0] in good_ateco:
                    if company_name not in companies.keys():     # websites of the same company together
                        companies[company_name] = dict()
                        companies[company_name]['websites'] = dict()
                        companies[company_name]['company_total_score'] = 0

                    # construct output part: contains websites, with their metadata
                    companies[company_name]['websites'][web_name] = {'metadata': dictionary['output'][web_name]['metadata'],
                                                                     'scores': dictionary['output'][web_name]['scores'],
                                                                     'total_score': dictionary['output'][web_name]['total_score'],
                                                                     'link': dictionary['output'][web_name]['link']}

                    # and we add for every company the total score, the link to atoka and the ateco code
                    companies[company_name]['company_total_score'] += dictionary['output'][web_name]['total_score']
                    companies[company_name]['atoka_link'] = 'https://atoka.io/azienda/-/' + company_id + "/"
                    companies[company_name]['ateco'] = id_key[str(company_id)]['ateco']
                    companies[company_name]['location'] = id_key[str(company_id)]['location']

        for company in companies:
            # for every company compute the mean score
            companies[company]['company_total_score'] /= float(len(companies[company])-1)

        # -----------------------------------------------------------------------------------------
        # create the output part (suggested companies with their information) - FILTER BY LOCATION
        # -----------------------------------------------------------------------------------------

        comp_to_be_filtered = list()
        if location != 'false':             # so its 'macroregon', 'region', 'province' or 'municipality'

            for company in companies:       # for every company that its good for the ateco filter
                locations = companies[company]['location'][location]    # list of the location we're interest in
                to_be_filtered = True

                for loc in locations:          # if at least one is in the input list of locations, keep the company in
                    if loc in location_list:   # the output list
                        to_be_filtered = False
                        break       # no need to complete the cycle

                if to_be_filtered:
                    comp_to_be_filtered.append(company)         # maybe smarter way of doing it?

            for item in comp_to_be_filtered:
                del companies[item]

        # order the companies that still are in output by means of the total score
        output = OrderedDict(sorted(companies.items(), key=lambda x: x[1]['company_total_score'])[:n])

        json_obj = {'input_company_metadata': companies_input,
                    'output': [{'company': key, 'websites': value['websites'],
                                'atoka_link': value['atoka_link'], 'ateco': value['ateco'],
                                'company_total_score': value['company_total_score'], 'location': value['location']}
                               for key, value in output.iteritems()]}

        # --------------------------------------------------------------------------------------
        # create the output part (suggested companies with their information) - FILTER BY LOCATION
        # --------------------------------------------------------------------------------------

    else:       # if not dictionary (i.e. no websites are found in the models by get_json)
        json_obj = {'error': 'websites of the companies not present in the models'}

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


def ateco_filter(ateco_par, ateco_company_1, ateco_company_2, ateco_dist=5):
    # we expect ateco_par to be 'strict', 'distance' or 'false'.
    # ateco_company_1 to be a list of lists code-label and ateco_company_2 a list code-label

    if isinstance(ateco_company_1[0], list):
        ateco_codes = [item[0] for item in ateco_company_1]         # list of all the codes
        ateco_labels = [item[1] for item in ateco_company_1]        # list of all the labels
    else:
        ateco_codes = [ateco_company_1[0]]
        ateco_labels = [ateco_company_1[1]]
        ateco_company_1 = [ateco_company_1]

    if ateco_par == 'strict':
        # 'strict' means that we eliminate all the companies that have ateco codes
        # different from all the ones in the input companies list
        if ateco_company_2[0] in ateco_codes or ateco_company_2[1] in ateco_labels:
            return True
        else:
            return False

    elif ateco_par == 'distance':
        # 'distance' means that eliminate all the companies that have an ateco code
        # that is too distant from the ones in the companies list
        for item in ateco_company_1:
            if ateco_distance(item, ateco_company_2) <= ateco_dist:  # 5 was chosen by sentiment
                return True
        return False

    elif ateco_par == 'false':
        # 'false' means that everything is fine
        return True


def ateco_distance(ateco_company_1, ateco_company_2):
    """compute the distance between ateco codes, as shortest path in the ateco tree"""

    # if the ateco label is the same, the distance is zero!
    if ateco_company_1[1] == ateco_company_2[1]:
        return 0

    else:

        # eliminate the dots from the ateco code format
        ac1 = re.sub(r'\.+', "", ateco_company_1[0])
        ac2 = re.sub(r'\.+', "", ateco_company_2[0])

        # compute the level (of the tree) of those ateco code
        al1_lvl = len(ac1) - 1
        al2_lvl = len(ac2) - 1

        # if they are not at least of length 2, there is something wrong with the data
        if al1_lvl <= 0 or al2_lvl <= 0:
            return 20

        # compute how much they share (the more, the most similar they are)
        i = 0
        while i < min(len(ac1), len(ac2)) and ac1[i] == ac2[i]:
            i += 1

        if i == 0 or i == 1:
            if ateco_letter(int(ac1[:2])) == ateco_letter(int(ac2[:2])):
                return al1_lvl + al2_lvl
            else:
                return 20
        else:
            return al1_lvl + al2_lvl - ((i - 1) * 2)


def ateco_letter(number):
    """function that associate atecoCode numeric with its father letter """
    if number < 4:
        return "A"
    elif number < 10:
        return "B"
    elif number < 34:
        return "C"
    elif number == 35:
        return "D"
    elif number < 40:
        return "E"
    elif number < 44:
        return "F"
    elif number < 48:
        return "G"
    elif number < 54:
        return "H"
    elif number < 57:
        return "I"
    elif number < 64:
        return "J"
    elif number < 67:
        return "K"
    elif number == 68:
        return "L"
    elif number < 76:
        return "M"
    elif number < 83:
        return "N"
    elif number == 84:
        return "O"
    elif number == 85:
        return "P"
    elif number < 89:
        return "Q"
    elif number < 94:
        return "R"
    elif number < 97:
        return "S"
    elif number < 99:
        return "T"
    elif number == 99:
        return "U"
    else:
        raise KeyError


def ateco_auto(input_list, output_list):
    # create a dictionary to store the frequencies of the ateco codes
    frequency = dict()

    # list with all the atecos (maybe in the future we should weight more the input companies one)
    total_list = [item[0] for item in input_list + output_list]

    # count the frequency of every ateco code
    for element in total_list:
        if element not in frequency.keys():
            frequency[element] = 0
        frequency[element] += 1

    max_freq = max(frequency.values())           # how frequent is the most frequent ateco?

    good_ateco = list()         # list for the ateco code that are frequent enough

    for code in frequency:
        if frequency[code] >= 0.25 * max_freq:          # keep just atecos that are at least 25%*max frequent.
            good_ateco.append(code)                     # probably there is a better choice

    return good_ateco
