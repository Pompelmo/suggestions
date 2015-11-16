# ----------------------------------------------------------------------
# Given a company, get its similar ones. (using websites)
# ----------------------------------------------------------------------

from collections import OrderedDict
import re


def company_similarity(create_json, sf, num_min, only_website, company_ids, id_key, web_key, num_max, ateco):

    companies_input = dict()        # here it is going to be stored the metadata
    weblist = list()                # here are going to be stored all the websites of the companies asked
    ateco_list = list()             # we are going to store the ateco codes and labels of the companies
    i = 0

    for company_id in company_ids:                  # iterate through all the companies in the query

        try:                                        # and try to find their websites
            websites = id_key[company_id]['websites']
            weblist += websites                     # and add the found websites to their websites list

        except KeyError:
            companies_input[company_id] = "company not found"
            i += 1

        try:
            atk = id_key[company_id]['ateco']
            ateco_list += atk
        except KeyError:
            pass

    if len(company_ids) == i:                           # if no company is found...
        json_obj = {'error': 'no companies found'}
        return json_obj                                 # return an error

    # with all the websites, apply gen_json--> find similar websites!!!
    dictionary = create_json.get_json(weblist, sf, num_min, only_website)
    n = min(int(num_max), len(dictionary[u'output']))

    if dictionary:          # if we have similar websites, let's create the json object to be returned
        # ------------------------------------------------------------------------------
        # create the input_website_metadata part

        ateco_input = list()

        for website in dictionary['input_website_metadata'].keys():                 # for all the websites

            try:            # not sure it is necessary
                com_name = web_key[website]['legalName']    # company name
                com_id = web_key[website]['id']             # company id to create atoka link
                ateco_input.append(web_key[website]['ateco'])

            except KeyError:
                continue

            if com_name not in companies_input.keys():
                # specify we have a dictionary for every company, since we add the key 'websites' below
                companies_input[com_name] = dict()
                companies_input[com_name] = {'atoka_link': 'https://atoka.io/azienda/-/' + com_id + "/",
                                             'ateco_code': web_key[website]['ateco'],
                                             'websites': {}}

            companies_input[com_name]['websites'][website] = dictionary['input_website_metadata'][website]

        # ------------------------------------------------------------------------------
        # create the output part
        # ------------------------------------------------------------------------------

        companies = dict()

        if ateco == 'false' or ateco == 'strict' or ateco == 'distance':
            # create a dictionary company -> list of websites
            for web_name in dictionary['output']:               # for every website in output

                try:                                        # try to find its company
                    company_id = web_key[web_name]['id']     # try to find its name
                    company_name = web_key[web_name]['legalName']

                except KeyError:
                    continue

                ateco_code = web_key[web_name]['ateco']

                if ateco_filter(ateco, ateco_list, ateco_code):
                    # only for the companies that respect the ateco request
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
                    companies[company_name]['ateco_code'] = web_key[web_name]['ateco']

            for company in companies:
                # for every company compute the mean score
                companies[company]['company_total_score'] /= float(len(companies[company])-1)

            # order the companies in output by means of the total score
            output = OrderedDict(sorted(companies.items(), key=lambda x: x[1]['company_total_score'])[:n])

            json_obj = {'input_company_metadata': companies_input,
                        'output': [{'company': key, 'websites': value['websites'],
                                    'atoka_link': value['atoka_link'], 'ateco_code': value['ateco_code'],
                                    'company_total_score': value['company_total_score']}
                                   for key, value in output.iteritems()]}

        elif ateco == 'auto':
            # create a dictionary company -> list of websites
            ateco_output = list()

            for web_name in dictionary['output']:               # for every website in output

                try:                                        # try to find its company
                    ateco_output.append(web_key[web_name]['ateco'])     # try to find its ateco

                except KeyError:
                    continue

            good_ateco = ateco_auto(ateco_input, ateco_output)

            for web_name in dictionary['output']:
                try:
                    company_id = web_key[web_name]['id']     # try to find its name
                    company_name = web_key[web_name]['legalName']
                except KeyError:
                    continue

                if web_key[web_name]['ateco'][0] in good_ateco:
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
                    companies[company_name]['ateco_code'] = web_key[web_name]['ateco']

            for company in companies:
                # for every company compute the mean score
                companies[company]['company_total_score'] /= float(len(companies[company])-1)

            # order the companies in output by means of the total score
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


def ateco_filter(ateco_par, ateco_company_1, ateco_company_2):
    # we expect ateco_par to be 'strict', 'distance' or 'false'.
    # ateco_company_1 to be a list of lists code-label and ateco_company_2 a list code-label

    if isinstance(ateco_company_1[0], list):
        ateco_codes = [item[0] for item in ateco_company_1]         # list of all the codes
        ateco_labels = [item[1] for item in ateco_company_1]        # list of all the labels
    else:
        ateco_codes = ateco_company_1[0]
        ateco_labels = ateco_company_1[1]
        ateco_company_1 = [ateco_company_1]

    if ateco_par == 'strict':
        # 'strict' means that we eliminate all the companies that have ateco codes
        # different from all the ones in the input companies list
        if ateco_company_2[0] in ateco_codes or ateco_company_2[1] in ateco_labels:
            # if ateco_codes and ateco_labels are just string and not lists, 'in' works as an equality
            return True
        else:
            return False

    elif ateco_par == 'distance':
        # 'distance' means that eliminate all the companies that have an ateco code
        # that is too distant from the ones in the companies list
        for item in ateco_company_1:
            if ateco_distance(item, ateco_company_2) <= 5:  # 5 was chosen by sentiment
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

    max_freq = max(frequency.values())

    good_ateco = list()         # list for the ateco code that are frequent enough

    for code in frequency:
        if frequency[code] >= 0.25 * max_freq:
            good_ateco.append(code)

    return good_ateco