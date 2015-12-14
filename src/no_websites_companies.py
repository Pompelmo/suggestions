# ---------------------------------------------------------------------------------------
# script to have a company similarity for companies without websites
# ---------------------------------------------------------------------------------------

from collections import OrderedDict
from elasticsearch import Elasticsearch

es = Elasticsearch(['http://es-idg:9200'])
index = 'companies-latest'


def revenue(size_list):
    sizes = [int(item) for item in size_list]
    s_min = min(sizes)
    s_max = max(sizes)

    if s_min == 0:
        gte = 0
    elif s_min == 1:          # a company with a revenue under 2mln is considered 'micro' company
        gte = 0
    elif s_min == 2:       # a company with a revenue under 10mln is considered 'small' company
        gte = 2000000
    elif s_min == 3:       # a company with a revenue under 50mln is considered 'medium' company
        gte = 10000000
    elif s_min == 4:                           # otherwise is considered a 'big' company
        gte = 50000000

    if s_max == 0:
        lt = 1000000000000000
    elif s_max == 1:          # a company with a revenue under 2mln is considered 'micro' company
        lt = 2000000
    elif s_max == 2:       # a company with a revenue under 10mln is considered 'small' company
        lt = 10000000
    elif s_max == 3:       # a company with a revenue under 50mln is considered 'medium' company
        lt = 50000000
    elif s_max == 4:                           # otherwise is considered a 'big' company
        lt = 1000000000000000

    return gte, lt


def revenue_to_size(rev):
    try:
        rev = int(rev)
    except TypeError:
        return '0'

    if rev <= 2000000:          # a company with a revenue under 2mln is considered 'micro' company
        return '1'
    elif rev <= 10000000:       # a company with a revenue under 10mln is considered 'small' company
        return '2'
    elif rev <= 50000000:       # a company with a revenue under 50mln is considered 'medium' company
        return '3'
    else:                           # otherwise is considered a 'big' company
        return '4'


def no_web_info(num_max, company_ids, id_key):
    # --------------------------------------------------
    # construct input metadata part
    # --------------------------------------------------
    companies_input = dict()

    for comp_id in company_ids:    # for all the websites in input

        comp_id = str(comp_id)
        com_name = id_key[comp_id]['legalName']    # company name
        atec = id_key[comp_id]['ateco']
        loc = id_key[comp_id]['location']
        siz = id_key[comp_id]['size']
        web = id_key[comp_id]['websites']

        if com_name not in companies_input.keys():
            # for every new company in input create a dictionary with its informations...
            companies_input[com_name] = {'atoka_link': 'https://atoka.io/azienda/-/' + comp_id + "/",  # link atoka
                                         'ateco': atec,
                                         'locations': loc,
                                         'size': siz,
                                         'websites': web}      # its websites (added in next line)

    # --------------------------------------------------
    # construct the output part
    # --------------------------------------------------
    atecoes = list()
    locations = list()
    size_list = list()

    for company_id in company_ids:                  # iterate through all the companies in the query
        try:                                        # and try to find their websites
            ateco = id_key[str(company_id)]['ateco']
            atecoes.append(ateco[0])
            location = id_key[str(company_id)]['location']['province']
            locations += location
            size_list.append(id_key[str(company_id)]['size'])
        except KeyError:
            pass

    gte, lt = revenue(size_list)
    n = int(num_max) + len(company_ids)

    query = {           # query the index: we want at least same ateco and same province
        'size': n,
        'query': {
            'filtered': {
                'filter': {
                    'bool': {
                        'must': [{
                            'terms': {
                                'ateco.leafCode': atecoes
                            }
                        },
                            {
                            'terms': {
                                'activityAdminDiv.province': locations
                            }
                        }],
                        'should': {
                            'range': {
                                'ranking.revenue.value': {'gte': gte, 'lt': lt}
                            }
                        }
                    }
                }
            }
        }
    }

    res = es.search(index=index, body=query)

    companies = dict()

    for company in res['hits']['hits']:         # metadata for the output
        source = company['_source']
        data = source.keys()
        atoka_link = 'https://atoka.io/azienda/-/' + company['_id'] + "/"
        lname = source['legalName']

        if 'websites' in data:          # their websites
            websites = [website['website'] for website in source['websites']]
        else:
            websites = []

        if 'ateco' in data:             # their ateco
            ateco = [source['ateco']['leafCode'], source['ateco']['label']]
        else:
            ateco = []

        if 'activityAdminDiv' in data:          # their location
            locations = source['activityAdminDiv']
        else:
            locations = dict()

        if 'ranking' in data and 'revenue' in data['ranking']:          # and their dimension
            size = revenue_to_size(int(source['ranking']['revenue']['value']))
        else:
            size = '0'

        # create the output part for every company hit:
        companies[lname] = {'websites': websites, 'atoka_link': atoka_link, 'ateco': ateco, 'location': locations,
                            'size': size, 'score': company['_score']}

    # order by means of elasticsearch score
    output = OrderedDict(sorted(companies.items(), key=lambda x: x[1]['score']))

    # create the final object
    json_obj = {'input_company_metadata': companies_input,
                'output': [{'company': key, 'websites': value['websites'],
                            'atoka_link': value['atoka_link'], 'ateco': value['ateco'],
                            'company_total_score': value['company_total_score'], 'location': value['location'],
                            'size': value['size']}
                           for key, value in output.iteritems()]}
    return json_obj
