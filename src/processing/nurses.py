import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict, namedtuple


DATADIR = '../../data'


def get_field(line):
    field = line[:5].strip()
    if len(field) == 0:
        field = None
    return field, line[8:]


def get_cbsa_df():
    datadir = os.path.join(DATADIR, 'geography')

    with open(os.path.join(datadir, '0312msa.txt'), 'rb') as f:
        lines = f.readlines()
    lines = lines[11:-21] # skip headers and footers

    conversion = defaultdict(list)
    for line in lines:
        line = line.decode('utf-8',errors='ignore').strip()
        if len(line) == 0: continue
        cbsa, line = get_field(line)
        div, line = get_field(line)
        county, components = get_field(line)
        if not county:
            key = cbsa
            conversion[key] = {'div': [], 'county': [], 'components': []}
            continue
        conversion[key]['div'].append(div)
        conversion[key]['county'].append(county)
        conversion[key]['components'].append(components)


    df = pd.DataFrame(columns=['code', 'county_code', 'div_code', 'county_name'])
    for i, (k, v) in enumerate(conversion.items()):
        df.loc[i] = [k, v['county'], v['div'], v['components']]

    cbsa_df = pd.read_csv(os.path.join(datadir, 'cbsas.csv')).astype({'CBSA Code': str})
    cbsa_df.columns = ['code', 'title', 'category']

    df = pd.merge(cbsa_df, df, on='code', how='outer')
    df = df[df.category == 'Metropolitan']
    df.columns = ['area', 'title', 'category', 'county_code', 'div_code', 'county_name']

    return df


def get_metarea_df():
    datadir = os.path.join(DATADIR, 'nurses')
    occ_codes = ['29-1141'] # 29-1141 Registered Nurses
    filename = 'MSA_M2019_dl'
    try:
        df = pickle.load(open('MSA_M2019_dl.pkl', 'rb'))
    except:
        df = pd.read_excel(os.path.join(datadir, 'oesm19ma', filename + '.xlsx'))
        df = df[df.occ_code.isin(occ_codes)].astype({'area': str})
        with open(filename + '.pkl', 'wb') as pickle_file:
            pickle.dump(df, pickle_file)
    return df


def get_population_df():
    df = pd.read_excel(os.path.join(DATADIR, 'general/co-est2019-annres.xlsx'), skiprows=4, skipfooter=6)
    df.columns = ['Area', 'Census', 'Estimates Base', 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    df['county'] = df.apply(lambda x: x.Area.split(', ')[0].split(' County')[0][1:], axis=1)
    df['state'] = df.apply(lambda x: x.Area.split(', ')[1], axis=1)

    counties = pd.read_csv(os.path.join(DATADIR, 'geography/county_codes.csv')).astype({'code': str})
    counties.code = counties.apply(lambda x: '0' + x.code if len(x.code) < 5 else x.code, axis=1)
    counties.columns = ['code', 'county', 'abbrev']
    states = pd.read_csv(os.path.join(DATADIR, 'geography/state_name.csv'))
    states.columns = ['state', 'nickname', 'abbrev']
    counties = pd.merge(counties, states, on='abbrev', how='outer')

    df = pd.merge(df, counties, on=['county', 'state'], how='inner')
    return df


def get_population(entry, df):
    if isinstance(entry, list):
        return [df[df.code == str(code)][2019].iat[0] for code in entry if str(code) in df.code.tolist()]
    return np.nan


def main():
    cbsa_df = get_cbsa_df()
    population_df = get_population_df()
    df = get_metarea_df()
    df = pd.merge(df, cbsa_df, on='area', how='outer')
    df['populations'] = df.apply(lambda x: get_population(x.county_code, population_df), axis=1)
    print(df.head())


if __name__ == '__main__':
    main()
