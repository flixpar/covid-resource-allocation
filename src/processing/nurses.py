import pandas as pd
import numpy as np
import os
import pickle
from collections import defaultdict


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


def get_county_df():
    county_df = pd.read_csv(os.path.join(DATADIR, 'geography/county_codes.csv')).astype({'code': str})
    county_df.code = county_df.apply(lambda x: '0' + x.code if len(x.code) < 5 else x.code, axis=1)
    county_df.columns = ['code', 'county', 'abbrev']
    state_df = pd.read_csv(os.path.join(DATADIR, 'geography/state_name.csv'))
    state_df.columns = ['state', 'nickname', 'abbrev']
    df = pd.merge(county_df, state_df, on='abbrev', how='outer')
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


def get_population_df(county_df):
    populations_df = pd.read_excel(os.path.join(DATADIR, 'general/co-est2019-annres.xlsx'), skiprows=4, skipfooter=6)
    populations_df.columns = ['Area', 'Census', 'Estimates Base', 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019]
    populations_df['county'] = populations_df.apply(lambda x: x.Area.split(', ')[0].split(' County')[0][1:], axis=1)
    populations_df['state'] = populations_df.apply(lambda x: x.Area.split(', ')[1], axis=1)
    df = pd.merge(populations_df, county_df, on=['county', 'state'], how='inner')
    return df


def get_hospital_beds(county_df):
    def _clean_fips(entry):
        entry = entry.split('.0')[0]
        return '0' + entry if len(entry) < 5 else entry

    hospital_bed_df = pd.read_csv(os.path.join(DATADIR, 'hospitals/Definitive_Healthcare__USA_Hospital_Beds.csv')).astype({'FIPS': str})
    hospital_bed_df.rename(columns={'FIPS': 'code'}, inplace=True)
    hospital_bed_df.code = hospital_bed_df.apply(lambda x: _clean_fips(x.code), axis=1)
    df = pd.merge(hospital_bed_df, county_df, on='code', how='inner')
    return df


def disaggregate_df(df, feature_df, feature):
    def process_entry(entry, df, feature):
        if isinstance(entry, list):
            if feature == 'population':
                return [df[df.code == str(code)][2019].iat[0].astype(int) for code in entry if str(code) in df.code.tolist()]
            elif feature == 'hospital_beds':
                return [df[df.code == str(code)]['NUM_STAFFED_BEDS'].iat[0].astype(int) for code in entry if str(code) in df.code.tolist()]
        return np.nan

    df[feature] = df.apply(lambda x: process_entry(x.county_code, feature_df, feature), axis=1)
    return df


def distribute_resources(df, feature):
    def process_entry(employees, features):
        if isinstance(features, list):
            total = np.sum(features)
            if total == 0 or employees == '**' or employees == np.nan:
                return [0 for feature in features]
            return [np.floor(employees * feature / total).astype(int) for feature in features]
        return np.nan


    df['emp_distributed'] = df.apply(lambda x: process_entry(x.tot_emp, x[feature]), axis=1)
    return df



def main():
    feature = 'hospital_beds'

    cbsa_df = get_cbsa_df()
    county_df = get_county_df()
    population_df = get_population_df(county_df)
    hospital_bed_df = get_hospital_beds(county_df)
    metarea_df = get_metarea_df()#.astype({'tot_emp': int})
    df = pd.merge(metarea_df, cbsa_df, on='area', how='outer')
    df = disaggregate_df(df, hospital_bed_df, feature)
    df = distribute_resources(df, feature)

    df.to_csv(os.path.join(DATADIR, 'nurses/deaggregated_by_' + feature + '.csv'), index=False)

if __name__ == '__main__':
    main()
