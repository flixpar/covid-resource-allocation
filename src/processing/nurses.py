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

    df = pd.merge(cbsa_df, df, on='code', how='inner')
    df = df[df.category == 'Metropolitan']
    df.columns = ['area', 'title', 'category', 'county_code', 'div_code', 'county_name']

    return df


def get_county_df():
    county_df = pd.read_csv(os.path.join(DATADIR, 'geography/county_codes.csv')).astype({'code': str})
    county_df.code = county_df.apply(lambda x: '0' + x.code if len(x.code) < 5 else x.code, axis=1)
    county_df.columns = ['code', 'county', 'abbrev']
    state_df = pd.read_csv(os.path.join(DATADIR, 'geography/state_name.csv'))
    state_df.columns = ['state', 'nickname', 'abbrev']
    df = pd.merge(county_df, state_df, on='abbrev', how='inner')
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
        df.tot_emp = df.tot_emp.apply(pd.to_numeric, errors='coerce').fillna(0, downcast='infer')
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
    hospital_bed_df.rename(columns={'FIPS': 'code', 'NUM_STAFFED_BEDS': 'staffed_beds', 'NUM_ICU_BEDS': 'icu_beds'}, inplace=True)
    hospital_bed_df.code = hospital_bed_df.apply(lambda x: _clean_fips(x.code), axis=1)
    hospital_bed_df.staffed_beds = hospital_bed_df.staffed_beds.apply(pd.to_numeric, errors='coerce').fillna(0, downcast='infer')
    hospital_bed_df.icu_beds = hospital_bed_df.icu_beds.apply(pd.to_numeric, errors='coerce').fillna(0, downcast='infer')
    df = pd.merge(hospital_bed_df, county_df, on='code', how='inner')
    return df


def disaggregate_df(df, feature_df, feature):
    def process_entry(entry, df, feature):
        if isinstance(entry, list):
            if feature == 'population':
                return [df[df.code == str(code)][2019].iat[0].astype(int) for code in entry if str(code) in df.code.tolist()]
            elif 'beds' in feature:
                return [df[df.code == str(code)][feature].iat[0].astype(int) for code in entry if str(code) in df.code.tolist()]
        return np.nan

    df[feature] = df.apply(lambda x: process_entry(x.county_code, feature_df, feature), axis=1)
    return df


def distribute_resources(df, feature):
    def process_entry(employees, features):
        if isinstance(features, list):
            total = np.sum(features)
            if total == 0 or employees == '**':
                return [0 for feature in features]
            return [np.floor(employees * feature / total).astype(int) for feature in features]
        return np.nan

    df['emp_distributed_by_' + feature] = df.apply(lambda x: process_entry(x.tot_emp, x[feature]), axis=1)
    return df


def weighted_distribution(df):
    WEIGHTS = {'staffed_beds': 5/7, 'icu_beds': 2/7}
    def process_entry(employees, staffed_beds, icu_beds):
        assert len(staffed_beds) == len(icu_beds)
        if isinstance(staffed_beds, list) and isinstance(icu_beds, list):
            total_staffed = np.sum(staffed_beds)
            total_icu = np.sum(icu_beds)
            if employees == '**' or (total_staffed == 0 and total_icu == 0):
                return [0 for staffed_bed, icu_bed in zip(staffed_beds, icu_beds)]
            elif total_staffed > 0 and total_icu == 0:
                return [np.floor(employees * staffed_bed / total_staffed).astype(int) for staffed_bed in staffed_beds]
            elif total_staffed == 0 and total_icu > 0:
                return [np.floor(employees * icu_bed / total_icu).astype(int) for icu_bed in icu_beds]
            else:
                return [np.floor(employees * (WEIGHTS['staffed_beds'] * staffed_bed / total_staffed
                    + WEIGHTS['icu_beds'] * icu_bed / total_icu)).astype(int) for staffed_bed, icu_bed in zip(staffed_beds, icu_beds)]
        return np.nan

    df['weighted_emp_distribution'] = df.apply(lambda x: process_entry(x.tot_emp, x['staffed_beds'], x['icu_beds']), axis=1)
    return df


def main():
    features = ['staffed_beds', 'icu_beds']

    cbsa_df = get_cbsa_df()
    county_df = get_county_df()
    population_df = get_population_df(county_df)
    hospital_bed_df = get_hospital_beds(county_df)
    metarea_df = get_metarea_df()
    df = pd.merge(metarea_df, cbsa_df, on='area', how='inner')
    for feature in features:
        df = disaggregate_df(df, hospital_bed_df, feature)
        df = distribute_resources(df, feature)
    df = weighted_distribution(df)


    df = df.loc[:, ['area', 'tot_emp', 'county_name', 'staffed_beds', 'icu_beds', 'emp_distributed_by_staffed_beds', 'emp_distributed_by_icu_beds', 'weighted_emp_distribution']]
    df.to_csv(os.path.join(DATADIR, 'nurses/deaggregated_by_hospital_beds.csv'), index=False)

if __name__ == '__main__':
    main()
