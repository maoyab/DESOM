import os
import numpy as np
from pandas import date_range
import pandas as pd
from datetime import datetime
from datetime import timedelta
from constants import *
from et_models import *



def swc_to_pot_CH(theta, theta_s, b, psi0):
    # Brooks and Corey
    se = theta  / theta_s
    psi = (se ** - b)  * psi0
    return psi


def filter_daytime(df):
    df = df[(df['H_F_MDS'] >= 5) & (df['SW_IN_F_MDS'] >= 50)]
    return df


def filter_negative(df, var):
    return df[df[var] > 0]


def filter_dew_evaporation(df):
    df = df[(df['RH'] < 95)]
    return df


def quality_check(df, qc_th=0):
    # QC: 0 = measured; 1 = good quality gapfill; 2 = medium; 3 = poor
    qc_keys = [k for k in df.keys() if k.endswith('_QC')]
    for k in qc_keys:
        if 'SWC' in k: # daily SWC
            df.loc[df[k] > 1, k[:-3]] = np.nan
        else:
            df.loc[df[k] > qc_th, k[:-3]] = np.nan
    df = df[[k for k in df.keys() if k not in qc_keys]]
    return df


def remove_outliers(df, var, mad_th=5, b=1.4826, window='5D'):
    if window is not None:
        df['median'] = df[var].resample(window).median()
    else:
        df['median'] = df[var].median()
    df['median'] = df['median'].interpolate('pad')
    df['delta_abs'] = np.abs(df[var] - df['median'])
    if window is not None:
        df['MAD'] = b * df['delta_abs'].resample(window).median()
    else:
        df['MAD'] = b * df['delta_abs'].median()
    df['MAD'] = df['MAD'].interpolate('pad')
    df['outlier'] = df['delta_abs'] / df['MAD']
    df.loc[df['outlier'] > mad_th, var] = np.nan
    df = df.drop(columns=['outlier', 'MAD', 'delta_abs', 'median'])
    return df


def select_daily_obs_count(df, var, obs_th=8):
    df['count'] = df[var].resample('D').count()
    df['count'] = df['count'].interpolate('pad')
    df.loc[df['count'] < obs_th, var] = np.nan
    df = df.drop(columns=['count', ])
    return df


def select_days(f_site, data_directory, swc_i, p_th=0.1, gpp_th=None, t_th=-np.inf):
    # days with no rain; precedent day with no rain; 
    #growing season based on air T threhold and relative GPP threshold
    
    dir_i = os.path.join(data_directory, f_site)
    [flx, site_name, flx15, fullset, years, suffix] = f_site.split('_')
    timestep = 'DD'
    f = '%s_%s_%s_%s_%s_%s_%s.csv' % (flx, site_name, flx15, fullset, timestep, years, suffix)
    f = os.path.join(dir_i, f)

    data = pd.read_csv(f, 
                      header = 0, index_col = 'TIMESTAMP', parse_dates = True, 
                      infer_datetime_format = True, na_values=-9999)
    data = data[['P_F', 'GPP_NT_VUT_REF', 'TA_F_MDS', 'SWC_F_MDS_%s' % swc_i, 'SWC_F_MDS_%s_QC' % swc_i]]
    
    if p_th is None:
        data['P_F'] = 0
    data.loc[data['P_F'] < p_th, 'P_F'] = 0
    
    if gpp_th is not None:
        gpp_th_ = np.percentile(data['GPP_NT_VUT_REF'].dropna(), gpp_th[0]) * gpp_th[1]
    else:
        gpp_th_ = -np.inf
    
    selected_days = []
    for day, data_i in data.iterrows():
        try:
            p_i = data.iloc[day - timedelta(days=1)]['P_F']
        except:
            p_i = 0
        if site_name.startswith('US-Ne'): # sites missing SWC_QC variable
            if (data_i['P_F'] == 0) \
                            and (data_i['TA_F_MDS'] > t_th) \
                            and (p_i == 0) \
                            and (data_i['GPP_NT_VUT_REF'] >= gpp_th_):
                selected_days.append(day)
        else:
            if (data_i['P_F'] == 0) \
                            and (data_i['TA_F_MDS'] > t_th) \
                            and (p_i == 0) \
                            and (data_i['GPP_NT_VUT_REF'] >= gpp_th_)\
                            and (data_i[ 'SWC_F_MDS_%s_QC' % swc_i] >0):
                selected_days.append(day)
    return data, selected_days


def exclude_periods(df, site_name, df_p):
    # based on metadata and references
    max_year = df_p['max_year'].values[0]
    min_year = df_p['min_year'].values[0]
    selected_datetimes = [dt for dt in df.index if (dt.year >= min_year) and (dt.year <= max_year)]
    df = df.ix[selected_datetimes]

    if site_name == 'DE-Kli':
        x_year = [2007, ] # no corn
    elif site_name == 'FR-Gri':
        x_year = [2008, 2012] # no corn
    elif site_name =='US-ARM':
        x_year = [2005, 2008] # no corn
    elif site_name =='US-Ne2':
        x_year = [2001, 2003, 2005, 2007, 2009, 2011, 2013] # only soy years
    elif site_name =='US-Ne3':
        x_year = [2002, 2004, 2006, 2008, 2010, 2012, 2014] # only maize years
    else:
        x_year = None

    if x_year is not None:
        selected_datetimes = [dt for dt in df.index if dt.year not in x_year]
        df = df.ix[selected_datetimes]
    return df


def unit_conversion(df):
    df['CO2'] = df['CO2_F_MDS'] * 10 ** (-6)                      # [mol/mol]
    df['GPP'] = df['GPP_NT_VUT_REF'] * 10 ** (-6)                 # [mol/m2/s]
    df['VPD_a'] = df['VPD_F_MDS'] / 10. * 1000                    # [Pa]
    df['PARin'] = df['PPFD_IN'] * 10 ** (-6)                      # [molPhoton m-2 s-1]
    df['SWC'] = df['SWC_F_MDS'] / 100.                            # [-]
    df['P_air'] = df['PA_F'] * 1000                               # [Pa]

    df['VPD_l'] = df['VPD_a'] / df['P_air']                       # mol / mol
    df['rho_air_d'] = air_density_dry(df['TA_F_MDS'], df['P_air'], df['RH'])
    df['ET_obs'] = df['LE_F_MDS'] / Lv(df['TA_F_MDS']) /  Mw      # mol H20/m2/s

    df = df.drop(columns=['CO2_F_MDS', 'GPP_NT_VUT_REF', 'VPD_F_MDS', 'PPFD_IN', 'PA_F'])

    return df

if __name__ == "__main__": 
    pass

