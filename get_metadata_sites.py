import os
import numpy as np
import pandas as pd
from datetime import datetime
from data_management import *


zm_multi = {'AT-Neu': 2.5, # selected measurement heights at sites with multiple values in DB
            'BE-Lon': 2.7,
            'BE-Vie': 40,
            'CA-SF3': 20,
            'CA-TP1': 5,
            'CA-TP3': 18,
            'CZ-BK1': 18,
            'DE-Geb': 3.5,
            'DK-Fou': 2.6,
            'FI-Hyy': 24,
            'FR-Gri': 3,
            'FR-LBr': 41.5,
            'IT-BCi': 2.25,
            'IT-CA1': 6.7,
            'IT-CA3': 5.5,
            'IT-Col': 30,
            'IT-Cpz': 25,
            'IT-PT1': 30,
            'IT-Ren': 33,
            'IT-SRo': 22,
            'IT-Tor': 2.5,
            'US-Blo': 10.5,
            'US-KS2': 3.5,
            'US-Ne1': 6,
            'US-Ne2': 6,
            'US-Ne3': 6,
            'US-Oho': 33,
            'US-PFa': 30,
            'US-Whs': 6.5,
            'US-Wkg': 6.4}


def get_fluxnet_BIF_data(site):
    dfi = df_BIF[df_BIF['SITE_ID']==site]

    lat = np.float(dfi[dfi['VARIABLE']=='LOCATION_LAT']['DATAVALUE'].values[0])
    lon = np.float(dfi[dfi['VARIABLE']=='LOCATION_LONG']['DATAVALUE'].values[0])
    igbp = dfi[dfi['VARIABLE']=='IGBP']['DATAVALUE'].values[0]

    hc = np.round(np.mean([np.float(i) for i in dfi[dfi['VARIABLE']=='HEIGHTC']['DATAVALUE'].values]), 1)

    z_ids = dfi[dfi['DATAVALUE']=='USTAR']['GROUP_ID'].values 
    zm = []
    t_zm = []
    for z_id in z_ids:
        zmi = np.float(dfi[(dfi['GROUP_ID']==z_id) & (dfi['VARIABLE']=='VAR_INFO_HEIGHT')]['DATAVALUE'].values[0])
        zm.append(zmi)
        t_zmi = dfi[(dfi['GROUP_ID']==z_id) & (dfi['VARIABLE']=='VAR_INFO_DATE')]['DATAVALUE'].values[0]
        t_zm.append(t_zmi)
    if np.mean(zm)==zm[0]:
        zm = np.mean(zm)
    else:
        zm = zm_multi[site]

    return lat, lon, igbp, hc, zm


def get_soil_text_data(soil_tex_id):
    #GLDAS table
    MAXSMC = ds_sp[ds_sp['ID']==soil_tex_id]['MAXSMC'].values[0]
    BB = ds_sp[ds_sp['ID']==soil_tex_id]['BB'].values[0]
    SATPSI = ds_sp[ds_sp['ID']==soil_tex_id]['SATPSI'].values[0]  * -9.8067 * 10 ** -3 # m to MPa
    SATDK = ds_sp[ds_sp['ID']==soil_tex_id]['SATDK'].values[0] # m s-1
    return BB, SATPSI, SATDK, MAXSMC


def get_modis_LAI_peak(site):
    df_lai_i = df_lai[(df_lai['siteID'] == site) &
                    (df_lai['MCD15A3H_006_FparLai_QC_MODLAND_Description'] == 'Good quality (main algorithm with or without saturation)') 
                    & (df_lai['MCD15A3H_006_FparLai_QC_CloudState_Description'] == 'Significant clouds NOT present (clear)')]
    if len(df_lai_i.index)>0:
        lai95 = np.percentile(df_lai_i['MCD15A3H_006_Lai_500m'].values, 95)
    else:
        lai95 = np.nan
    return lai95


def exclude_periods_(df, site_name, min_year, max_year):
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
    
    if np.isnan(min_year)==0:
        selected_datetimes = [dt for dt in df.index if dt.year >= min_year]
        df = df.ix[selected_datetimes]
    if np.isnan(max_year)==0:
        selected_datetimes = [dt for dt in df.index if dt.year <= max_year]
        df = df.ix[selected_datetimes]
    if x_year is not None:
        selected_datetimes = [dt for dt in df.index if dt.year not in x_year]
        df = df.ix[selected_datetimes]
    return df


def hh_data(f_site, data_directory, swc_i, min_year, max_year):                  
    dir_i = os.path.join(data_directory, f_site)
    [flx, site_name, flx15, fullset, years, suffix] = f_site.split('_')
    try:
        timestep = 'HH'
        f = '%s_%s_%s_%s_%s_%s_%s.csv' % (flx, site_name, flx15, fullset, timestep, years, suffix)
        data = pd.read_csv(os.path.join(dir_i, f), 
                          header = 0, index_col = 'TIMESTAMP_START', parse_dates = True, 
                          infer_datetime_format = True, na_values=-9999, error_bad_lines=False)
    except:
        timestep = 'HR'
        f = '%s_%s_%s_%s_%s_%s_%s.csv' % (flx, site_name, flx15, fullset, timestep, years, suffix)
        data = pd.read_csv(os.path.join(dir_i, f), 
                          header = 0, index_col = 'TIMESTAMP_START', parse_dates = True, 
                          infer_datetime_format = True, na_values=-9999, error_bad_lines=False)

    data['SWC'] = data['SWC_F_MDS_%s' % swc_i] / 100.
    data['SWC'] = [di if (di>0.01 or np.isnan(di)) else 0.01 for di in data['SWC'].values]
    data = exclude_periods_(data, site_name, min_year, max_year)
    return data 


BIF_f = '/mnt/f/Original_data/FLUXNET/FLUXNET2015_2020dl/FLX_AA-Flx_BIF_ALL_20200217/FLX_AA-Flx_BIF_HH_20200217.csv'
df_BIF = pd.read_csv(BIF_f, error_bad_lines=False, engine='python')

lai_file = '/mnt/f/Original_data/FLUXNET/lai-combined-1/lai-combined-1-MCD15A3H-006-results.csv'
df_lai = pd.read_csv(lai_file)

soil_params_file_RUC = '../../DATA/EEGS/NLDAS_soilParams_RUC.csv'
ds_sp = pd.read_csv(soil_params_file_RUC)

data_directory = '/mnt/f/Original_data/FLUXNET/FLUXNET2015_2020dl/unzipped'

selected_vars_0 = ['TA_F_MDS', 'TA_F_MDS_QC',           # deg C
                    'VPD_F_MDS', 'VPD_F_MDS_QC', 'RH',  # hPa
                    'PA_F', 'PA_F_QC',                  # kPa
                    'WS_F', 'WS_F_QC', 'USTAR',         # ms-1
                    'SW_IN_F_MDS',                      # W m-2
                    'CO2_F_MDS', 'CO2_F_MDS_QC',        # µmolCO2 mol-1
                    'SWC', 
                    'LE_F_MDS', 'LE_F_MDS_QC',          # W m-2
                    'H_F_MDS', 'H_F_MDS_QC',            # W m-2
                    'GPP_NT_VUT_REF',                   # µmolCO2 m-2 s-1
                    'PPFD_IN']                          # µmolPhoton m-2 s-1

if __name__ == "__main__":

    selected_f = '../../DATA/EEGS/sel2_sites_list.csv'
    df = pd.read_csv(selected_f)
    
    flx_sites_files = [f for f in os.listdir(data_directory) if f.startswith('FLX')]
    
    site_params = []
    
    for ix in df['ix'].values:
        swc_i = df[df['ix']==ix]['z_sel'].values[0]
        site = df[df['ix']==ix]['site_id'].values[0]
        flx_f = [f for f in flx_sites_files if site in f][0]
        print(site)
        
        min_year = df[df['ix']==ix]['min_year'].values[0]
        max_year = df[df['ix']==ix]['max_year'].values[0]
        soil_tex_id = df[df['ix']==ix]['soil_tex_id'].values[0]
        BB, SATPSI, SATDK, MAXSMC_r  = get_soil_text_data(soil_tex_id)
        lai95 = get_modis_LAI_peak(site)
        lat, lon, igbp, hc, zm = get_fluxnet_BIF_data(site)

        data = hh_data(flx_f, data_directory, swc_i, min_year, max_year)
        data = data[selected_vars_0].dropna()
        
        data = quality_check(data, qc_th=0)
        data = filter_daytime(data)
        data = filter_dew_evaporation(data) 
        for v in [s for s in selected_vars_0 if s.endswith('QC')==0]:
            data = filter_negative(data, v)

        daily_data, selected_days = select_days(flx_f, data_directory, swc_i, 
                                                t_th=2, p_th=0.1, gpp_th=[95, 0.25])
        selected_datetimes = [dt for dt in data.index if dt.date() in selected_days]
        data_0 = data
        data = data.ix[selected_datetimes]
        data = data.dropna()

        MAXSMC = np.nanmax(data['SWC'])
        data['soil_water_pot'] = swc_to_pot_CH(data['SWC'], MAXSMC, BB, SATPSI)
        p_th = np.nanmin(data.resample('D').mean()['soil_water_pot'].values)
        s_th = np.nanmin(data.resample('D').mean()['SWC'].values)
        if p_th < -10:
            BB = (np.log(10) - np.log(-SATPSI)) / (np.log(MAXSMC) - np.log(s_th))
            data['soil_water_pot'] = swc_to_pot_CH(data['SWC'], MAXSMC, BB, SATPSI)
        
        
        data_d = data.resample('D').mean().dropna()
        years = [yi for yi in range(np.min(data_d.index.year), np.max(data_d.index.year) + 1) if yi in data_d.index.year]

        site_params.append([lat, lon, igbp, hc, zm, lai95,
                            np.min(data.index.year), np.max(data.index.year), len(years), len(data_d.dropna().index),
                            BB, MAXSMC, SATPSI, SATDK,
                            np.nanmin(data['soil_water_pot']), np.percentile(data['soil_water_pot'].values, 5),
                            np.nanmin(data_d['soil_water_pot']), np.percentile(data_d['soil_water_pot'].values, 5),
                            ])
        print(site)
    lat, lon, igbp, hc, zm, lai95,\
            min_year, max_year, nyears, ldays, \
            BB, MAXSMC, SATPSI, SATDK, \
            p_min, p_5, p_min_d, p_5_d, = zip(*site_params)
    
    df['lat'] = lat
    df['lon'] = lon
    df['igbp'] = igbp
    df['hc'] = hc
    df['zm'] = zm
    df['lai95'] = lai95
    df['BB'] = BB
    df['MAXSMC'] = MAXSMC
    df['SATPSI'] = SATPSI
    df['min_year'] = min_year
    df['max_year'] = max_year
    df['nyears'] = nyears
    df['ldays'] = ldays
    df['p_min'] = p_min
    df['p_min_d'] = p_min_d
    df['p_5'] = p_5
    df['p_5_d'] = p_5_d


    print(df)
    df.to_csv( '../../DATA/EEGS/sel2_sites_params.csv')
    