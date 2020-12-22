import os
import numpy as np
import pandas as pd
from datetime import datetime
from data_management import *
from et_models import *
from gs_models import *

lai_file = '/mnt/m/Original_data/FLUXNET/lai-combined-1/lai-combined-1-MCD15A3H-006-results.csv'
sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')
pft_params = pd.read_csv( '../../DATA/EEGS/selected_pft_params.csv')
data_directory = '/mnt/m/Original_data/FLUXNET/FLUXNET2015_2020dl/unzipped'

def insert_LAI(df, site_0, df_p):
    df_lai = pd.read_csv(lai_file)
    site = site_0
    while site.endswith('R'):
        site = site[:-1]

    if site=='IT-Ro1':
        df_lai = df_lai[(df_lai['siteID']=='IT-Ro2')
                        & (df_lai['MCD15A3H_006_FparLai_QC_MODLAND_Description']=='Good quality (main algorithm with or without saturation)') 
                        & (df_lai['MCD15A3H_006_FparLai_QC_CloudState_Description']=='Significant clouds NOT present (clear)') 
                        & (df_lai['MCD15A3H_006_FparExtra_QC_Aerosol_Description']=='No or low atmospheric aerosol levels detected')
                        & (df_lai['MCD15A3H_006_FparLai_QC_SCF_QC_Description']=='Main (RT) method used, best result possible (no saturation)')
                        ]
    else:
        df_lai = df_lai[(df_lai['siteID'] == site)
                        & (df_lai['MCD15A3H_006_FparLai_QC_MODLAND_Description']=='Good quality (main algorithm with or without saturation)') 
                        & (df_lai['MCD15A3H_006_FparLai_QC_CloudState_Description']=='Significant clouds NOT present (clear)') 
                        & (df_lai['MCD15A3H_006_FparExtra_QC_Aerosol_Description']=='No or low atmospheric aerosol levels detected')
                        &(df_lai['MCD15A3H_006_FparLai_QC_SCF_QC_Description']=='Main (RT) method used, best result possible (no saturation)')
                        ]
    lai = df_lai['MCD15A3H_006_Lai_500m'].values

    date = [datetime.strptime(dt, '%m/%d/%Y')
                             for dt in df_lai['Date'].values]

    df_lai = pd.DataFrame({'LAI': lai}, index=date)
    df_lai['smooth_LAI'] = np.nan
    df_lai = df_lai.resample('D').mean()
    df_lai = df_lai.interpolate()
    df_lai = df_lai.reindex(pd.date_range(start='2003-01-01',
                                        end='2014-12-31',
                                        freq='D'))

    df_lai = df_lai[~((df_lai.index.month == 2) & (df_lai.index.day == 29))]
    df_lai['Year'] = df_lai.index.year
    years = [y for y in list(set( df_lai['Year'].values))]
    doy = []
    for y in years:
        doy = np.concatenate((doy, range(1,366)), axis=None)
    doy = [np.int(di) for di in doy]
    df_lai['DOY'] = doy
    
    m_avg_lai = []
    for c, year in enumerate(years):
        df_lai_ii = df_lai[df_lai['Year']==year]
        m_avg_lai.append(df_lai_ii['LAI'].values)
    m_avg_lai = zip(*m_avg_lai)
    m_avg_lai = [np.nanmean(mi) for mi in m_avg_lai]
    df_lai['smooth_LAI'] = [m_avg_lai[di-1] for di in df_lai['DOY'].values]
    m_avg_lai_r = np.convolve(df_lai['smooth_LAI'].values, np.ones(31) / 31, mode='same')
    m_avg_lai_r = list(m_avg_lai_r[365:365*2])
    m_avg_lai_r.insert(31+28, m_avg_lai_r[31+27])
    lai_dates = pd.date_range(start='2000-01-01',
                              end='2000-12-31',
                              freq='1D')

    df['LAI'] =  np.nan
    for month, day, lai in zip(lai_dates.month, lai_dates.day, m_avg_lai_r):
        df.loc[(df.index.day == day) & (df.index.month == month), 'LAI'] = lai
    return df

def insert_soil_water_potential(df, df_p):        
    BB = df_p['BB'].values[0]
    SATPSI = df_p['SATPSI'].values[0]
    MAXSMC = df_p['MAXSMC'].values[0]
    psi = swc_to_pot_CH(df['SWC'], MAXSMC, BB, SATPSI) 
    df['soil_water_pot'] = psi
    df['S'] = df['SWC'] / MAXSMC
    return df

def insert_conductances(df, df_p):
    hm = df_p['zm'].values[0]                           # [m] WS measurement height 
    hc = df_p['hc'].values[0]                           # [m] canopy height
    df['Rn-G'] = df['LE_F_MDS'] + df['H_F_MDS'] 
    
    df['Ga'] = [aerodynamic_cond_it(ws, h_flux, air_temp, ustar, hm, hc, rho) \
                    for ws, h_flux, air_temp, ustar, rho \
                        in zip(df['WS_F'], df['H_F_MDS'],
                               df['TA_F_MDS'], df['USTAR'], df['rho_air_d'])]                           # [m/s]
    df['G_surf'] = surface_conductance_PM(df['P_air'], df['TA_F_MDS'], df['VPD_a'],
                                         df['Ga'], df['LE_F_MDS'], df['H_F_MDS'], df['rho_air_d'])      # [m/s]
    df['G_surf_mol'] =  g_to_mol(df['G_surf'], df['P_air'], df['TA_F_MDS'])                             # [mol m-2 s-1]
    df['Ga_mol'] =  g_to_mol(df['Ga'], df['P_air'], df['TA_F_MDS'])                                     # [mol m-2 s-1]
    df['VPD_l_inv'] = leaf_vpd(df['ET_obs'], df['G_surf_mol'], df['Ga_mol'])                            # [mol / mol]
    
    df = filter_negative(df, 'Ga')
    df = filter_negative(df, 'G_surf_mol')
    df = df.drop(columns=['H_F_MDS', 'USTAR', 'WS_F', 'RH'])

    return df

def insert_photosynthesis_variables(df, df_p):
    pft = df_p['pft'].values[0]
    pft_params_i = pft_params[pft_params['PFT'] == pft]

    a2_25C = Kc_25C * ( 1 - Cao / Ko_25C)
    J_max_25C = pft_params_i['J_max'].values[0]
    Vc_max_25C = pft_params_i['Vc_max'].values[0]
    Ha_Vcmax = pft_params_i['Ha_Vcmax'].values[0]
    Topt_Vcmax = pft_params_i['Topt_Vcmax'].values[0]
    Ha_Jmax = pft_params_i['Ha_Jmax'].values[0]
    Topt_Jmax = pft_params_i['Topt_Jmax'].values[0]

    J_25p = electron_transport_rate(df['PARin'], J_max_25C)
    df['k2_0'] = photo_k2(J_25p, Vc_max_25C,  a2_25C)
    df['k1_0'] = photo_k1(J_25p)
    df['gamma_star_0'] = gamma_star_25C
    
    a2 = cal_a2_Rub(df['TA_F_MDS'])
    J_max = peaked_function_Medlyn(J_max_25C, df['TA_F_MDS'], Topt_Jmax, Ha_Jmax)
    Vc_max = peaked_function_Medlyn(Vc_max_25C, df['TA_F_MDS'], Topt_Vcmax, Ha_Vcmax)
    J = electron_transport_rate(df['PARin'], J_max)
    df['k2_Tl'] = photo_k2(J, Vc_max, a2)
    df['k1_Tl'] = photo_k1(J)
    df['gamma_star_Tl'] =  cal_gamma_star_temp(df['TA_F_MDS'])

    return df

def insert_hydraulics(df, df_p):
    pft = df_p['pft'].values[0]
    pft_params_i = pft_params[pft_params['PFT'] == pft]
    df['a_ref'] = pft_params_i['a'].values[0]
    df['psi_50_ref'] = pft_params_i['psi_50'].values[0]
    hc = df_p['hc'].values[0]
    df['kmax'] = pft_params_i['kmax'].values[0] * np.max(df['LAI']) # mol m-2 s-2 MPa-1
    
    df['canopy_water_pot_pd'] = df['soil_water_pot'] - hc * grav_acc * rho_w * 10 ** -6
    df['leaf_water_pot_invET'] = cal_psi_leaf(df['canopy_water_pot_pd'], 
                                                pft_params_i['psi_50'].values[0],
                                                pft_params_i['kmax'].values[0] * np.max(df['LAI']),
                                                pft_params_i['a'].values[0],
                                                df['ET_obs'])
    df['leaf_water_pot_invET1'] = cal_psi_leaf(df['canopy_water_pot_pd'], 
                                                pft_params_i['psi_50'].values[0],
                                                pft_params_i['kmax'].values[0] * np.max(df['LAI']),
                                                1,
                                                df['ET_obs'])
    df['leaf_water_pot_invET1b'] = cal_psi_leaf_1(df['soil_water_pot'], 
                                                pft_params_i['psi_50'].values[0],
                                                pft_params_i['kmax'].values[0] * np.max(df['LAI']),
                                                hc,                
                                                df['ET_obs'])
    return df

def make_df(f_site):
    selected_vars_0 = ['TA_F_MDS', 'TA_F_MDS_QC',           # deg C
                        'VPD_F_MDS', 'VPD_F_MDS_QC', 'RH',  # hPa
                        'PA_F', 'PA_F_QC',                  # kPa
                        'WS_F', 'WS_F_QC', 'USTAR',         # ms-1
                        'SW_IN_F_MDS',                      # W m-2
                        'CO2_F_MDS', 'CO2_F_MDS_QC',        # µmolCO2 mol-1
                        'SWC_F_MDS', 'SWC_F_MDS_QC',        # %
                        'LE_F_MDS', 'LE_F_MDS_QC',          # W m-2
                        'H_F_MDS', 'H_F_MDS_QC',            # W m-2
                        'GPP_NT_VUT_REF',                   # µmolCO2 m-2 s-1
                        'PPFD_IN']                          # µmolPhoton m-2 s-1
       
    dir_i = os.path.join(data_directory, f_site)
    [flx, site_name, flx15, fullset, years, suffix] = f_site.split('_')
    sites_params_i = sites_params[sites_params['site_id'] == site_name]  

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
    
    swc_i = sites_params_i['z_sel'].values[0]
    data['SWC_F_MDS'] = data['SWC_F_MDS_%s' % swc_i]
    data['SWC_F_MDS_QC'] = data['SWC_F_MDS_%s_QC' % swc_i]

    data = data[selected_vars_0]

    data = exclude_periods(data, site_name, sites_params_i)
    data = quality_check(data, qc_th=0)
    data = filter_daytime(data)
    data = filter_dew_evaporation(data)
    for v in [s for s in selected_vars_0 if s.endswith('QC')==0]:
        data = filter_negative(data, v)
    
    data = unit_conversion(data)

    data = insert_LAI(data, site_name, sites_params_i)
    data = insert_soil_water_potential(data, sites_params_i)
    data = insert_conductances(data, sites_params_i)

    data = data.reindex(pd.date_range(start=data.index.min(),
                                        end=data.index.max(),
                                        freq='30min'))

    for var in ['GPP', 'Ga', 'CO2', 'TA_F_MDS', 'PARin', 'VPD_l', 'LE_F_MDS']:
        data = remove_outliers(data, var, mad_th=5, window='5D')
    
    daily_data, selected_days = select_days(f_site, data_directory, swc_i, 
                                            t_th=2, p_th=0.1, gpp_th=[95, 0.25])
    selected_datetimes = [dt for dt in data.index if dt.date() in selected_days]
    data = data.ix[selected_datetimes]
    data = data.dropna()

    data = insert_hydraulics(data, sites_params_i)
    data = insert_photosynthesis_variables(data, sites_params_i)
    
    return data
    

if __name__ == "__main__": 

# get data ......................................................................................

    out_directory = '../../DATA/EEGS/HH_data_revision_w5'
    sites_files = [f for f in os.listdir(data_directory) if f.startswith('FLX')]
    for ix in sites_params['ix'].values:
        swc_i = sites_params[sites_params['ix']==ix]['z_sel'].values[0]
        site = sites_params[sites_params['ix']==ix]['site_id'].values[0]
        f_site = [f for f in os.listdir(data_directory) if (f.startswith('FLX')) and (site in f)][0]

        print(site, swc_i, '.......................................................')
        data = make_df(f_site)

        print(site, 'len_data', len(data.index), '.......................................')
        out_file = os.path.join(out_directory, '%s-s%s.csv' % (site, swc_i))
        data.to_csv(out_file)
            


