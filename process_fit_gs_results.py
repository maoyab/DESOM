import os
import sys
import pandas as pd
from pickle import dump, load
import numpy as np
from gs_models import *
from fit_gs_params import *
from lmfit import Parameters

sel = 'revision_0'

res_dir = '../../DATA/EEGS/fitted_models_%s' % sel
res_dir_2 = '../../DATA/EEGS/results_params'

fig_dir = '../../PROJECTS/Stomatal_conductance_eval/results_figures/2_param_uncertainty'
sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')

params_by_models = [['WUE_d', ['lww', 'b']],
                    ['CM_d2', ['b1', 'b2']],
                    ['CM_d4', ['b1', 'b2', 'a', 'psi_50']],
                    ['SOX_d1', ['kmax', ]],
                    ['SOX_d3', ['kmax', 'a', 'psi_50']],
                    ['WUE_i', ['lww', 'b']],
                    ['CM_i4', ['b1', 'b2', 'a', 'psi_50']],
                    ['SOX_i3', ['kmax', 'a', 'psi_50']],
                    ]

mod_list = ['WUE_d', 'CM_d2', 'CM_d4', 'SOX_d1', 'SOX_d3', 'WUE_i', 'CM_i4', 'SOX_i3']

pft_list = ['NF', 'BF', 'G-C3', 'C-C3']
sites = sites_params['site_id'].values

def by_site_params(f_name):
    out_file_m = os.path.join(res_dir_2, f_name)
    all_params = [] 
    for site_i in sites:
        print (site_i)
        site_p = []
        pft = sites_params[sites_params['site_id'] == site_i]['pft'].values[0]
        pickle_file = [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                    if fi.endswith('pickle') 
                                                    and (site_i in fi)][0]
        data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                    if fi.endswith('bs.csv')
                                                    and (site_i in fi)][0]
        with open(pickle_file, 'rb') as fp:
            df = load(fp)
        
        data_i = pd.read_csv(data_file,
                        header=0, index_col=0,
                        parse_dates=True,
                        infer_datetime_format=True)
        
        psi_min = np.nanmin(data_i['soil_water_pot'])
        psi_5 = np.percentile(data_i['soil_water_pot'], 5)
        l_hh = len(data_i.index)
        l_dd = len(data_i.resample('D').mean()['S'].dropna().index)
        n_y = len(list(set(data_i.index.year)))
        
        site_p = [site_i, pft, psi_min, psi_5, l_hh, l_dd, n_y]
        keys = ['site_id', 'pft', 'psi_min', 'psi_5', 'l_hh', 'l_dd', 'n_y']
        for m, param_list_m in params_by_models:
            site_p.append(df[m]['a_p_med'])
            site_p.append(np.abs(df[m]['a_fu1_med'])
                          + np.abs(df[m]['a_fu2_med'])
                          + np.abs(df[m]['a_fr_med'])
                          + np.abs(df[m]['a_fs_med']))
            keys.append('Ap_%s' % m)
            keys.append('Af_%s' % m)
            for p in param_list_m:
                site_p.append(np.nanmedian(df[m][p]))
                keys.append('%s_%s' % (p, m))
                site_p.append(np.nanstd(df[m][p]))
                keys.append('%s_%s_std' % (p, m))
        all_params.append(site_p)
    
    all_params = zip(*all_params)
    df = {}
    for k, pi in zip(keys, all_params):
        df[k] = pi
    
    all_params = pd.DataFrame(df)
    all_params.to_csv(out_file_m)


def gen_loso_params(f_name):
    param_results = []
    out_file_m = os.path.join(res_dir_2, f_name)

    for mod, param_list_m in params_by_models:
        for p in param_list_m:
            for site in sites:
                pft = sites_params[(sites_params['site_id'] == site)]['pft'].values[0]
                print(site, mod, p, pft)
                pickle_file = [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                            if fi.endswith('pickle') and (site in fi)][0]
                with open(pickle_file, 'rb') as pf:
                    df = load(pf)
                x_site = np.array(df[mod][p])
                
                sites_re = sites_params[(sites_params['pft'] == pft) 
                                & (sites_params['site_id'] != site)]['site_id'].values
                x = np.array([])
                for site_i in sites_re:
                    pickle_file = [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                            if fi.endswith('pickle') and (site_i in fi)][0]
                    with open(pickle_file, 'rb') as pf:
                        df = load(pf)
                    xx = np.array(df[mod][p])
                    x = np.concatenate((x, xx), axis=None)
                
                param_results.append([site, mod, p, pft,
                                    np.nanmean(x), np.nanmedian(x), np.nanstd(x),
                                    np.nanmax(x), np.nanmin(x),
                                    np.percentile(x, 75), np.percentile(x, 25),
                                    np.nanmedian(x_site)])
  
    site, mod, p, pft, p_mean, p_median, p_std, p_max, p_min, p_p75, p_p25, s_median = zip(*param_results)

    gen_pft_params = pd.DataFrame({'site_id': site, 'model': mod, 'parameter': p, 'pft': pft,
                                'mean': p_mean, 'median': p_median, 'std': p_std,
                                'max': p_max,  'min': p_min, 
                                'p75': p_p75,  'p25': p_p25, 
                                'site_median': s_median})

    gen_pft_params.to_csv(out_file_m)


def overall_pft_params(f_name):
    param_results = []
    out_file_m = os.path.join(res_dir_2, f_name) 
    for mod, param_list_m in params_by_models:
        for p in param_list_m:
            for pft in pft_list:
                print(mod, p, pft)
                
                x = np.array([])
                sites_pft = sites_params[(sites_params['pft']==pft)]['site_id'].values
                for site_i in sites_pft:
                    pickle_file = [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                            if fi.endswith('pickle') and (site_i in fi)][0]
                    with open(pickle_file, 'rb') as pf:
                        df = load(pf)

                    xx = np.array(df[mod][p])
                    x = np.concatenate((x, xx), axis=None)

                param_results.append([mod, p, pft, np.nanmean(x), np.nanmedian(x), np.nanstd(x),
                                    np.nanmax(x), np.nanmin(x), np.percentile(x, 75), np.percentile(x, 25)])
      
    
    mod, p, pft, p_mean, p_median, p_std, p_max, p_min, p_p75, p_25 = zip(*param_results)

    pft_params = pd.DataFrame({'model': mod, 'parameter': p,
                                'pft': pft, 'mean': p_mean, 'median': p_median,
                                'std': p_std, 'max': p_max, 'min': p_min, 
                                'p75': p_p75, 'p25': p_25})
    
    pft_params.to_csv(out_file_m)


def cal_gen_model(f_name_i):
    sites_files = [os.path.join(res_dir, f) 
                    for f in os.listdir(res_dir) 
                        if f.endswith('bs.csv')]
    
    gen_params_0 =  pd.read_csv(os.path.join(res_dir_2, 'gen_params_table_%s.csv' % sel))
    
    for site_i in sites:
        pft_i = sites_params[sites_params['site_id'] == site_i]['pft'].values[0]
        
        gen_params = gen_params_0[gen_params_0['site_id']==site_i]
        pickle_file = [os.path.join(res_dir, fi) 
                            for fi in os.listdir(res_dir) 
                                if fi.endswith('pickle') and (site_i in fi)][0]
        data_file =  [os.path.join(res_dir, fi) 
                            for fi in os.listdir(res_dir) 
                            if fi.endswith('bs.csv') and (site_i in fi)][0]
        
        data_i = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                          infer_datetime_format = True)
        with open(pickle_file, 'rb') as f:
            splitfit_results = load(f) 
        
        xx = [data_i['VPD_l'].values, data_i['CO2'].values, data_i['GPP'].values, 
              data_i['k1'].values, data_i['k2'].values, data_i['gamma_star'].values, 
              data_i['S'].values, data_i['canopy_water_pot_pd'].values, data_i['leaf_water_pot'].values,
              data_i['Ga_mol'].values, data_i['Rn-G'].values,
              data_i['TA_F_MDS'].values, data_i['VPD_a'].values, 
              data_i['P_air'].values, data_i['rho_air_d']]

        xx_l = [data_i['VPD_l'].values, data_i['CO2'].values, data_i['GPP'].values, 
              data_i['k1'].values, data_i['k2'].values, data_i['gamma_star'].values, 
              data_i['S'].values, data_i['canopy_water_pot_pd'].values, data_i['leaf_water_pot'].values,
              data_i['Ga_mol'].values, data_i['Rn-G'].values,
              data_i['TA_F_MDS'].values, data_i['VPD_a'].values, 
              data_i['P_air'].values, data_i['rho_air_d'], data_i['LAI']]

        for tag in mod_list:
            if tag.startswith('WUE'):
                pp_m = ['lww', 'b']

            elif tag.startswith('CM'):
                pp_m = ['b2', 'b1', 'a', 'psi_50']
                
            elif tag.startswith('SOX'):
                pp_m = ['kmax', 'a', 'psi_50']

            g_params = Parameters()
            g_params.add('tag%s' % tag, value=0, vary=False)
            g_params.add('gsoil_max', value= data_i['gsoil_max_0_d'].values[0], vary=False)

            m_params = Parameters()
            m_params.add('tag%s' % tag, value=0, vary=False)
            m_params.add('gsoil_max', value= data_i['gsoil_max_0_d'].values[0], vary=False)

            for pi in ['lww', 'b', 'b2', 'b1', 'a', 'psi_50', 'kmax']:
                if pi in pp_m:
                    if (pi in ['a', 'psi_50']) and (tag in ['SOXa_d1', 'SOX_d1', 'SOXait_d1', 
                                                            'SOXit_d1',  'SOXa_i1', 'SOX_i1', 
                                                            'CM_d2', 'CM_i2']):
                        pi_g = data_i['%s_ref' % pi].values[0]
                        pi_m = data_i['%s_ref' % pi].values[0]
                    else:
                        print(tag, pi, pft_i)
                        pi_g = gen_params[(gen_params['model'] == tag) 
                                      & (gen_params['parameter'] == pi)]['median'].values[0]
                        pi_m = np.median(splitfit_results[tag][pi])
                    m_params.add(pi, value=pi_m, vary=False)
                    g_params.add(pi, value=pi_g, vary=False)
                else:
                    g_params.add(pi, value=np.nan, vary=False)
                    m_params.add(pi, value=np.nan, vary=False)

            g_surf_mod, g_soil_mod, g_canopy_mod, mwue =cal_g_surf(g_params, xx)
            LE, nse_LE, mape_LE = cal_models_LE(data_i, g_surf_mod)
            data_i['LE_%s_gen' % tag] = LE
            g_surf_mod_, g_soil_mod_, g_canopy_mod_, mwue_ = cal_g_surf(g_params, xx_l, leaf=1)
            LE, nse_LE, mape_LE = cal_models_LE(data_i, g_surf_mod_)
            data_i['LE_LAI_%s_gen' % tag] = LE
            ci = opt_ci(mwue_, data_i['VPD_l'].values, data_i['CO2'].values,
                        data_i['k1'].values, data_i['k2'].values, data_i['gamma_star'].values)
            data_i['GPP_LAI_%s_gen' % tag] = 1 / 1.6 * g_canopy_mod_ * (data_i['CO2'].values - ci)

            g_surf_mod_, g_soil_mod_, g_canopy_mod_, mwue_ = cal_g_surf(m_params, xx_l, leaf=1)
            LE, nse_LE, mape_LE = cal_models_LE(data_i, g_surf_mod_)
            data_i['LE_LAI_%s' % tag] = LE
            ci = opt_ci(mwue_, data_i['VPD_l'].values, data_i['CO2'].values,
                        data_i['k1'].values, data_i['k2'].values, data_i['gamma_star'].values)
            data_i['GPP_LAI_%s' % tag] = 1 / 1.6 * g_canopy_mod_ * (data_i['CO2'].values - ci)

        out_file_m = os.path.join(res_dir, '%s_%s' % (site_i, f_name_i))
    
        data_i.to_csv(out_file_m)
        print(site_i, 'done')


if __name__ == "__main__":
    
    by_site_params('param_results_allsites_%s.csv' % sel)
    gen_loso_params('gen_params_table_%s.csv' % sel)
    overall_pft_params('pft_params_table_%s.csv' % sel)
    cal_gen_model('fitted_models_LE_gen.csv')
