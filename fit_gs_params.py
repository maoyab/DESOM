import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimize
import pandas as pd
from datetime import datetime
from scipy.stats import pearsonr
from lmfit import Minimizer, Parameters
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing
import random

from data_management import *
from gs_models import *
from et_models import *
from information_metrics import *


def min_model(params, x, obs_var, fit_var_name):
    [vpd_l, co2, gpp, k1, k2, gamma_star, s, psi_pd, psi_l, 
    ga, rad, t_air, vpd_air, p_air, rho_air_dry] = x
    
    if fit_var_name == 'LE_F_MDS':
        g_surf_mod, g_soil_mod, g_canopy_mod, mwue =  cal_g_surf(params, x)
        mod_var = cal_LE(g_surf_mod, ga, rad, vpd_air, p_air, t_air, rho_air_dry)
    elif fit_var_name == 'ET_obs':
        g_surf_mod, g_soil_mod, g_canopy_mod, mwue =  cal_g_surf(params, x)
        mod_var = cal_E(g_surf_mod, ga, vpd_l)
    elif fit_var_name == 'G_surf_mol':
        g_surf_mod, g_soil_mod, g_canopy_mod, mwue =  cal_g_surf(params, x)
        mod_var = g_surf_mod

    return (obs_var - mod_var) ** 2


def cal_g_surf(params, x, leaf=None):
    if leaf is None:
        [vpd_l, co2, gpp, k1, k2, gamma_star, s, psi_pd, psi_l, 
        ga, rad, t_air, vpd_air, p_air, rho_air_dry] = x
    else:
        [vpd_l, co2, gpp, k1, k2, gamma_star, s, psi_pd, psi_l, 
        ga, rad, t_air, vpd_air, p_air, rho_air_dry, lai] = x

    tag = [k for k in params.keys() if k.startswith('tag')][0]
    tag = tag.strip('tag')

    if (tag == '0_d') or tag.startswith('WUE_d') or tag.startswith('E_d'):
        mwue = WUE_mwue(psi_pd, params['lww'].value, params['b'].value)

    elif tag.startswith('WUE_i'):
        mwue = WUE_mwue(psi_l, params['lww'].value, params['b'].value)
    
    elif tag.startswith('CM_d'):
        mwue = CM_mwue(psi_pd, psi_pd, params['b2'].value, params['b1'].value,
                                 params['psi_50'].value, params['a'].value)
    elif tag.startswith('CM_i'):
        mwue = CM_mwue(psi_pd, psi_l, params['b2'].value, params['b1'].value,
                                 params['psi_50'].value, params['a'].value)
    elif tag.startswith('SOX_d') and (leaf is None):
        mwue = SOX_mwue(psi_pd, psi_pd, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value)
    elif tag.startswith('SOX_i') and (leaf is None):
        mwue = SOX_mwue(psi_pd, psi_l, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value)
    elif tag.startswith('SOXa_d') and (leaf is None):
        mwue = aSOX_mwue(psi_pd, psi_pd, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value)
    elif tag.startswith('SOXa_i') and (leaf is None):
        mwue = aSOX_mwue(psi_pd, psi_l, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value)
    elif tag.startswith('SOXait_d'):
        mwue = aSOX_mwue_it(psi_pd, psi_pd, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value, vpd_l, co2, k1, k2, gamma_star)
    elif tag.startswith('SOXait_i'):
        mwue = aSOX_mwue_it(psi_pd, psi_l, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value, vpd_l, co2, k1, k2, gamma_star)
    elif tag.startswith('SOXit_d'):
        mwue = SOX_mwue_it(psi_pd, psi_pd, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value, vpd_l, co2, k1, k2, gamma_star)
    elif tag.startswith('SOXit_i'):
        mwue = SOX_mwue_it(psi_pd, psi_l, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value, vpd_l, co2, k1, k2, gamma_star)
    elif tag.startswith('SOX_d') and (leaf==1):
        mwue = SOX_mwue_it(psi_pd, psi_pd, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value, vpd_l, co2, k1, k2, gamma_star)
    elif tag.startswith('SOX_i') and (leaf==1):
        mwue = SOX_mwue_it(psi_pd, psi_l, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value, vpd_l, co2, k1, k2, gamma_star)
    elif tag.startswith('SOXa_d') and (leaf==1):
        mwue = aSOX_mwue_it(psi_pd, psi_pd, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value, vpd_l, co2, k1, k2, gamma_star)
    elif tag.startswith('SOXa_i') and (leaf==1):
        mwue = aSOX_mwue_it(psi_pd, psi_l, gpp, params['kmax'].value,
                              params['psi_50'].value, params['a'].value, vpd_l, co2, k1, k2, gamma_star)
    if leaf is None:
        g_canopy_mod = gs_opt_canopy(gpp, co2, vpd_l, k2, gamma_star, mwue)
    else:
        gs_mod = gs_opt_leaf(mwue, vpd_l, co2, k1, k2, gamma_star)
        g_canopy_mod = gs_mod * lai

    if tag == 'E_d':
        beta = model_beta(s,  params['s_star'].value,  params['s_w'].value)
        g_canopy_mod = beta * g_canopy_mod

    g_soil_mod = model_g_soil_lin(s, params['gsoil_max'].value)
    
    g_surf_mod = g_soil_mod + g_canopy_mod
    
    return g_surf_mod, g_soil_mod, g_canopy_mod, mwue


def setup_params(data_i, tag):
    g_soil_lim = [10**-3, 0.1] # mol m-2 s-2 MPa-1 
    lww_lims = [0.001*10**-3,  5*10**-3] # mol mol-1 
    b1_lims = [0.001*10**-3, 5*10**-3] # mol mol-1 
    k_lim = [0.001*10**-3, 5*10**-3] # mol m-2 s-2 MPa-1
    
    b2_lims = [0.01*10**-3, 5*10**-3] # mol mol-1 MPa-1
    b_lims = [0.01, 5] # MPa-1
    a_lim = [0.01, 10] # unitless
    p50_lim = [-10, -0.01] # MPa
    
    params = Parameters()
    
    params.add('tag%s' % tag, value=0, vary=False)

    if (tag == '0_d') or tag.endswith('p'):
        params.add('gsoil_max',  value=0.01, min=g_soil_lim[0],  max=g_soil_lim[1], vary=True)
    else:
        params.add('gsoil_max', value=data_i['gsoil_max_0_d'].values[0], vary=False)

    if tag == 'E_d':
        params.add('s_star', value=0.8, min=0.01, max=1, vary=True)
        params.add('dws', value=0.2, min=0.01, max=1, vary=True)
        params.add('s_w', expr='s_star - dws', min=0.01, max=1)
    else:
        params.add('s_star', value=np.nan, vary=False)
        params.add('dws', value=np.nan, vary=False)
        params.add('s_w', value=np.nan, vary=False)

    if (tag == '0_d') or (tag == 'E_d') or tag.startswith('WUE'):
        params.add('b2', value=np.nan, vary=False)
        params.add('b1', value=np.nan, vary=False)
        params.add('a', value=np.nan, vary=False)
        params.add('psi_50', value=np.nan, vary=False)
        params.add('kmax', value=np.nan, vary=False)
    elif tag.startswith('CM'):
        params.add('kmax', value=np.nan, vary=False)
        params.add('lww', value=np.nan, vary=False)
        params.add('b', value=np.nan, vary=False)
    elif tag.startswith('SOX'):
        params.add('b2', value=np.nan, vary=False)
        params.add('b1', value=np.nan, vary=False)
        params.add('lww', value=np.nan, vary=False)
        params.add('b', value=np.nan, vary=False)

    if (tag == '0_d') or (tag == 'E_d'):
        params.add('lww', value=random.uniform(lww_lims[0], lww_lims[1]), min=lww_lims[0], max=lww_lims[1]*10, vary=True)
        params.add('b', value=0, vary=False)
        
    elif tag.startswith('WUE'):
        params.add('lww', value=random.uniform(lww_lims[0], lww_lims[1]), min=lww_lims[0], max=lww_lims[1], vary=True)
        params.add('b', value=random.uniform(b_lims[0], b_lims[1]), min=b_lims[0], max=b_lims[1], vary=True)

    elif tag.startswith('CM'):
        params.add('b2', value=random.uniform(b2_lims[0], b2_lims[1]), min=b2_lims[0], max=b2_lims[1], vary=True)
        params.add('b1', value=random.uniform(b1_lims[0], b1_lims[1]), min=b1_lims[0], max=b1_lims[1], vary=True)
        if tag.endswith('2') or tag.endswith('2_p'):
            params.add('a', value=data_i['a_ref'].values[0], vary=False)
            params.add('psi_50', value=data_i['psi_50_ref'].values[0], vary=False)
        elif tag.endswith('3a') or tag.endswith('3a_p'):
            params.add('a', value=random.uniform(a_lim[0], a_lim[1]), min=a_lim[0], max=a_lim[1], vary=True)
            params.add('psi_50', value=data_i['psi_50_ref'].values[0], vary=False)
        elif tag.endswith('3b') or tag.endswith('3b_p'):
            params.add('a', value=data_i['a_ref'].values[0], vary=False)
            params.add('psi_50', value=random.uniform(p50_lim[0], p50_lim[1]), min=p50_lim[0], max=p50_lim[1], vary=True)
        elif tag.endswith('4') or tag.endswith('4_p'):
            params.add('a', value=random.uniform(a_lim[0], a_lim[1]), min=a_lim[0], max=a_lim[1], vary=True)
            params.add('psi_50', value=random.uniform(p50_lim[0], p50_lim[1]), min=p50_lim[0], max=p50_lim[1], vary=True)
            
    elif tag.startswith('SOX'):
        params.add('kmax',  value=random.uniform(k_lim[0], k_lim[1]), min=k_lim[0], max=k_lim[1], vary=True)
        if tag.endswith('1') or tag.endswith('1_p'):
            params.add('a', value=data_i['a_ref'].values[0], vary=False)
            params.add('psi_50',  value=data_i['psi_50_ref'].values[0], vary=False)
        elif tag.endswith('2a') or tag.endswith('2a_p'):
            params.add('a', value=random.uniform(a_lim[0], a_lim[1]), min=a_lim[0], max=a_lim[1], vary=True)
            params.add('psi_50',  value=data_i['psi_50_ref'].values[0], vary=False)
        elif tag.endswith('2b') or tag.endswith('2b_p'):
            params.add('a', value=data_i['a_ref'].values[0], vary=False)
            params.add('psi_50',  value=random.uniform(p50_lim[0], p50_lim[1]), min=p50_lim[0], max=p50_lim[1], vary=True)
        elif tag.endswith('3') or tag.endswith('3_p'):
            params.add('a', value=random.uniform(a_lim[0], a_lim[1]), min=a_lim[0], max=a_lim[1], vary=True)
            params.add('psi_50',  value=random.uniform(p50_lim[0], p50_lim[1]), min=p50_lim[0], max=p50_lim[1], vary=True)

    return params


def fit_gsurf_params(data_i, tag, min_fit_var='LE_F_MDS'):
    params = setup_params(data_i, tag)
    xx = [data_i['VPD_l'].values, data_i['CO2'].values, data_i['GPP'].values, 
          data_i['k1'].values, data_i['k2'].values, data_i['gamma_star'].values, 
          data_i['S'].values, data_i['canopy_water_pot_pd'].values, data_i['leaf_water_pot'].values,
          data_i['Ga_mol'].values, data_i['Rn-G'].values,
          data_i['TA_F_MDS'].values, data_i['VPD_a'].values, 
          data_i['P_air'].values, data_i['rho_air_d']]
    minner = Minimizer(min_model, params, fcn_args=(xx, data_i[min_fit_var].values, min_fit_var))
    fit = minner.minimize()
    g_surf_mod, g_soil_mod, g_canopy_mod, mwue =cal_g_surf(fit.params, xx)
    return g_surf_mod, g_soil_mod, g_canopy_mod, mwue, fit.params


def fit_gsurf_params_bs(data_i, tag, fitting_results, min_fit_var='LE_F_MDS', bs=200,  nbins=15, norm=1):
    n = np.int(np.float(len(data_i.index)) / 4.)
    fitting_results[tag] = {}
    all_params = []
    all_fit_diagnostics_val = []
    all_fit_diagnostics_cal = []
    m_params = Parameters()
    pnames = ['gsoil_max', 'lww', 'b', 'b2', 'b1', 'a', 'psi_50', 'kmax', 's_star', 's_w']
    for i in range(bs):
        sel_random = random.sample(list(data_i.index), n)
        data_cal = data_i.loc[sel_random]
        data_val = data_i.loc[~data_i.index.isin(sel_random)]
        sel_random_val = random.sample(list(data_val.index), n)
        data_val = data_val.loc[sel_random_val]
        g_surf_mod_cal, g_soil_cal, g_canopy_cal, mwue_cal, params = fit_gsurf_params(data_cal, tag, min_fit_var=min_fit_var)
        all_params.append([params[k].value for k in pnames])
        
        LE_cal, nse_LE_cal, mape_LE_cal = cal_models_LE(data_cal, g_surf_mod_cal)
        data_cal['LE_cal'] = LE_cal

        a_p_cal, a_fu1_cal, a_fu2_cal,\
        a_fs_cal, a_fr_cal, a_ft_cal, a_f_cal = cal_it_performance(data_cal, 'LE_cal', 'LE_F_MDS', 'S', 'VPD_a',  nbins=nbins, norm=norm)

        all_fit_diagnostics_cal.append([nse_LE_cal, mape_LE_cal,
                                        a_p_cal, a_fu1_cal, a_fu2_cal,
                                        a_fs_cal, a_fr_cal, a_ft_cal])

        xx_val = [data_val['VPD_l'].values, data_val['CO2'].values, data_val['GPP'].values, 
                  data_val['k1'].values, data_val['k2'].values, data_val['gamma_star'].values, 
                  data_val['S'].values, data_val['canopy_water_pot_pd'].values, data_val['leaf_water_pot'].values,
                  data_val['Ga_mol'].values, data_val['Rn-G'].values,
                  data_val['TA_F_MDS'].values, data_val['VPD_a'].values, 
                  data_val['P_air'].values, data_val['rho_air_d']]

        g_surf_mod_val, g_soil_mod_val, g_canopy_mod_val, mwue_val =cal_g_surf(params, xx_val)

        LE_val, nse_LE_val, mape_LE_val = cal_models_LE(data_val, g_surf_mod_val)
        data_val['LE_val'] = LE_val
        
        a_p_val, a_fu1_val, a_fu2_val,\
        a_fs_val, a_fr_val, a_ft_val, a_f_val = cal_it_performance(data_val, 'LE_val', 'LE_F_MDS', 'S', 'VPD_a', nbins=nbins, norm=norm)

        all_fit_diagnostics_val.append([nse_LE_val, mape_LE_val,
                                        a_p_val, a_fu1_val, a_fu2_val,
                                        a_fs_val, a_fr_val, a_ft_val])
    
    all_fit_diagnostics_val =  list(zip(*all_fit_diagnostics_val))
    fitting_results[tag]['nse_LE_val'] = all_fit_diagnostics_val[0]
    fitting_results[tag]['mape_LE_val'] = all_fit_diagnostics_val[1]
    fitting_results[tag]['a_p_val'] = all_fit_diagnostics_val[2]
    fitting_results[tag]['a_fu1_val'] = all_fit_diagnostics_val[3]
    fitting_results[tag]['a_fu2_val'] = all_fit_diagnostics_val[4]
    fitting_results[tag]['a_fs_val'] = all_fit_diagnostics_val[5]
    fitting_results[tag]['a_fr_val'] = all_fit_diagnostics_val[6]
    fitting_results[tag]['a_ft_val'] = all_fit_diagnostics_val[7]

    all_fit_diagnostics_cal =  list(zip(*all_fit_diagnostics_cal))
    fitting_results[tag]['nse_LE_cal'] = all_fit_diagnostics_cal[0]
    fitting_results[tag]['mape_LE_cal'] = all_fit_diagnostics_cal[1]
    fitting_results[tag]['a_p_cal'] = all_fit_diagnostics_cal[2]
    fitting_results[tag]['a_fu1_cal'] = all_fit_diagnostics_cal[3]
    fitting_results[tag]['a_fu2_cal'] = all_fit_diagnostics_cal[4]
    fitting_results[tag]['a_fs_cal'] = all_fit_diagnostics_cal[5]
    fitting_results[tag]['a_fr_cal'] = all_fit_diagnostics_cal[6]
    fitting_results[tag]['a_ft_cal'] = all_fit_diagnostics_cal[7]

    all_params = list(zip(*all_params))
    m_params.add('tag%s' % tag, value=0, vary=False)
    for k, param in zip(pnames, all_params):
        m_params.add(k, value = np.nanmedian(param), vary=False)
        fitting_results[tag][k] = param

    xx = [data_i['VPD_l'].values, data_i['CO2'].values, data_i['GPP'].values, 
          data_i['k1'].values, data_i['k2'].values, data_i['gamma_star'].values, 
          data_i['S'].values, data_i['canopy_water_pot_pd'].values, data_i['leaf_water_pot'].values,
          data_i['Ga_mol'].values, data_i['Rn-G'].values,
          data_i['TA_F_MDS'].values, data_i['VPD_a'].values, 
          data_i['P_air'].values, data_i['rho_air_d']]

    g_surf_mod, g_soil_mod, g_canopy_mod, mwue = cal_g_surf(m_params, xx)

    return g_surf_mod, g_soil_mod, g_canopy_mod, mwue, m_params, fitting_results


def cal_models_LE(data_i, g_surf_mol_mod):
    G_surf_mod = g_from_mol(g_surf_mol_mod,  data_i['P_air'], data_i['TA_F_MDS'])
    LE = cal_LE(g_surf_mol_mod, data_i['Ga_mol'], data_i['Rn-G'],  \
                data_i['VPD_a'], data_i['P_air'], data_i['TA_F_MDS'], data_i['rho_air_d'])
    try:
        nse_LE = cal_nse(data_i['LE_F_MDS'], LE)
        mape_LE = cal_mape(data_i['LE_F_MDS'], LE)
    except:
        xx, yy = list(zip(*[[xi, yi] for xi, yi in zip(data_i['LE_F_MDS'], LE) if (np.isnan(yi)==0)]))
        nse_LE = cal_nse(xx, yy)
        mape_LE = cal_mape(xx, yy)
    return LE, nse_LE, mape_LE


def process_model_alternative(data_i, tag): 
    g_surf_mod, g_soil_mod, g_canopy_mod, mwue, params = fit_gsurf_params(data_i, tag)
    
    for p in ['gsoil_max','lww', 'b', 'b1', 'b2', 'a', 'psi_50', 'kmax', 's_star', 's_w']:
        data_i['%s_%s' % (p, tag)] = params[p].value

    data_i['g_soil_%s' % tag] = g_soil_mod
    data_i['g_canopy_%s' % tag] = g_canopy_mod
    data_i['g_surf_mod_%s' % tag] = g_surf_mod
    data_i['E/ET %s' % tag] = g_soil_mod / g_surf_mod
    data_i['mwue_%s' % tag] = mwue

    gs_leaf = gs_opt_leaf(mwue, data_i['VPD_l'], data_i['CO2'], data_i['k1_0'], data_i['k2_0'], data_i['gamma_star_0'])
    
    data_i['g_canopy_leaf_%s' % tag] = gs_leaf * data_i['LAI']
    data_i['g_leaf_%s' % tag] = gs_leaf
    data_i['eLAI_%s' % tag] = data_i['g_canopy_%s' % tag] / data_i['g_leaf_%s' % tag]

    LE, nse_LE, mape_LE = cal_models_LE(data_i, g_surf_mod)
    
    data_i['LE_%s' % tag] = LE
    data_i['nse_LE_%s' % tag] = nse_LE
    data_i['mape_LE_%s' % tag] = mape_LE

    print('%s\t\t%-3.2f\t%3.2f\t\t%3.2f\t%-3.2f\t%-3.2f\t%-3.2f\t%-3.2f\t%-3.2f\t%-3.2f\t%-3.2f\t%-3.2f'% 
        (tag, np.nanmedian(data_i['eLAI_%s' % tag]),
        np.nanmedian(data_i['mwue_%s' % tag]*10**3), 
        data_i['lww_%s' % tag].values[0]*10**3, data_i['b_%s' % tag].values[0],
        data_i['b2_%s' % tag].values[0]*10**3, data_i['b1_%s' % tag].values[0]*10**3,
        data_i['kmax_%s' % tag].values[0]*10**3,
        data_i['a_%s' % tag].values[0], data_i['psi_50_%s' % tag].values[0],
        nse_LE, mape_LE))
    return data_i
    

def process_model_alternative_bs(data_i, tag, fitting_results,  nbins=15, norm=1):
    d0 = datetime.now()
    print(tag, datetime.now())
    g_surf_mod, g_soil_mod, g_canopy_mod, mwue,\
        params, fitting_results = fit_gsurf_params_bs(data_i, tag, fitting_results)

    for p in ['gsoil_max','lww', 'b', 'b1', 'b2', 'a', 'psi_50', 'kmax', 's_star', 's_w']:
        fitting_results[tag]['%s_med' % p] = params[p].value
    
    data_i['g_soil_%s' % tag] = g_soil_mod
    data_i['g_canopy_%s' % tag] = g_canopy_mod
    data_i['g_surf_mod_%s' % tag] = g_surf_mod
    data_i['E/ET %s' % tag] = g_soil_mod / g_surf_mod
    data_i['mwue_%s' % tag] = mwue

    gs_leaf = gs_opt_leaf(mwue, data_i['VPD_l'], data_i['CO2'],
                            data_i['k1_0'], data_i['k2_0'],
                            data_i['gamma_star_0'])
    
    data_i['g_canopy_leaf_%s' % tag] = gs_leaf * data_i['LAI']
    data_i['g_leaf_%s' % tag] = gs_leaf
    data_i['eLAI_%s' % tag] = data_i['g_canopy_%s' % tag] / data_i['g_leaf_%s' % tag]

    LE, nse_LE, mape_LE = cal_models_LE(data_i, g_surf_mod)
    data_i['LE_%s' % tag] = LE

    fitting_results[tag]['nse_LE_med'] = nse_LE
    fitting_results[tag]['mape_LE_med'] = mape_LE

    a_p_med, a_fu1_med,\
    a_fu2_med, a_fs_med,\
    a_fr_med, a_ft_med, a_f_med = cal_it_performance(data_i, 
                                            'LE_%s' % tag, 'LE_F_MDS', 'S', 'VPD_a', 
                                             nbins=nbins, norm=norm)
    fitting_results[tag]['a_p_med'] = a_p_med
    fitting_results[tag]['a_fu1_med'] = a_fu1_med
    fitting_results[tag]['a_fu2_med'] = a_fu2_med
    fitting_results[tag]['a_fs_med'] = a_fs_med
    fitting_results[tag]['a_fr_med'] = a_fr_med
    fitting_results[tag]['a_ft_med'] = a_ft_med

    print(tag, datetime.now() - d0)
    return data_i, fitting_results


def fit_nn(data_i, y_var='LE_F_MDS',  x_vars = ['VPD_a', 'SWC_F_MDS', 'CO2', 'GPP', 'PARin', 'TA_F_MDS', 'Ga', 'Rn-G']):
    x_scaled = preprocessing.scale(data_i[x_vars].values)
    nn = MLPRegressor(solver='lbfgs')
        
    nn.fit(x_scaled, data_i[y_var].values)
    y_ann = nn.predict(x_scaled)

    r_nn = pearsonr(y_ann, data_i[y_var].values)[0]
    nse_nn = cal_nse(y_ann, data_i[y_var].values)
    mape_nn = cal_mape(y_ann, data_i[y_var].values)

    return y_ann, r_nn, nse_nn, mape_nn


def fit_plots_checks():
    out_figures_directory = '../../PROJECTS/Stomatal_conductance_eval/result_figures/lmfit_test_plots'
    sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')
    pft_params = pd.read_csv( '../../DATA/EEGS/selected_pft_params.csv')
 
    fig_suffix =  '_test'

    data_directory = '../../DATA/EEGS/HH_data_revision_w5'

    sites_files = [os.path.join(data_directory, f) 
                    for f in os.listdir(data_directory)]

    sites_sel = sites_params['site_id'].values

    for f_site in sites_files:
        
        site_name = f_site.split('/')[-1][:-4]
        site = site_name[:-3]
        swc_i = np.int(site_name[-1])
        sites_params_i = sites_params[(sites_params['site_id'] == site) & (sites_params['z_sel'] == swc_i)]
        pft = sites_params_i['pft'].values[0]
        pft_params_i = pft_params[pft_params['PFT'] == pft]
        zm = sites_params_i['zs'].values[0]
        fig_name = os.path.join(out_figures_directory, '%s_fit%s.png' % (site_name, fig_suffix))

        data = pd.read_csv(f_site, header = 0, index_col = 0, parse_dates = True, 
                      infer_datetime_format = True,)
        print(site_name, pft, zm, np.max(data.index.year) - np.min(data.index.year) + 1)
        
        data['k2'] = data['k2_Tl']  # change for Tl or 0
        data['k1'] = data['k1_Tl']  # change for Tl or 0
        data['gamma_star'] =  data['gamma_star_Tl']  # change for Tl or 0
        data['leaf_water_pot'] = data['leaf_water_pot_invET']  #  placeholder until invT

        LE_ann, r_nn, nse_LE_nn, mape_LE_nn = fit_nn(data, y_var='LE_F_MDS')
        data['LE_NN'] = LE_ann
        data['nse_LE_NN'] = nse_LE_nn
        data['mape_LE_NN'] = mape_LE_nn
        
        model_alternatives_d_p = ['WUE_d_p',
                                  'CM_d4_p',
                                  'SOX_d3_p'] 
        model_alternatives_d = [
                                'WUE_d', 'WUE_i',
                                'CM_d4', 'CM_i4',
                                'SOX_d3', 'SOX_i3',
                                ] 

        model_alternatives_plot = [ 
                                    'WUE_d', 'WUE_i',
                                    'CM_d4', 'CM_i4',
                                    'SOX_d3', 'SOX_i3',
                                    ] 

        print('tag\t\tlc\tmwue\t\tlww\tb\tb2\tb1\tKmax\ta\tp50\tNSE_LE\tMAPE_LE')

        for tag in model_alternatives_d_p:
            data = process_model_alternative(data, tag)
        avg_gsoil_max = np.sum([data['gsoil_max_WUE_d_p'][0],
                                data['gsoil_max_CM_d4_p'][0], 
                                data['gsoil_max_SOX_d3_p'][0]]) / 3.
        
        data['gsoil_max_0_d'] = avg_gsoil_max
        data['g_soil_0_d'] = data['S'] * avg_gsoil_max
        data['T'] = data['ET_obs'] * (1 - data['g_soil_0_d'] / data['G_surf_mol'])
        data['leaf_water_pot'] = cal_psi_leaf(data['canopy_water_pot_pd'], 
                                                pft_params_i['psi_50'].values[0],
                                                pft_params_i['kmax'].values[0] * np.max(data['LAI']),
                                                pft_params_i['a'].values[0],
                                                data['T'])
        print(np.nanmin(data['leaf_water_pot']), np.nanmax(data['leaf_water_pot']))
        for tag in model_alternatives_d:
            data = process_model_alternative(data, tag)
         
        
        fig = plt.figure(figsize=(15, 10))

        data['G_surf_n'] = data['G_surf_mol'] / np.nanmax(data['G_surf_mol'])
        data['VPD_n'] = data['VPD_l'] / np.nanmax(data['VPD_l'])
        
        n_vars = ['VPD_n', 'G_surf_n', 'S', 'LAI', 'G_surf_mol', 'g_soil_0_d', 'soil_water_pot', 'leaf_water_pot_invET', 'leaf_water_pot_invET1', 'leaf_water_pot']
        for tag in model_alternatives_plot:
            n_vars.append('eLAI_%s' % tag)
        data_d = data[n_vars].resample('D').mean().dropna()
        
        ax = fig.add_subplot(4, 2, 1)
        
        data_d[['VPD_n', 'G_surf_n', 'S']].plot(ax=ax, lw=0, marker='.', legend=False)
        ax.set_ylim([0, 1])
        title = '%s; %s; [%sd; %shh]; %sz; %5.1fLAI ; %5.2fpsmin' % (site, pft, len(data_d.index), 
                                                                                  len(data.index), zm, np.nanmax(data['LAI']), 
                                                                                  np.nanmin(data['soil_water_pot']))
        plt.legend()
        plt.title(title)

        ax = fig.add_subplot(4, 2, 3)
        data_d['soil_water_pot_'] = -data_d['soil_water_pot']
        data_d['leaf_water_pot_0'] = -data_d['leaf_water_pot_invET']
        data_d['leaf_water_pot_'] = -data_d['leaf_water_pot']
        data_d[['soil_water_pot_',  'leaf_water_pot_0',  'leaf_water_pot_']].plot(ax=ax, lw=0, marker='.', legend=False)
        ax.axhline(1, color='k', linestyle=':')
        ax.set_yscale('log')
        plt.legend()

        ax = fig.add_subplot(4, 2, 5)
        elai_l = []
        for tag in model_alternatives_plot:
            elai_l.append('eLAI_%s' % tag)
        elai_l.append('LAI')
        data_d[elai_l].plot(ax=ax, lw=0, marker='.', legend=False)
        ax.set_ylim([0, np.nanmax(data_d['LAI'])*1.1])
        
        ax = fig.add_subplot(4, 2, 7)
        E_r_T = 'T/ET%-5.1f' % (np.round(np.nanmedian(data['g_soil_0_d'] / data['G_surf_mol']) * 100., 1))
        data_d[['G_surf_mol', 'g_soil_0_d']].plot(ax=ax, lw=0, marker='.', legend=False)
        plt.legend()
        ax.set_ylabel(E_r_T)
        print(title, E_r_T)

        ax = fig.add_subplot(1, 2, 2)
        max_v = np.nanmax([data['LE_F_MDS'] * 1.2])
        for tag in model_alternatives_plot:
            ax.scatter(data['LE_%s' % tag], data['LE_F_MDS'], alpha=0.1,
                     label= '%s: %-5.2f' % (tag, data['nse_LE_%s' % tag].values[0]))
        plt.title('%s / gsoilmax: %-5.4f %-5.4f / lww: %-5.2f %-5.2f / b: %-5.2f %-5.2f' % (
                                            np.max(data.index.year) - np.min(data.index.year) + 1,
                                            data['gsoil_max_WUE_d_p'].values[0], data['gsoil_max_0_d'].values[0],
                                            data['lww_WUE_d_p'].values[0]*10**3, data['lww_WUE_d'].values[0]*10**3,
                                            data['b_WUE_d_p'].values[0], data['b_WUE_d'].values[0]))

        plt.legend()
        ax.set_ylim([0, max_v])
        ax.set_xlim([0, max_v])
        ax.set_xlabel('LE mod')
        ax.set_ylabel('LE obs')

        plt.savefig(fig_name)
        
        print(' ')
        plt.close()


if __name__ == "__main__":

    
    sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')
    pft_params = pd.read_csv( '../../DATA/EEGS/selected_pft_params.csv')
    out_data_directory = '../../DATA/EEGS/fitted_models_revision_0'
 
    fig_suffix =  ''

    data_directory = '../../DATA/EEGS/HH_data_revision_w5'

    sites_files = [os.path.join(data_directory, f) 
                    for f in os.listdir(data_directory)]

    sites_sel = sites_params['site_id'].values
    print(len(sites_files))
    for f_site in sites_files:

        site_name = f_site.split('/')[-1][:-4]
        site = site_name[:-3]
        sites_params_i = sites_params[(sites_params['site_id'] == site)]
        pft = sites_params_i['pft'].values[0]
        pft_params_i = pft_params[pft_params['PFT'] == pft]
        data = pd.read_csv(f_site, header = 0, index_col = 0, parse_dates = True, 
                      infer_datetime_format = True,)
        print(site_name)
        
        fitting_results = {}

        out_file_m = os.path.join(out_data_directory, '%s_fitted_models_bs.csv' % (site_name))
        out_file_r = os.path.join(out_data_directory, '%s_fitting_results_bs.pickle' % (site_name))

        data['k2'] = data['k2_0'] # change for Tl or 0
        data['k1'] = data['k1_0'] # change for Tl or 0
        data['gamma_star'] =  data['gamma_star_0'] # change for Tl or 0
        data['leaf_water_pot'] = data['leaf_water_pot_invET'] # placeholder until invT

        nn_x_vars = ['VPD_a', 'SWC_F_MDS', 'CO2', 'GPP', 'PARin', 'TA_F_MDS', 'Ga', 'Rn-G']

        LE_ann, r_nn, nse_LE_nn, mape_LE_nn = fit_nn(data, y_var='LE_F_MDS', x_vars=nn_x_vars)
        data['LE_NN'] = LE_ann
        data['nse_LE_NN'] = nse_LE_nn
        data['mape_LE_NN'] = mape_LE_nn
        fitting_results['LE_NN'] = {'NSE': nse_LE_nn, 'mape': mape_LE_nn}

        model_alternatives_d_p = ['WUE_d_p',
                                 'CM_d4_p',
                                 'SOX_d3_p'] 

        model_alternatives_d = ['WUE_d', 'WUE_i', 
                                'CM_d2',
                                'CM_d4', 'CM_i4',
                                'SOX_d1', 
                                'SOX_d3', 'SOX_i3']

        for tag in model_alternatives_d_p:
            data = process_model_alternative(data, tag)
        avg_gsoil_max = np.sum([data['gsoil_max_WUE_d_p'][0],
                                data['gsoil_max_CM_d4_p'][0],
                                data['gsoil_max_SOX_d3_p'][0]]) / 3.
        
        data['gsoil_max_0_d'] = avg_gsoil_max
        data['g_soil_0_d'] = data['S'] * avg_gsoil_max
        data['T'] = data['ET_obs'] * (1 - data['g_soil_0_d'] / data['G_surf_mol'])
        data['leaf_water_pot'] = cal_psi_leaf(data['canopy_water_pot_pd'], 
                                                pft_params_i['psi_50'].values[0],
                                                pft_params_i['kmax'].values[0] * np.max(data['LAI']),
                                                pft_params_i['a'].values[0],
                                                data['T'])

        drop_v = [k for k in data.keys() if k.endswith('_p')==0]
        data = data[drop_v]
        for tag in model_alternatives_d:
            data, fitting_results = process_model_alternative_bs(data, tag, fitting_results)

        with open(out_file_r, 'wb') as f:
            dump(fitting_results, f)
        data.to_csv(out_file_m)
        print(site_name, 'done')

        print(' ')
                