import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from information_metrics import *
from pickle import dump, load
from scipy.stats import spearmanr

sel = 'revision_0'
res_dir = '../../DATA/EEGS/fitted_models_%s' % sel
fig_dir = '../../PROJECTS/Stomatal_conductance_eval/NP_figures'
sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')
sel_sites = sites_params['site_id'].values 

n_bins = 15
source_1 = 'S'
source_2 = 'VPD_a'
sensitivity_var = source_1
target_m = 'LE'
target_o = 'LE_F_MDS'

pft_list = [[3, 'Needleleaf', ['NF', ]],
                [4, 'Broadleaf', ['BF', ]],
                [7, 'Grass', ['G-C3', ]],
                [8, 'Crop', ['C-C3', ]]]


def data2rank(data):
    data_rank = []
    for x in data:
        x_s = list(np.sort(x))
        data_rank.append([1 - x_s.index(xi) / np.float(len(x) - 1) for xi in x])
    data_rank = list(zip(*data_rank))
    return [np.mean(r) for r in data_rank]


def get_diagnostics(pft, dic_mod):
    n_params = [len(mod_p) for mod_p in dic_mod['mod_params']]

    result = {'n_params': n_params, 'models': dic_mod['models'],
              'delta_nse': [], 'delta_mape': [],
              'param_cv': [], 'param_equif': [], 'param_entropy': [],
              'full_rec': {},
              'dry': {},
              'med': {},
              'wet': {},
              }

    for k in ['aic', 'a_p', 'a_ft', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_fp']:
        result['full_rec'][k] = []
        result['dry'][k] = []
        result['med'][k] = []
        result['wet'][k] = []

    for k in ['d_a_p', 'd_a_ft', 'd_a_fu1', 'd_a_fu2', 'd_a_fs', 'd_a_fr', 'd_a_fp']:
        result['full_rec'][k] = []
        result['dry'][k] = []
        result['med'][k] = []
        result['wet'][k] = []

    for site_i in sel_sites:
        pickle_file = [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                        if fi.endswith('pickle') and (site_i in fi)][0]
        data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                        if fi.endswith('gen.csv') and (site_i in fi)][0]
        pft_i = sites_params[sites_params['site_id'] == site_i]['pft'].values[0]

        if pft_i in pft:
            data_0 = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                              infer_datetime_format = True,)
            with open(pickle_file, 'rb') as f:
                df = load(f) 
            
            vv = [source_1, source_2, target_o, ]
            for mi in dic_mod['models']:
                vv.append('%s_%s' % (target_m, mi))
                vv.append('%s_%s_gen' % (target_m, mi))
            data_0 = data_0[vv].dropna()

            p25 = np.percentile(data_0[sensitivity_var].values, 25)
            p75 = np.percentile(data_0[sensitivity_var].values, 75)

            data_dry = data_0[data_0[sensitivity_var] <= p25]
            data_med = data_0[(data_0[sensitivity_var] < p75) & (data_0[sensitivity_var] > p25)]
            data_wet = data_0[data_0[sensitivity_var] >= p75]
            
            site_crit = []
            for mi in dic_mod['models']:
                val = 1 - np.array(df[mi]['nse_LE_val'])
                cal = 1 - np.array(df[mi]['nse_LE_cal'])
                delta_nse_i = (val-cal)/cal
                site_crit.append(np.nanmean(delta_nse_i))
            result['delta_nse'].append(site_crit)
            
            site_crit = []
            for mi in dic_mod['models']:
                val = np.array(df[mi]['mape_LE_val'])
                cal = np.array(df[mi]['mape_LE_cal'])
                delta_mape_i = (val-cal)/cal
                site_crit.append(np.nanmean(delta_mape_i))
            result['delta_mape'].append(site_crit)

            site_crit = []
            for mi, mod_p in zip(dic_mod['models'], dic_mod['mod_params']):
                param_cv_i = [np.std(df[mi][p]) / np.nanmean(df[mi][p]) 
                                    for p in mod_p]
                site_crit.append(np.nanmean(param_cv_i))
            result['param_cv'].append(site_crit)

            site_crit = []
            for mi, mod_p in zip(dic_mod['models'], dic_mod['mod_params']):
                h = []
                for p in mod_p:
                    if p in ['lww', 'b1', 'kmax']:
                        bins = np.arange(10 ** -6, 5 * 10 ** -3, 0.05/1000.)
                    elif p in [ 'b2', ]:
                        bins = np.arange(0.01/1000., 5/1000., 0.05/1000.)
                    elif p =='b':
                        bins = np.arange(0.01, 5, 0.05)
                    elif p == 'a':
                        bins = np.arange(0.01, 10, 0.1)
                    elif p == 'psi_50':
                        bins = np.arange(-10, -0.01, 0.1)
                    else:
                        print('oops')
                    xx = df[mi][p]
                    h.append(shannon_entropy(xx, [bins]))
                site_crit.append(np.nanmean(h))
            result['param_entropy'].append(site_crit)


            site_crit = []
            for mi, mod_p in zip(dic_mod['models'], dic_mod['mod_params']):
                equif_i = []
                for p1 in mod_p:
                    for p2 in [p for p in mod_p if p!=p1]:
                        rho = spearmanr(np.array(df[mi][p1]) , np.array(df[mi][p2]))[0]
                        equif_i.append(np.abs(rho))
                if mi not in ['SOXa_d1', 'SOX_d1']:
                    site_crit.append(np.nanmean(equif_i))
                else:
                    site_crit.append(0)
            result['param_equif'].append(site_crit)

            for tag, dd in zip(['full_rec', 'med', 'wet', 'dry'], [data_0, data_med, data_wet, data_dry]):
                site_crit = []
                for mi, mod_p in zip(dic_mod['models'], dic_mod['mod_params']):
                    target_g = '%s_%s_gen' % (target_m, mi)
                    nparams = len(mod_p)
                    aic = aikaike_criterion_rss(dd[target_o].values, dd[target_g].values, nparams)
                    site_crit.append(aic)
                result[tag]['aic'].append(site_crit)

            
            for tag, dd in zip(['full_rec', 'med', 'wet', 'dry'], [data_0, data_med, data_wet, data_dry]):
                site_crit_gen = []
                site_crit_site = []
                for mi in dic_mod['models']:
                    target_g = '%s_%s_gen' % (target_m, mi)
                    target_s = '%s_%s' % (target_m, mi)
                    a_gen = cal_it_performance(dd, target_g, target_o, source_1, source_2, nbins=n_bins, norm=1)
                    a_site = cal_it_performance(dd, target_s, target_o, source_1, source_2, nbins=n_bins, norm=1)
                    site_crit_gen.append([np.abs(ai) for ai in a_gen])
                    site_crit_site.append([np.abs(ai) for ai in a_site])
                site_crit_gen = list(zip(*site_crit_gen))
                site_crit_site = list(zip(*site_crit_site))
                for ai_gen, ai_si, k in zip(site_crit_gen, site_crit_site, ['a_p', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_ft', 'a_fp']):
                    result[tag][k].append(ai_gen)
                    result[tag]['d_%s'%k].append(np.array(ai_gen) - np.array(ai_si))
    return result


def rank_table(result, conditions):
    rank_im = []
    rank_criteria = []

    rank_im_ii = []
    for k in ['delta_nse', 'delta_mape']:
        rank_im_ii.append(data2rank(result[k]))
    rank_im_ii = np.mean(rank_im_ii, axis=0)
    rank_im.append(rank_im_ii)
    rank_criteria.append('(a) Cal/val\ngoodness-of-fit')

    rank_im_ii = []
    for k in [ 'param_entropy', 'param_cv']:
        rank_im_ii.append(data2rank(result[k]))
    rank_im_ii = np.mean(rank_im_ii, axis=0)
    rank_im.append(rank_im_ii)
    rank_criteria.append('(b) Parameter\nuncertainty')

    rank_im_ii = []
    for k in conditions:
        rank_im_ii.append(data2rank(result[k]['aic']))
    rank_im_ii = np.mean(rank_im_ii, axis=0)
    rank_im.append(rank_im_ii)
    rank_criteria.append('(c) Adequate\nparsimony')

    rank_im_ii = []
    for k in conditions:
        rank_im_ii.append(data2rank(result[k]['a_p']))
    rank_im_ii = np.mean(rank_im_ii, axis=0)
    rank_im.append(rank_im_ii)
    rank_criteria.append('(d) Predictive\naccuracy')

    rank_im_ii = []
    for ai in ['a_ft', 'a_fp',]:
        for k in conditions:
            rank_im_ii.append(data2rank(result[k][ai]))
    rank_im_ii = np.mean(rank_im_ii, axis=0)
    rank_im.append(rank_im_ii)
    rank_criteria.append('(e) Functional\naccuracy')

    return rank_criteria, rank_im


def rank_table_all(result, conditions, dic_mod):
    rank_im = []
    rank_criteria = []
    if (len(conditions) == 3) or (conditions[0]=='full_rec'):
        for ai in ['delta_nse', 'delta_mape', 'param_entropy', 'param_cv']:
            rank_im.append(data2rank(result[ai]))
            rank_criteria.append(ai)
    else:
        for ai in ['delta_nse', 'delta_mape', 'param_entropy', 'param_cv']:
            rank_im.append([np.nan for mi in dic_mod['models']])
            rank_criteria.append(ai)

    for ai in ['aic', 'a_p', 'd_a_p', 'a_ft', 'a_fp', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr']:
        rank_im_ii = []
        for k in conditions:
            rank_im_ii.append(data2rank(result[k][ai]))
        rank_im_ii = np.mean(rank_im_ii, axis=0)
        rank_im.append(rank_im_ii)
        rank_criteria.append(ai)

    return rank_criteria, rank_im


def fig_3M(fig_name, conditions=['full_rec', ]): # conditions = ['dry', 'med', 'wet']
    fig_name = os.path.join(fig_dir, fig_name)
    mod_list_labels = ['WUE', 'CM',  'SOX']
    mod_p = [['lww', 'b'],  ['b2', 'b1', 'a', 'psi_50'], ['kmax', 'a', 'psi_50']]
    timescale = 'd'
    mod_list = ['WUE_%s' % timescale,
                'CM_%s4' % timescale,
               'SOX_%s3' % timescale]
    dic_mod= {'models': mod_list, 'mod_params': mod_p}
    fig = plt.figure(figsize=(8.5, 4))

    for ii, pft_name, pft in pft_list:
        result = get_diagnostics(pft, dic_mod)
        rank_criteria, rank_im = rank_table(result, conditions)
        ax = fig.add_subplot(2, 4, ii)
        c = ax.imshow(np.array(rank_im), cmap="RdBu", vmin=0, vmax=1)
        plt.title(pft_name, fontsize=8)
        ax.set_xticks(range(len(mod_list_labels)))
        ax.set_xticklabels(mod_list_labels, fontsize=8)
        ax.set_yticks(range(len(rank_criteria) + 1))
        ax.set_yticklabels(['(a)', '(b)', '(c)', '(d)', '(e)'], fontsize=8)
        ax.set_ylim([len(rank_criteria) - 0.5, -0.5])

    pft = ['NF', 'BF', 'G-C3', 'C-C3'] 
    result = get_diagnostics(pft, dic_mod)
    rank_criteria, rank_im_all_avg = rank_table(result, conditions)
    ax = fig.add_subplot(1, 2, 1)
    print('Performance Scores')
    print('\tWUE\tCM\tSOX')
    for li, ri in zip(['(a)', '(b)', '(c)', '(d)', '(e)'], rank_im_all_avg):
        print('%s\t%-5.2f\t%-5.2f\t%-5.2f'%(li, ri[0], ri[1], ri[2]))
    c = ax.imshow(np.array(rank_im_all_avg), cmap="RdBu", vmin=0, vmax=1)
    
    ax.set_yticks(range(len(rank_criteria) + 1))
    ax.set_yticklabels(rank_criteria, fontsize=11)
    ax.set_xticks(range(len(mod_list_labels)))
    ax.set_xticklabels(mod_list_labels, fontsize=12)
    ax.set_ylim([len(rank_criteria)-0.5, -0.5])
    
    cbar = fig.colorbar(c, ticks=[0, 0.5, 1], )
    cbar.set_label('Average  performance  score', size=12)
    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def fig_SI_3M_a(fig_name):
    fig_name = os.path.join(fig_dir, 'SI', fig_name)
    mod_list_labels = ['WUE',  r'CM', r'SOX']
    mod_p = [['lww', 'b'],  ['b2', 'b1', 'a', 'psi_50'],  ['kmax', 'a', 'psi_50']]
    mod_list = ['WUE_d',
                'CM_d4',
                'SOX_d3']
    dic_mod= {'models': mod_list, 'mod_params': mod_p}
    pft = ['NF', 'BF', 'G-C3', 'C-C3']
    result = get_diagnostics(pft, dic_mod)
    fig = plt.figure(figsize=(8, 8))
    for ii, conditions, c_name in [[1, ['full_rec', ], 'Full record'],
                                   [2, ['dry', ], 'Dry quartile'],
                                   [3, ['med', ], 'Mesic interquartile'],
                                   [4, ['wet', ], 'Wet quartile']]:

        rank_criteria, rank_im_all_avg = rank_table_all(result, conditions, dic_mod)
        ax = fig.add_subplot(1, 4, ii)
        print('%s performance scores' % c_name)
        print('\t\tWUE\tCM\tSOX')
        for li, ri in zip(rank_criteria, rank_im_all_avg):
            if  np.isnan(ri[0])==0:
                if li in ['delta_nse', 'delta_mape', 'param_entropy', 'param_cv']:
                    print('%s\t%-5.2f\t%-5.2f\t%-5.2f'%(li, ri[0], ri[1], ri[2]))
                else:
                    print('%s\t\t%-5.2f\t%-5.2f\t%-5.2f'%(li, ri[0], ri[1], ri[2]))
        print('')
        c = ax.imshow(np.array(rank_im_all_avg), cmap="RdBu", vmin=0, vmax=1)
        ax.set_yticks(range(len(rank_criteria) + 1))
        ax.set_xticks(range(len(mod_list_labels)))
        ax.set_xticklabels(mod_list_labels, fontsize=12, rotation=35)
        ax.set_ylim([len(rank_criteria)-0.5, -0.5])
        plt.title(c_name, fontsize=12)

        rank_criteria_list = [r'$\Delta$NSE', r'$\Delta$MAPE',
                             r'$H(X)$', r'$CV(X)$', 
                             r'AIC', r'$A_p$', r'$\Delta A_{p}$',
                             r'$A_{f,T}$', r'$A_{f,p}$',
                             r'$A_{f,\theta}$', r'$A_{f,D}$', r'$A_{f,S}$', r'$A_{f,R}$',
                             ]
        if ii == 1:
            ax.set_yticklabels(rank_criteria_list, fontsize=12)
        else:
            ax.set_yticklabels([])
        
    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def fig_SI_5M_s(fig_name, conditions=['full_rec', ]): # conditions = ['dry', 'med', 'wet']
    fig_name = os.path.join(fig_dir, 'SI', fig_name)
    mod_list_labels = ['WUE', r'CM$_r$', r'CM', r'SOX$_r$', r'SOX']
    mod_p = [['lww', 'b'], ['b2', 'b1'], ['b2', 'b1', 'a', 'psi_50'], ['kmax', ], ['kmax', 'a', 'psi_50']]
    mod_list = ['WUE_d', 'CM_d2', 'CM_d4', 'SOX_d1', 'SOX_d3']
    dic_mod= {'models': mod_list, 'mod_params': mod_p}
    fig = plt.figure(figsize=(11, 4))

    for ii, pft_name, pft in pft_list:
        result = get_diagnostics(pft, dic_mod)
        rank_criteria, rank_im = rank_table(result, conditions)
        ax = fig.add_subplot(2, 4, ii)
        c = ax.imshow(np.array(rank_im), cmap="RdBu", vmin=0, vmax=1)
        plt.title(pft_name, fontsize=8)
        ax.set_xticks(range(len(mod_list_labels)))
        ax.set_xticklabels(mod_list_labels, fontsize=8, rotation=35)
        ax.set_yticks(range(len(rank_criteria) + 1))
        ax.set_yticklabels(['(a)', '(b)', '(c)', '(d)', '(e)'], fontsize=8)
        ax.set_ylim([len(rank_criteria) - 0.5, -0.5])

    pft = ['NF', 'BF', 'G-C3', 'C-C3'] 
    result = get_diagnostics(pft, dic_mod)
    rank_criteria, rank_im_all_avg = rank_table(result, conditions)
    ax = fig.add_subplot(1, 2, 1)
    #print('Performance Scores')
    #print('\tWUE\tCMr\tCM\tSOXr\tSOX')
    #for li, ri in zip(['(a)', '(b)', '(c)', '(d)', '(e)'], rank_im_all_avg):
    #    print('%s\t%-5.3f\t%-5.3f\t%-5.3f\t%-5.3f\t%-5.3f'%(li, ri[0], ri[1], ri[2], ri[3], ri[4]))
    c = ax.imshow(np.array(rank_im_all_avg), cmap="RdBu", vmin=0, vmax=1)
    
    ax.set_yticks(range(len(rank_criteria) + 1))
    ax.set_yticklabels(rank_criteria, fontsize=11)
    ax.set_xticks(range(len(mod_list_labels)))
    ax.set_xticklabels(mod_list_labels, fontsize=12)
    ax.set_ylim([len(rank_criteria)-0.5, -0.5])
    
    cbar = fig.colorbar(c, ticks=[0, 0.5, 1], )
    cbar.set_label('Average  performance  score', size=12)
    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


if __name__ == "__main__":
    pass