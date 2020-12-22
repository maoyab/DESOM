import os
import sys
import pandas as pd
from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt
from data_management import *
from constants import *
from gs_models import *
from et_models import *
from fit_gs_params import *
from information_metrics import *


mod_list = ['WUE_d',  'CM_d4', 'SOX_d3']
model_colors = ['#04ABC2', '#B7CE63', '#FDE10D']
pft_list = ['NF', 'BF',  'G-C3', 'C-C3']
pft_list_names = ['Needleleaf', 'Broadleaf',  'Grass', 'Crop']
markers = [ 'o', 's', '^']

sites_params = pd.read_csv( '../../DATA/EEGS/sel2_sites_params.csv')
sites = sites_params['site_id'].values

sel = 'revision_0'
res_dir = '../../DATA/EEGS/fitted_models_%s' % sel
fig_dir = '../../PROJECTS/Stomatal_conductance_eval/NP_figures/SI'
sites_files = [os.path.join(res_dir, f) 
                for f in os.listdir(res_dir) 
                    if f.endswith('gen.csv')]

def get_result_dict_i(v_keys, n_bins = 15):
    [res_dir_i, source_1, source_2, sensitivity_var, target_m, target_o, mod_list_all] = v_keys
    result = {}

    for l in ['site', 'general']:
        result[l] = {}
        for k in [ 'Wet', 'Dry', 'Mesic', 'Overall', 'Full']:
            result[l][k] = {}
            for a in ['a_p', 'a_ft', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_fp']: 
                result[l][k][a] = {}
                for mod in mod_list_all:
                    result[l][k][a][mod] = []
    
    for site_i in sites:
        data_file =  [os.path.join(res_dir_i, fi) for fi in os.listdir(res_dir_i) 
                                                        if fi.endswith('gen.csv') and (site_i in fi)][0]
        data_0 = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                          infer_datetime_format = True,)
       
        vv = [source_1, source_2, target_o]
        for mi in mod_list_all:
            vv.append('%s_%s' % (target_m, mi))
            vv.append('%s_%s_gen' % (target_m, mi))
        data_0 = data_0[vv].dropna()

        p25 = np.percentile(data_0[sensitivity_var].values, 25)
        p75 = np.percentile(data_0[sensitivity_var].values, 75)

        data_dry = data_0[data_0[sensitivity_var] <= p25]
        data_med = data_0[(data_0[sensitivity_var] < p75) & (data_0[sensitivity_var] > p25)]
        data_wet = data_0[data_0[sensitivity_var] >= p75]
        
        for mi in mod_list_all:
            a_o = []
            for tag, dd in zip(['Mesic', 'Wet', 'Dry', 'Full'], [data_med, data_wet, data_dry, data_0]):
                target_g = '%s_%s_gen' % (target_m, mi)
                a = cal_it_performance(dd, target_g, target_o, source_1, source_2, nbins=n_bins, norm=1)
                if tag != 'Full':
                    a_o.append(a)
                for ai, k in zip(a, ['a_p', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_ft', 'a_fp']):
                    result['general'][tag][k][mi].append(ai)
            a_o = zip(*a_o)
            a_o = [np.mean(aa) for aa in a_o]
            for ai, k in zip(a_o, ['a_p', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_ft', 'a_fp']):
                result['general']['Overall'][k][mi].append(ai)

        for mi in mod_list_all:
            a_o = []
            for tag, dd in zip(['Mesic', 'Wet', 'Dry', 'Full'], [data_med, data_wet, data_dry, data_0]):
                target = '%s_%s' % (target_m, mi)
                a = cal_it_performance(dd, target, target_o, source_1, source_2, nbins=n_bins, norm=1)
                if tag != 'Full':
                    a_o.append(a)
                for ai, k in zip(a, ['a_p', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_ft', 'a_fp']):
                    result['site'][tag][k][mi].append(ai)
            a_o = zip(*a_o)
            a_o = [np.mean(aa) for aa in a_o]
            for ai, k in zip(a_o, ['a_p', 'a_fu1', 'a_fu2', 'a_fs', 'a_fr', 'a_ft', 'a_fp']):
                result['site']['Overall'][k][mi].append(ai)
    return result


def scatter_T_variant(fig_name, cond='Full'):
    fig_name = os.path.join(fig_dir, '%s' % (fig_name))
    
    source_1 = 'S'
    sensitivity_var = source_1
    source_2 = 'VPD_a'
    target_m = 'LE'
    target_m2 = 'LE_LAI'
    target_o = 'LE_F_MDS'
    
    results_all = {}
    results_all_2 = {}
    for sel in ['0', 'T']:
        res_dir_i = '../../DATA/EEGS/fitted_models_revision_%s' % sel
        v_keys = [res_dir_i, source_1, source_2, sensitivity_var, target_m, target_o, mod_list]
        v_keys_2 = [res_dir_i, source_1, source_2, sensitivity_var, target_m2, target_o, mod_list]

        results_all[sel] = get_result_dict_i(v_keys)
        results_all_2[sel] = get_result_dict_i(v_keys_2)
        
    fig = plt.figure(figsize=(8.5, 8))
    ax = fig.add_subplot(2, 2, 1)
    for mod, color, marker in zip(mod_list, model_colors, markers):
        x = results_all['0']['general'][cond]['a_p'][mod]
        y = results_all['T']['general'][cond]['a_p'][mod]
        ax.scatter(x, y, color=color, marker=marker, label='$%s$'%mod.split('_')[0])
    lim = [0.35, 1]
    ax.plot(lim, lim, linestyle=':', color='k')
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    ax.set_ylabel('$A_p$ (T dependency)', fontsize=14)
    ax.set_xlabel('$A_p$ (T=25°C)', fontsize=14)
    plt.title('(a) Model with GPP substitution', fontsize=14)
    plt.legend(frameon=False)

    ax = fig.add_subplot(2, 2, 2)
    for mod, color, marker in zip(mod_list, model_colors, markers):
        x = results_all_2['0']['general'][cond]['a_p'][mod]
        y = results_all_2['T']['general'][cond]['a_p'][mod]
        ax.scatter(x, y, color=color, marker=marker)
    lim = [0.35, 1]
    ax.plot(lim, lim, linestyle=':', color='k')
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    ax.set_ylabel('$A_p$ (T dependency)', fontsize=14)
    ax.set_xlabel('$A_p$ (T=25°C)', fontsize=14)
    plt.title('(b) Model scaling with MODIS LAI', fontsize=14)

    ax = fig.add_subplot(2, 2, 3)
    for mod, color, marker in zip(mod_list, model_colors, markers):
        x = results_all['0']['general'][cond]['a_fp'][mod]
        y = results_all['T']['general'][cond]['a_fp'][mod]
        ax.scatter(x, y, color=color, marker=marker)
    lim = [0, 1]
    ax.plot(lim, lim, linestyle=':', color='k')
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    ax.set_ylabel('$A_{f,P}$ (T dependency)', fontsize=14)
    ax.set_xlabel('$A_{f,P}$ (T=25°C)', fontsize=14)

    ax = fig.add_subplot(2, 2, 4)
    for mod, color, marker in zip(mod_list, model_colors, markers):
        x = results_all_2['0']['general'][cond]['a_fp'][mod]
        y = results_all_2['T']['general'][cond]['a_fp'][mod]
        ax.scatter(x, y, color=color, marker=marker)
    lim = [0, 1]
    ax.plot(lim, lim, linestyle=':', color='k')
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    ax.set_ylabel('$A_{f,P}$ (T dependency)', fontsize=14)
    ax.set_xlabel('$A_{f,P}$ (T=25°C)', fontsize=14)

    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def scatter_psi_variant(fig_name, cond='Full'):
    fig_name = os.path.join(fig_dir, '%s' % (fig_name))

    source_1 = 'S'
    sensitivity_var = source_1
    source_2 = 'VPD_a'
    target_m = 'LE'
    target_o = 'LE_F_MDS'
    g = 'general'
    mod_list_z = ['WUE_z', 'CM_z4', 'SOX_z3']
    mod_list_all = ['WUE_d', 'CM_d4', 'SOX_d3',
                    'WUE_i', 'CM_i4', 'SOX_i3']
    
    v_keys = [res_dir, source_1, source_2, sensitivity_var, target_m, target_o, mod_list_all]
    results_all = get_result_dict_i(v_keys)

    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 2, 1)
    for mod, color, marker in zip(mod_list_z, model_colors, markers):
        x = results_all[g][cond]['a_p'][mod.replace("z", "d")]
        y = results_all[g][cond]['a_p'][ mod.replace("z", "i")]
        ax.scatter(x, y, color=color, marker=marker, label='$%s$'%mod.split('_')[0])
    lim = [0.35, 0.85]
    ax.plot(lim, lim, linestyle=':', color='k')
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    ax.set_ylabel(r'$A_p$ (canopy $\psi$)', fontsize=14)
    ax.set_xlabel(r'$A_p$ (predawn $\psi$)', fontsize=14)
    plt.title('(a)', fontsize=14)
    plt.legend(frameon=False)

    ax = fig.add_subplot(1, 2, 2)
    for mod, color, marker in zip(mod_list_z, model_colors, markers):
        x = results_all[g][cond]['a_fp'][mod.replace("z", "d")]
        y = results_all[g][cond]['a_fp'][ mod.replace("z", "i")]
        ax.scatter(x, y, color=color, marker=marker)
    lim = [0, 1]
    ax.plot(lim, lim, linestyle=':', color='k')
    ax.set_ylim(lim)
    ax.set_xlim(lim)
    ax.set_ylabel(r'$A_{f,P}$ (canopy $\psi$ )', fontsize=14)
    ax.set_xlabel(r'$A_{f,P}$ (predawn $\psi$ )', fontsize=14)
    plt.title('(b)', fontsize=14)

    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def avg_ET_partitioning(fig_name):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(12, 3))
    color = 'k'
    mod = 'WUE_d'
    et_partitioning = []
    site_ids = []
    for pft in pft_list:
        sites_i = sites_params[(sites_params['pft'] == pft)]['site_id'].values
        for site_i in sites_i:
            data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                            if fi.endswith('gen.csv') and (site_i in fi)][0]
            pickle_file = [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                                if fi.endswith('pickle') and (site_i in fi)][0]
            with open(pickle_file, 'rb') as f:
                    splitfit_results = load(f) 
            
            data = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                          infer_datetime_format = True)
            data = data[data['G_surf_mol']>0]

            gsoilmax = np.nanmedian(splitfit_results[mod]['gsoil_max'])
            g_surf = data['G_surf_mol']
            g_soil = data['S'] * gsoilmax
            t = np.nanmedian(np.median(g_soil/g_surf))
            et_partitioning.append(t)
            site_ids.append('%s (%s)' % (site_i, pft[0]))


    ax = fig.add_subplot(1, 1, 1)
    x = np.array(range(len(et_partitioning)))
    ax.bar(x, et_partitioning, color=color, align='center')
    ax.set_xticks(x)
    ax.set_xticklabels(site_ids, rotation=90, fontsize=10)
    ax.tick_params(direction='inout')
    ax.set_ylim([0, 0.3])
    ax.set_ylabel(r'$G_{soil}$/$G_{surf}$', fontsize=16)
    plt.tight_layout()
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def ts_scalling_avg(fig_name):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(8, 10))
    ii = 1
    for jj, (pft, pft_n) in enumerate(zip(pft_list, pft_list_names)):
        sites_i = sites_params[(sites_params['pft'] == pft)]['site_id'].values
        for ii, site_i in enumerate(sites_i):
            data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                            if fi.endswith('gen.csv') and (site_i in fi)][0]
            data = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                          infer_datetime_format = True)
            data = data[data['G_surf_mol']>0]

            ax = fig.add_subplot(9, 4, 1 + jj + ii * 4)
            data = data.reindex(pd.date_range(start=data.index.min(),
                                        end=data.index.max(),
                                        freq='30min'))
            data['DOY'] = data.index.dayofyear
            data['Date'] = data.index.date
            doy = data.groupby(['DOY'])['DOY'].mean()
            lai = data.groupby(['DOY'])['LAI'].median()

            elai_models = []
            for elai_mod in ['eLAI_WUE_d', 'eLAI_CM_d4', 'eLAI_SOX_d3']:
                data = remove_outliers(data, elai_mod, mad_th=20, window=None)
                data = remove_outliers(data, elai_mod, mad_th=3, window='1D')
                elai_models.append(data[elai_mod])
            
            e_lai = [ np.nanmean([ ei if ei<3*np.nanmin(elai_d) else np.nan for ei in elai_d ]) for elai_d in zip(*elai_models)]

            data['e_LAI_avg'] = e_lai
            elai = data.groupby(['DOY'])['e_LAI_avg'].median()

            ax.plot(doy, lai, color='tomato', lw=0, marker='.', markersize=1)
            ax.plot(doy, elai, color='k', lw=0, marker='.', markersize=2)
            if site_i.endswith('R'):
                site_i = site_i[:-1]
            ax.plot(doy, elai, color='none', lw=0, marker='', label=site_i)

            ax.set_xlim([1, 365])
            ax.set_ylim([0, 3.5])
            ax.set_yticks([0, 2])
            if ii == 0:
                plt.title(pft_n)
            if ii == len(sites_i) - 1:
                ax.set_xlabel('Day of year')
            else:
                ax.set_xticklabels([])
            if (jj ==0) and (ii == 4):
                ax.set_ylabel('$G_{canopy}$/$g_s$', fontsize=18)
            #if jj>0:
            #    ax.set_yticklabels([])
            #else:
            ax.set_yticklabels([0, 2])
            ax.tick_params(direction='inout')
            plt.legend(frameon=False, fontsize=8, loc='upper right')
    plt.savefig('%s.png' % fig_name, dpi=600)
    plt.savefig('%s.eps' % fig_name)


def mwue_avg(fig_name, cond='Full'):
    fig_name = os.path.join(fig_dir, fig_name)
    fig = plt.figure(figsize=(8, 4))
    w = 0.1
    for ii, (color, mod) in enumerate(zip(model_colors, mod_list)):
        mwue = []
        site_ids = []
        for pft in pft_list:
            sites_i = sites_params[(sites_params['pft'] == pft)]['site_id'].values
            for site_i in sites_i:
                data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                                if fi.endswith('gen.csv') and (site_i in fi)][0]
                data = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                              infer_datetime_format = True)
                data = data[data['G_surf_mol']>0]

                p25 = np.percentile(data['S'].values, 25)
                p75 = np.percentile(data['S'].values, 75)

                if cond == 'Full':
                    data = data
                elif cond == 'Dry':
                    data = data[data['S'] <= p25]
                elif cond == 'Mesic':
                    data = data[(data['S'] < p75) & (data['S'] > p25)]
                elif cond == 'Wet':
                    data = data[data['S'] >= p75]
                data['mwue_avg'] = (data['mwue_WUE_d'] + data['mwue_CM_d4'] + data['mwue_SOX_d3']) /3.
                mwue.append(np.nanmedian(data['mwue_%s' % mod])) #/ data['mwue_WUE_d']
                site_ids.append('%s (%s)' % (site_i, pft[0]))

        ax = fig.add_subplot(1, 1, 1)
        x = np.array(range(len(mwue)))
        ax.bar(x + w*ii, mwue, color=color, width=w, align='center')

    ax.set_xticks(x + 2*w)
    ax.set_xticklabels(site_ids, rotation=90)
    ax.set_ylim([0, 0.02])
    ax.set_ylabel(r'$\frac{d\Theta}{dT}$')
    plt.tight_layout()
    plt.savefig(fig_name)


def mwue_ranks(fig_name, cond='Full'):
    fig_name = os.path.join(fig_dir, '%s_%s' % (cond, fig_name))
    fig = plt.figure(figsize=(8, 4))
    w = 0.1
   
    mwue_ranks = []
    site_ids = []
    test_a = []
    for pft in pft_list:
        sites_i = sites_params[(sites_params['pft'] == pft)]['site_id'].values
        for site_i in sites_i:
            data_file =  [os.path.join(res_dir, fi) for fi in os.listdir(res_dir) 
                                                            if fi.endswith('gen.csv') and (site_i in fi)][0]
            data = pd.read_csv(data_file, header = 0, index_col = 0, parse_dates = True, 
                          infer_datetime_format = True)
            
            data = data[data['G_surf_mol'] > 0]

            p25 = np.percentile(data['S'].values, 25)
            p75 = np.percentile(data['S'].values, 75)

            if cond == 'Full':
                data = data
            elif cond == 'Dry':
                data = data[data['S'] <= p25]
            elif cond == 'Mesic':
                data = data[(data['S'] < p75) & (data['S'] > p25)]
            elif cond == 'Wet':
                data = data[data['S'] >= p75]
            test_a_WUE = 1.6 * data['VPD_l'] * data['mwue_WUE_d'] / (data['CO2'] - gamma_star_25C)
            test_a_CM = 1.6 * data['VPD_l'] * data['mwue_CM_d4'] / (data['CO2'] - gamma_star_25C)
            test_a_SOX = 1.6 * data['VPD_l'] * data['mwue_SOX_d3'] / (data['CO2'] - gamma_star_25C)
            test_a_i = [np.round(sum([1 for ti in test_a_WUE if ti<1])/np.float(len(test_a_WUE))*100),
                        np.round(sum([1 for ti in test_a_CM if ti<1])/np.float(len(test_a_CM))*100),
                        np.round(sum([1 for ti in test_a_SOX if ti<1])/np.float(len(test_a_SOX))*100)]
            #print(pft, site_i, test_a_i)
            test_a.append(test_a_i)
            rank_wue = [list(np.sort([wue, cm, sox])).index(wue) + 1 for wue, cm, sox in zip( data['mwue_WUE_d'], data['mwue_CM_d4'],  data['mwue_SOX_d3'])]
            rank_cm = [list(np.sort([wue, cm, sox])).index(cm) + 1 for wue, cm, sox in zip( data['mwue_WUE_d'], data['mwue_CM_d4'],  data['mwue_SOX_d3'])]
            rank_sox = [list(np.sort([wue, cm, sox])).index(sox) + 1 for wue, cm, sox in zip( data['mwue_WUE_d'], data['mwue_CM_d4'],  data['mwue_SOX_d3'])]
            
            mwue_ranks.append([np.median(rank_wue), np.median(rank_cm), np.median(rank_sox)])
            site_ids.append('%s (%s)' % (site_i, pft[0]))
    test_a_WUE, test_a_CM, test_a_SOX = list(zip(*test_a))
    #print('a test', np.mean(test_a_WUE), np.mean(test_a_CM),  np.mean(test_a_SOX))
    ax = fig.add_subplot(1, 1, 1)
    for ii, (color, ranks) in enumerate(zip(model_colors, zip(*mwue_ranks))):
        print(color, np.sum([1 for r in ranks if r<2])/30*100)
        x = np.array(range(len(ranks)))
        ax.bar(x + w*ii, ranks, color=color, width=w, align='center')

    ax.set_xticks(x + 2*w)
    ax.set_xticklabels(site_ids, rotation=90)

    ax.set_ylabel(r'WUE rank')
    plt.tight_layout()
    plt.savefig(fig_name)



if __name__ == "__main__":
    pass
