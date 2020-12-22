import os
import numpy as np
import pandas as pd
import random
from scipy.stats import percentileofscore
from multiprocessing import Pool


def aikaike_criterion_rss(obs, mod, nparams):
    ni = np.float(len(obs))
    rss = np.sum([(oi - mi)**2 for oi, mi in zip(obs, mod)])
    aic = 2 * nparams + ni * np.log(rss)
    return aic


def aikaike_criterion_it(obs, mod, nparams, nbins=100):
    bins = np.linspace(0, 1, nbins+1)
    
    mod = (mod - np.nanmin(obs))/(np.nanmax(obs) - np.nanmin(obs))
    mod = [mi if mi>0 else 0 for mi in mod]
    mod = [mi if mi<1 else 1 for mi in mod]
    obs = (obs - np.nanmin(obs))/(np.nanmax(obs) - np.nanmin(obs))
    
    p_obs = np.histogramdd([obs], [bins])[0]
    p_obs = p_obs / np.sum(p_obs)
    p_mod = np.histogramdd([mod], [bins])[0]
    p_mod = p_mod / np.sum(p_mod)
    
    dl_score = np.sum([oi*np.log2(oi / mi)  if ((mi > 0) and (oi > 0)) else 0 for oi, mi in zip(p_obs, p_mod)])

    aic = 2 * nparams + 2 * dl_score
    return aic


def cal_nse(obs, mod):
    mo = np.nanmean(obs)
    a = np.nansum([(mi - oi) ** 2 for mi, oi in zip(mod, obs)])
    b = np.nansum([(oi - mo) ** 2 for oi in obs])
    return 1 - a / b


def cal_mape(obs, mod):
    mo = np.nanmean(obs)
    ape = [np.abs(mi - oi) / mo for mi, oi in zip(mod, obs)]
    return np.nanmean(ape)


def shannon_entropy(x, bins):
    c = np.histogramdd(x, bins)[0]
    p = c / np.sum(c)
    p = p[p > 0]
    h =  - np.sum(p * np.log2(p))
    return h


def interaction_information(mi_c, mi):
    i = mi_c - mi
    return i


def normalized_source_dependency(mi_s1_s2, H_s1, H_s2):
    i = mi_s1_s2 / np.min([H_s1, H_s2])
    return i


def redundant_information_bounds(mi_s1_tar, mi_s2_tar, interaction_info):
    r_mmi = np.min([mi_s1_tar, mi_s2_tar])
    r_min = np.max([0, - interaction_info])
    return r_mmi, r_min


def rescaled_redundant_information(mi_s1_s2, H_s1, H_s2, mi_s1_tar, mi_s2_tar, interaction_info):
    norm_s_dependency = normalized_source_dependency(mi_s1_s2, H_s1, H_s2)
    r_mmi, r_min = redundant_information_bounds(mi_s1_tar, mi_s2_tar, interaction_info)
    return r_min + norm_s_dependency * (r_mmi - r_min)


def mutual_information(dfi, source, target, bins, reshuffle=0):
    x = dfi[source].values
    y = dfi[target].values
    if reshuffle == 1:
        random.shuffle(x)
        random.shuffle(y)
    H_x = shannon_entropy([x], [bins])
    H_y = shannon_entropy([y], [bins])
    H_xy = shannon_entropy([x, y], [bins, bins])
    return H_x + H_y - H_xy


def mutual_information_0(dfi, source, target, bins, reshuffle=0):
    x = dfi[source].values
    y = dfi[target].values
    if reshuffle == 1:
        random.shuffle(x)
        random.shuffle(y)

    pxy = np.histogram2d(x, y, bins=bins)[0]
    pxy = pxy / (np.sum(pxy + 1e-6))
    px = np.sum(pxy, axis=0).reshape((-1, pxy.shape[0]))
    py = np.sum(pxy, axis=1).reshape((pxy.shape[1], -1))
    H_x = -np.sum(px * np.log2(px))
    H_y = -np.sum(py * np.log2(py))
    H_xy = -np.sum(pxy * np.log2(pxy)) 
    return H_x + H_y - H_xy


def conditional_mutual_information(df, source, target, condition, bins, reshuffle=0, ts='D'):
    #transfer entropy is  T(X(t) > T(t), tau) : X(t - tau)=x, Y(t)=y, Y(t - tau)=z
    if reshuffle == 1:
        x = df[source].values
        y = df[target].values
        random.shuffle(x)
        random.shuffle(y)
        df[source] = x
        df[target] = y
        if type(condition) == str:
            z = df[condition].values
            random.shuffle(z)
            df[condition] = z

    if type(condition) == str:
        df['x'] = df[source]
        df['z'] = df[condition]
        df['y'] = df[target]
    else:
        df = df.resample(ts).mean()
        y = df[target].values
        df = df.shift(periods=condition)
        df['x'] = df[source]
        df['z'] = df[target]
        df['y'] = y

    df = df[['x', 'y', 'z']]
    df = df.dropna()
    x = df['x'].values
    y = df['y'].values
    z = df['z'].values

    H_y = shannon_entropy([y],[bins])
    H_z = shannon_entropy([z],[bins])
    H_xz = shannon_entropy([x, z],[bins, bins])
    H_yz = shannon_entropy([y, z],[bins, bins])
    H_xyz = shannon_entropy([x, y, z],[bins, bins, bins])
    return H_xz + H_yz - H_z - H_xyz


def information_partitioning(df, source_1, source_2, target, bins, reshuffle=0):
    if reshuffle == 1:
        x = df[source_1].values
        z = df[source_2].values
        y = df[target].values
        random.shuffle(x)
        random.shuffle(y)
        random.shuffle(z)
        df[source_1] = x
        df[source_2] = z
        df[target] = y

    H_s1 = shannon_entropy(df[source_1].values, [bins])
    H_s2 = shannon_entropy(df[source_2].values, [bins])
    mi_s1_s2 = mutual_information(df, source_1, source_2, bins)
    mi_s1_tar = mutual_information(df, source_1, target, bins)
    mi_s2_tar = mutual_information(df, source_2, target, bins)
    mi_s1_tar_cs2 = conditional_mutual_information(df, source_1, target, source_2, bins)
    interaction_info = interaction_information(mi_s1_tar_cs2, mi_s1_tar)

    redundant = rescaled_redundant_information(mi_s1_s2, H_s1, H_s2, mi_s1_tar, mi_s2_tar, interaction_info)
    unique_s1 = mi_s1_tar - redundant
    unique_s2 = mi_s2_tar - redundant
    synergistic = interaction_info + redundant
    total_information = unique_s1 + unique_s2 + redundant + synergistic
    return total_information, unique_s1, unique_s2, redundant, synergistic


def sig_test_info_partitioning(df, source_1, source_2, target, bins, nshuffles=1000):
    df = df.dropna()
    H_tar = shannon_entropy(df[target].values, [bins])
    I_tar_s1 = mutual_information(df, source_1, target, bins, reshuffle=0)
    I_tar_s2 = mutual_information(df, source_2, target, bins, reshuffle=0) 
    total_information, unique_1, unique_2, redundant, synergistic = information_partitioning(df, source_1, source_2, target, bins, reshuffle=0)
    info_bootstrapping = [information_partitioning(df, source_1, source_2, target, bins, reshuffle=1) for i in range(nshuffles)]
    total_info_bootstrapping = zip(*info_bootstrapping)[0]
    return [total_information, unique_1, unique_2, synergistic, redundant, H_tar, I_tar_s1, I_tar_s2, \
            percentileofscore(total_info_bootstrapping, total_information) / 100.]


def sig_test_transfer_entropy(df, source, target, lag, bins, nshuffles=1000):
    df = df.dropna()
    te = conditional_mutual_information(df, source, target, lag, bins, reshuffle=0)
    te_bootstrapping = [conditional_mutual_information(df, source, target, lag, bins, reshuffle=1) for i in range(nshuffles)]
    return lag, te, percentileofscore(te_bootstrapping, te) / 100.


def sig_test_mutual_information(df, source, target, bins, nshuffles=1000):
    df = df.dropna()
    mi = mutual_information(df, source, target, bins, reshuffle=0)
    mi_bootstrapping = [mutual_information(df, source, target, bins, reshuffle=1) for i in range(nshuffles)]
    return mi, percentileofscore(mi_bootstrapping, mi) / 100.


def cal_it_performance(data_i, target, target_o, source_1, source_2, nbins=10, norm=0):
    def __rescale_variables(df):
        df[source_1] = (df[source_1] - np.nanmin(df[source_1]))\
                                        / (np.nanmax(df[source_1]) - np.nanmin(df[source_1]))
    
        df[source_2] = (df[source_2] - np.nanmin(df[source_2]))\
                            / (np.nanmax(df[source_2]) - np.nanmin(df[source_2]))

        df[target] = (df[target] - np.nanmin(df[target]))\
                             / (np.nanmax(df[target_o]) - np.nanmin(df[target_o]))
        df[target] = [ti if ti>0 else 0 for ti in df[target].values]
        df[target] = [ti if ti<1 else 1 for ti in df[target].values]
        
        df[target_o] = (df[target_o] - np.nanmin(df[target_o]))\
                            / (np.nanmax(df[target_o]) - np.nanmin(df[target_o]))
        return df
    
    bins = np.linspace(0, 1, nbins+1)
    
    data_i = data_i[[target, target_o, source_1, source_2]].dropna()
    data_i = __rescale_variables(data_i)

    H_tar_o = shannon_entropy(data_i[target_o].values, [bins])
    I_taro_tarm = mutual_information(data_i, target_o, target, bins)

    total_information_o, unique_1_o, unique_2_o, redundant_o, synergistic_o = information_partitioning(data_i, source_1, source_2, target_o, bins)
    total_information, unique_1, unique_2, redundant, synergistic = information_partitioning(data_i, source_1, source_2, target, bins)
    
    if norm == 0:
        a_p = H_tar_o - I_taro_tarm
        a_fu1 = unique_1 - unique_1_o
        a_fu2 = unique_2 - unique_2_o 
        a_fr = redundant - redundant_o
        a_fs = synergistic - synergistic_o
        a_ft = total_information - total_information_o
    else:
        a_p = (H_tar_o - I_taro_tarm) / H_tar_o
        a_fu1 = unique_1/total_information - unique_1_o/total_information_o
        a_fu2 = unique_2/total_information - unique_2_o/total_information_o
        a_fr = redundant/total_information - redundant_o/total_information_o
        a_fs = synergistic/total_information - synergistic_o/total_information_o
        a_ft = (total_information - total_information_o) / total_information_o
    
    a_fp = np.abs(a_fu1) + np.abs(a_fu2) + np.abs(a_fs) + np.abs(a_fr)
    
    return a_p, a_fu1, a_fu2, a_fs, a_fr, a_ft, a_fp


if __name__ == "__main__":

    pass

