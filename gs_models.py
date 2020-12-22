import numpy as np
from constants import *



def cal_MM_Kc_temp(Tl):
    # Tl in deg C
    # Bernacchi 2001 as reported in Medlyn et al. 2002 (eq. 5)
    Kc = Kc_25C * np.exp(79430 * (Tl - 25) / (298 * Rd * (Tl + 273.15)))
    return Kc


def cal_MM_Ko_temp(Tl):
    # Tl in deg C
    # Bernacchi 2001 as reported in Medlyn et al. 2002 (eq. 6)
    Ko = Ko_25C * np.exp(36380 * (Tl - 25) / (298 * Rd * (Tl + 273.15)))
    return Ko


def cal_gamma_star_temp(Tl):
    # Tl in deg C
    # Bernacchi 2001 as reported in Medlyn et al. 2002 (eq. 6)
    gamma_star = gamma_star_25C * np.exp(37830 * (Tl - 25) / (298 * Rd * (Tl + 273.15)))
    return gamma_star


def cal_a2_Rub(Tl):
    Kc = cal_MM_Kc_temp(Tl)
    Ko = cal_MM_Ko_temp(Tl)
    a2 = Kc * ( 1 - Cao / Ko)
    return a2


def peaked_function_Medlyn(k25, Tl, Topt, Ha):
    # peaked function (eq. 17) in Medlyn et al. 2002
    # Tl in deg C
    deltaS = entropy_factor(Topt, Ha)
    num = 1 + np.exp((298 * deltaS - Hd) / (298. * Rd)) 
    denom = 1 + np.exp(((Tl + 273.15) * deltaS - Hd)/((Tl + 273.15) * Rd)) 
    expon = np.exp(Ha * (Tl - 25) / (298. * Rd * (Tl + 273.15)))
    return  k25 * expon * num / denom


def entropy_factor(Topt, Ha):
    # inversion of Medlyn et al. 2002 eq 19
    deltaS = Hd / Topt + Rd * np.log(Ha / (Hd - Ha))
    return deltaS


def electron_transport_rate(par_in, j_max, quantum_yield=phi_PSII, theta_j=0.9, leaf_scattering=0.2):
    # Bonan Book 2019
    j_psii = quantum_yield / 2. * (1 - leaf_scattering) * par_in
    jj = j_psii + j_max
    j = (jj - (jj ** 2 - 4 * theta_j * j_psii * j_max) ** 0.5) / (2 * theta_j)
    return j


def photo_k2(j, vc_max, a2):
    # Vico et al., 2013
    k2 =  j / 4 * a2  / vc_max
    return k2


def photo_k1(j):
    # Vico et al., 2013
    k1 =  j / 4 
    return k1


def f_psi(psi, psi_50, a=1):
    # Manzoni et al., 2013
    f = 1 / (1 + (psi / psi_50) ** a)
    return f


def model_beta(s, s_star, s_w):
    # Feddes function
    def __b(si):
        if si >= s_star:
            return 1
        elif si<= s_w:
            return 0
        else:
            return (si - s_w) / (s_star - s_w)
    beta = [__b(si) for si in s]
    return beta


def WUE_mwue(psi, l_ww, l_b):
    # Manzoni et al., 2011
    mwue = l_ww * np.exp(-psi * l_b)
    return mwue


def CM_mwue(psi_pd, psi_l, b2, b1, psi_50, a):
    # Wolf et al., 2016
    psi_m = (psi_pd + psi_l) / 2
    mxte = -b2 * psi_l + b1
    f = f_psi(psi_m, psi_50, a=a)
    mwue = mxte / f
    return mwue


def aSOX_mwue(psi_pd, psi_l, gpp, kmax, psi_50, a):
    # Eller et al., 2020
    # dK/dT = dpsi/dT * Kmax * df/dpsi = -1/(Kmax * f) * Kmax *  df/dpsi = -1/f * df/dpsi
    psi = (psi_pd + psi_l) / 2
    psi_m = (psi + psi_50) / 2
    f = f_psi(psi, psi_50, a=a)
    fm = f_psi(psi_m, psi_50, a=a)
    dKdT = - 1 / f * (f - fm) / (psi - psi_m)
    mwue =  - gpp / (f * kmax) * dKdT
    return mwue


def aSOX_mwue_it(psi_pd, psi_l, gpp, kmax, psi_50, a, vpd, ca, k1, k2, g_star):
    def __aSOX_mwue_t(psi_pd_i, psi_l_i, gpp_i, vpd_i, ca_i, k1_i, k2_i, g_star_i):
        psi = (psi_pd_i + psi_l_i) / 2
        psi_m = (psi + psi_50) / 2
        f = f_psi(psi, psi_50, a=a)
        fm = f_psi(psi_m, psi_50, a=a)
        dKdT = - 1 / f * (f - fm) / (psi - psi_m)
        mwue_0 = - dKdT * gpp_i / (f * kmax)

        ci = opt_ci(mwue_0, vpd_i, ca_i, k1_i, k2_i, g_star_i)
        a_photo = colim_A(k1_i, k2_i, ci, g_star_i)
        mwue_i = - dKdT * a_photo / (f * kmax)
        it = 0
        while ((np.abs(mwue_i - mwue_0) / mwue_0) > 0.05) and (it < 100):
            mwue_0 = mwue_i
            ci = opt_ci(mwue_0, vpd_i, ca_i, k1_i, k2_i, g_star_i)
            a_photo = colim_A(k1_i, k2_i, ci, g_star_i)
            mwue_i = - dKdT * a_photo / (f * kmax)
            it = it + 1
        return mwue_i

    mwue = [__aSOX_mwue_t(psi_pd_i, psi_l_i, gpp_i, vpd_i, ca_i, k1_i, k2_i, g_star_i)
                for psi_pd_i, psi_l_i, gpp_i, vpd_i, ca_i, k1_i, k2_i, g_star_i 
                in zip(psi_pd, psi_l, gpp, vpd, ca, k1, k2, g_star)]
    return np.array(mwue)


def SOX_mwue(psi_pd, psi_l, gpp, kmax, psi_50, a):
    # Eller et al., 2018
    psi = (psi_pd + psi_l) / 2
    f = f_psi(psi, psi_50, a=a)
    dKdT = a / psi *  f * (psi / psi_50) ** a
    mwue = - dKdT * gpp / (f * kmax)
    return mwue


def SOX_mwue_it(psi_pd, psi_l, gpp, kmax, psi_50, a, vpd, ca, k1, k2, g_star):
    def __SOX_mwue_t(psi_pd_i, psi_l_i, gpp_i, vpd_i, ca_i, k1_i, k2_i, g_star_i):
        psi = (psi_pd_i + psi_l_i) / 2
        f = f_psi(psi, psi_50, a=a)
        dKdT = a / psi *  f * (psi / psi_50) ** a
        mwue_0 = - dKdT * gpp_i / (f * kmax)

        ci = opt_ci(mwue_0, vpd_i, ca_i, k1_i, k2_i, g_star_i)
        a_photo = colim_A(k1_i, k2_i, ci, g_star_i)
        mwue_i = - dKdT * a_photo / (f * kmax)
        it = 0
        while ((np.abs(mwue_i - mwue_0) / mwue_0) > 0.05) and (it < 100):
            mwue_0 = mwue_i
            ci = opt_ci(mwue_0, vpd_i, ca_i, k1_i, k2_i, g_star_i)
            a_photo = colim_A(k1_i, k2_i, ci, g_star_i)
            mwue_i = - dKdT * a_photo / (f * kmax)
            it = it + 1
        return mwue_i

    mwue = [__SOX_mwue_t(psi_pd_i, psi_l_i, gpp_i, vpd_i, ca_i, k1_i, k2_i, g_star_i)
                for psi_pd_i, psi_l_i, gpp_i, vpd_i, ca_i, k1_i, k2_i, g_star_i 
                in zip(psi_pd, psi_l, gpp, vpd, ca, k1, k2, g_star)]
    return np.array(mwue)


def gs_opt_canopy(gpp, ca, vpd, k2, g_star, mwue):
    # Dewar et al., 2018
    X = (k2 + g_star) / (1.6 * mwue * vpd)
    g = (1 + X ** 0.5) * gpp / (ca - g_star)
    return 1.6 * g


def gs_opt_leaf(mwue, vpd, ca, k1, k2, g_star):
    # Dewar et al., 2018
    w = (ca - g_star) / (k2 + g_star) 
    z = 1.6 * mwue * vpd / (k2 + g_star)
    x = 1 / (1 + z **0.5)
    g = k1 / (k2 + g_star) * x / (1 - x) / (x * w + 1)
    return 1.6 * g


def opt_ci(mwue, vpd, ca, k1, k2, g_star):
    # Dewar et al., 2018
    e = ((k2 + g_star) / (1.6 * mwue)) ** 0.5
    ci = e / (e + vpd ** 0.5) * (ca - g_star) + g_star
    return ci


def colim_A(k1, k2, ci, g_star):
    # Vico et al., 2013
    a = k1 * (ci - g_star) / (ci + k2)
    return a


def diffusion_A(mwue, vpd, ca, k1, k2, g_star):
    # Fick's law
    ci = opt_ci(mwue, vpd, ca, k1, k2, g_star)
    gs = gs_opt_leaf(mwue, vpd, ca, k1, k2, g_star)
    a = 1 / 1.6 * gs * (ca - ci)
    return a


def model_g_soil_lin(s, gsoil_max):
    # linear approximation of soil conductance
    gsoil = gsoil_max * s
    return gsoil



if __name__ == "__main__":
    pass