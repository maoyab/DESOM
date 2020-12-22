import numpy as np
from constants import *
import scipy.optimize as optimize


def g_to_mol(G, atm_pressure, air_tmep):
    # m/s to mol/m2/s
    g = G * atm_pressure / (Rd * (air_tmep + 273.15))
    return g

def g_from_mol(g, atm_pressure, air_tmep):
    # mol/m2/s to m/s
    G = g * (air_tmep + 273.15) * Rd / atm_pressure
    return G

def Lv(air_temp):
    # J/kg
    return (2.501 - 0.00236 * (air_temp - 273.16)) * 10 **6

def vapor_pressure_slope(air_temp):
    # Pa/C
    a =  np.exp((17.27 * air_temp) / (air_temp + 237.3)) 
    b = (air_temp + 237.3) ** 2
    delta = a / b * 2.504 * 10 ** 6 
    return delta

def psychrometric_cste(atm_pressure, air_temp):
    # Pa/C
    return cp_air * atm_pressure / (Lv(air_temp) * mwratio)

def zero_plane_displacement_h(hc):
    # m
    return 2/3. * hc

def roughness_length_heat_transfer(hc):
    # m
    return 0.01 * hc

def roughness_length_momentum_transfer(hc):
    # m
    return 0.1 * hc

def obukov_stability(h_flux, air_temp, ustar, zmeas, z0d, z0m, rho_air_d):
    # dimensionless Obukhov stability parameter (Campbell & Norman, 1998)
    L = rho_air_d * cp_air * air_temp * (ustar ** 3) / von_karmen * grav_acc * h_flux
    return - (zmeas - z0d) / L

def corr_factor_momentum(stability):
    # (Campbell & Norman, 1998)
    if stability < 0:
        return -2 * np.log((1 + (1 - 16 * stability)**0.5) / 2.)
    else:
        return 6 * np.log(1 + stability)

def corr_factor_heat(phi_m, stability):
    # (Campbell & Norman, 1998)
    if stability < 0:
        return phi_m / 0.6
    else:
        return phi_m

def friction_velocity(ws, zmeas, z0d, z0m, phi_m):
    # m/s
    z_ = (zmeas - z0d) / z0m
    return von_karmen * ws / (np.log(z_) + phi_m) 

def aerodynamic_cond_0(ws,  ustar):
    # simple eq.
    ga = ustar**2 / ws
    return ga

def aerodynamic_cond_1(ws, h_flux, air_temp, ustar, zmeas, hc, rho_air_d):
    z0m = roughness_length_momentum_transfer(hc)
    z0h = roughness_length_heat_transfer(hc)
    z0d = zero_plane_displacement_h(hc)

    stability = obukov_stability(h_flux, air_temp, ustar, zmeas, z0d, z0m, rho_air_d)
    phi_m = corr_factor_momentum(stability)
    phi_h = corr_factor_heat(phi_m, stability)
    hh = np.log((zmeas - z0d) / z0h) + phi_h
    mm = np.log((zmeas - z0d) / z0m) + phi_m
    ga = ustar ** 2 / ws * mm / hh
    return ga

def aerodynamic_cond_hc(h_flux, air_temp, ustar, z, hc, rho_air_d):
    z0m = roughness_length_momentum_transfer(hc)
    z0h = roughness_length_heat_transfer(hc)
    z0d = zero_plane_displacement_h(hc)

    stability = obukov_stability(h_flux, air_temp, ustar, z, z0d, z0m, rho_air_d)
    phi_m = corr_factor_momentum(stability)
    phi_h = corr_factor_heat(phi_m, stability)
    hh = np.log((z - z0d) / z0h) + phi_h
    ga = von_karmen * ustar / hh
    return ga

def aerodynamic_cond_it(ws, h_flux, air_temp, ustar, zmeas, hc, rho_air_d):
    # (Campbell & Norman, 1998)
    # m/s iterate with flux tower data
    z0m = roughness_length_momentum_transfer(hc)
    z0h = roughness_length_heat_transfer(hc)
    z0d = zero_plane_displacement_h(hc)

    stability = obukov_stability(h_flux, air_temp, ustar, zmeas, z0d, z0m, rho_air_d)
    phi_m = corr_factor_momentum(stability)
    phi_h = corr_factor_heat(phi_m, stability)
    ustar_i = friction_velocity(ws, zmeas, z0d, z0m, phi_m)

    i = 1
    while ((np.abs(ustar - ustar_i) / ustar) > 0.1) and i < 100:
        ustar = ustar_i
        stability = obukov_stability(h_flux, air_temp, ustar, zmeas, z0d, z0m, rho_air_d)
        phi_m = corr_factor_momentum(stability)
        phi_h = corr_factor_heat(phi_m, stability)
        ustar_i = friction_velocity(ws, zmeas, z0d, z0m, phi_m)
        i = i + 1

    hh = np.log((zmeas - z0d) / z0h) + phi_h
    mm = np.log((zmeas - z0d) / z0m) + phi_m
    ga = (ws * von_karmen**2) / (hh * mm)
    return ga

def surface_conductance_PM(atm_pressure, air_temp, vpd, ga, le, h, rho_air_d):
    # m/s PM inversion
    gamma =  psychrometric_cste(atm_pressure, air_temp)
    delta =  vapor_pressure_slope(air_temp)                         
    return gamma * ga * le / (delta * h + rho_air_d * cp_air * ga * vpd  - gamma * le)

def air_vapor_pressure_sat(self, air_temp):
    # Pa
    eStar = 610.8 * np.exp((17.27 * air_temp) / (air_temp + 237.3)) 
    return eStar  

def air_density_dry(air_temp, atm_pressure, RH):
    rd = 287.04  # Gas Constant dry air [J Kg-1 K-1]
    eStar = air_vapor_pressure_sat(air_temp, RH)
    e = eStar * RH / 100.
    rho_air_d = (atm_pressure - e) / ((air_temp + 273.16) * rd)  # [kg/m3]
    return rho_air_d

def leaf_vpd_(le, g_surf, ga, air_temp, rho_air_d):
    # mol mol-1
    G = g_surf * ga / (g_surf + ga)
    return le / (G * Lv(air_temp) * rho_air_d * mwratio)

def leaf_vpd(et, g_surf, ga):
    # mol mol-1
    G = g_surf * ga / (g_surf + ga)
    return et /  G

def cal_psi_leaf_1(psi_s, psi_50, Kmax, hc, T):
    # simple solution a=1
    Kmax = Kmax / hc
    a = 2 * Kmax * psi_50 * (psi_s - hc * grav_acc * rho_w * 10 ** -6)
    b = hc * T * (2 * psi_50 + psi_s)
    d = hc * T + 2 * Kmax * psi_50
    psi_l = (a - b) / d
    psi_l = [pi if pi<0 else 0 for pi in psi_l]
    return psi_l

def cal_psi_leaf(psi_pd, psi_50, Kmax, a, T):
    # non linear solution
    def __residual_T( psi_l_i, psi_pd_i, T_i):
        psi_m = (psi_pd_i + psi_l_i) / 2
        K = Kmax / (1 + (psi_m / psi_50) ** a)
        T_il = K * (psi_pd_i - psi_l_i)
        return np.abs(T_i - T_il)

    psi_l = [optimize.leastsq(__residual_T, psi_pd_i, args=(psi_pd_i, T_i))[0][0]
            for psi_pd_i, T_i in zip(psi_pd, T)]
    psi_l = [pi if pi < 0 else 0 for pi in psi_l]
    return psi_l

def cal_LE(g_surf, ga, rad, vpd_a, atm_pressure, air_temp, rho_air_dry):
    # penman monteith
    G_surf = g_from_mol(g_surf, atm_pressure, air_temp)
    Ga = g_from_mol(ga, atm_pressure, air_temp)
    gamma = psychrometric_cste(atm_pressure, air_temp)
    delta = vapor_pressure_slope(air_temp)
    LE_pm  = (delta * rad + rho_air_dry * cp_air * vpd_a * Ga)\
                / (delta + gamma * (1 + Ga / G_surf))
    return LE_pm

def cal_E(g_surf, g,  ga, vpd_l):
    # mol H20/m2/s
    G = g * ga / (g_surf + ga)
    e =  vpd_l * G
    return e 

if __name__ == "__main__":
    pass