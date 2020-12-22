rho_air_c = 1.225                     # kg/m3         air density  
cp_air = 1005                         # J/kg/k        specific heat of air
mwratio = 0.622                       # -             ratio molecular weight of water vapor/dry air
von_karmen = 0.41                     # -             Von Karmen constant
grav_acc = 9.81                       # m/s2          gravitational acceleration
Rd = 8.314                            # m3 Pa/K/mol   dry air gas constant
Mw = 18.01528 / 1000                  # kg/mol        Molar mass of water
rho_w = 1000                          # kg/m3         water density   
Cao = 210 * 10 ** (-3)                # [mol/mol]     atmospheric O2
Kc_25C = 404.9 * 10 ** (-6)           # [mol mol-1]   Michelis-Menten constants for CO2 inhibition at 25 C
Ko_25C = 278 * 10 ** (-3)             # [mol mol-1]   Michelis-Menten constants for O2 inhibition at 25 C
gamma_star_25C = 42.75 * 10 ** (-6)   # [mol mol-1]   CO2 compensation point at 25 C C3 plants
phi_PSII = 0.7						  # [-]           Quantum yield of photosynthesis II
Hd = 200 * 1000 					  # [/mol]        rate of peaked function decrease above optimum (Medlyn et al. 2002)


if __name__ == "__main__":
    pass