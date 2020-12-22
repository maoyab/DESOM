# DESOM

Diagnostics of ecosystem-scale stomatal optimization models. 

**Description**

This repository contains scripts in the python computing language necessary to repeat analysis in:

Bassiouni M. and G. Vico, *Parsimony versus predictive and functional performance of three stomatal optimization principles in a big-leaf framework, in revision*.

The following scripts manage data (get_metadata_sites.py, get_data_sites.py), calibrate models (fit_gs_params.py), and generate results (process_fit_gs_results.py) represented in the manuscript.

The ipython notebook (NP_figures.ipynb) executes scripts in the repository to analyze data or model output and recreate all figures in the manuscript.

The FLUXNET2015 data products used in this study are available at http://fluxnet.fluxdata.org/data/fluxnet2015-dataset/.

**Summary**

- Stomatal optimization models can improve estimates of water and carbon fluxes with relatively low complexity, yet there is no consensus which formulations are most appropriate for generalized ecosystem-scale applications. We implemented three existing analytical equations for stomatal conductance, based on different water penalty functions, in a big-leaf framework, and determined which optimization principles are most consistent with flux tower observations from a range of biomes. 

- We used information theory to dissect controls of soil water supply and atmospheric demand on evapotranspiration in wet to dry conditions and quantified predictive and functional accuracy of model variants. We ranked stomatal optimization models based on parameter uncertainty, parsimony, predictive performance, and functional accuracy of the interactions between soil moisture, vapor pressure deficit, and evapotranspiration. 

- Performance was high for all model variants. Stomatal optimization based on water use efficiency provided more information about ecosystem-scale evapotranspiration compared to those based on xylem vulnerability. The latter did not substantially improve predictive or functional accuracy and parameterizations were more uncertain. 

- Water penalty functions with explicit representation of plant hydraulics are less useful in improving ecosystem-scale evapotranspiration estimates than those based on water use efficiency, despite having stronger mechanistic underpinning and theoretically desirable qualities at the plant level.


