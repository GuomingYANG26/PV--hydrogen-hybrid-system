# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 13:24:36 2022
author: Guoming Yang
School of Electrical Engineering and Automation
Harbin Institute of Technology
email: yangguoming1995@gmail.com

"""
# PV power geneartion modeling considering actual physical process via PVLIB 
import pvlib
import numpy as np
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pvlib.iotools.psm3 import parse_psm3
import warnings
warnings.filterwarnings('ignore')

# The function to calculate the reflection loss
def loss_reflect_abs(aoi, n_glass=1.526, n_ar=1.3, n_air=1):
    '''adapted from section 8 in PVWatts Version 5 Manual'''
    #the angle of refraction into the antireflection coating
    theta_ar = np.rad2deg(np.arcsin(
        (n_air / n_ar) * np.sin(np.deg2rad(aoi))))
    #the transmittance through the antireflection coating 
    tau_ar = (1 - 0.5 * (
        ((np.sin(np.deg2rad(theta_ar - aoi))) ** 2)
        / ((np.sin(np.deg2rad(theta_ar + aoi))) ** 2))
        + ((np.tan(np.deg2rad(theta_ar - aoi))) ** 2)
        / ((np.tan(np.deg2rad(theta_ar + aoi))) ** 2))
    #the angle of refraction into the glass cover
    theta_glass = np.rad2deg(np.arcsin(
        (n_ar / n_glass) * np.sin(np.deg2rad(theta_ar))))
    #the transmittance through the glass cover
    tau_glass = (1 - 0.5 * (
        ((np.sin(np.deg2rad(theta_glass - theta_ar))) ** 2)
        / ((np.sin(np.deg2rad(theta_glass + theta_ar))) ** 2))
        + ((np.tan(np.deg2rad(theta_glass - theta_ar))) ** 2)
        / ((np.tan(np.deg2rad(theta_glass + theta_ar))) ** 2))
    #the effective transmittance
    tau_total = tau_ar * tau_glass
    return tau_total
#reflection losses for the direct beam irradiance
def loss_reflect(aoi, n_glass=1.526, n_ar=1.3, n_air=1):
    out = (loss_reflect_abs(aoi, n_glass, n_ar, n_air)
        / loss_reflect_abs(1e-6, n_glass, n_ar, n_air))
    out = out.fillna(0)
    return out


#reads the TMY data downloaded from NSRDB for Harbin  
with open('D:/Doctor/paper/first2022/code/edition3/harbin_tmy_data.csv', 'r') as f:
    data, metadata =  pvlib.iotools.psm3.parse_psm3(f,map_variables=True)
lat = metadata['latitude']
lon = metadata['longitude']
times = data.index

 

#acquires the information relating to the solar position
#method:nrel_numpy(default),nrel_numba,pyephem,ephemeris,nrel_c
solpos = pvlib.solarposition.get_solarposition(
    times, 
    lat, 
    lon,
    altitude=np.mean(pvlib.atmosphere.pres2alt(data['pressure'])),
    pressure=data['pressure'],
    method='nrel_numpy',
    temperature=data['temp_air'])



#The extraterrestrial irradiation present in watts per square meter
#dafault solar_constant=1366.1,method: nrel,pyephem,spencer,asce
ETI = pvlib.irradiance.get_extra_radiation(
    times, 
    solar_constant=1361.1, 
    method='nrel')
#Air mass is a relative measure of the optical length of the atmosphere. 
#method: kastenyoung1989(default),simple,kasten1966,youngirvine1967,gueymard1993,young1994,pickering2002
airmass = pvlib.atmosphere.get_relative_airmass(
    solpos['apparent_zenith'], 
    model='kastenyoung1989')
#Determines absolute (pressure-adjusted) airmass from relative airmass and pressure.
airmass_abs = pvlib.atmosphere.get_absolute_airmass(airmass, data['pressure'])



#gets the surface tilt and surface azimuth, systemtype == 'fixed':
axis_tilt = 46
axis_azimuth = 175
surface_tilt = axis_tilt
surface_azimuth = axis_azimuth
#Calculates the angle of incidence of the solar vector on a surface. 
aoi = pvlib.irradiance.aoi(
    surface_tilt, 
    surface_azimuth,
    solpos['apparent_zenith'], 
    solpos['azimuth'])
#Determines total in-plane irradiance and its beam, sky diffuse and ground reflected components
#model:Can be one of 'isotropic', 'klucher', 'haydavies', 'reindl', 'king', 'perez'. model_perez:Used only if model='perez'.
#Models 'haydavies', 'reindl', or 'perez' require ETI.
#The 'perez' model requires relative airmass (airmass) as input. 
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt, 
    surface_azimuth, 
    solpos['apparent_zenith'],
    solpos['azimuth'], 
    data['dni'], 
    data['ghi'], 
    data['dhi'], 
    dni_extra=ETI, 
    airmass=airmass,
    albedo=data['albedo'] , 
    model='perez', 
    model_perez='allsitescomposite1990')



#a reduction in the POA irradiance due to dust and dirt on the module surface
loss_soiling = 0.02
#as for external shading, poa_direct and poa_sky_diffuse does need to be multiplied by a coefficient，while poa_ground_diffuse does not. In this paper, ther are all assumed to zero.
loss_external_shading = 0.0    
poa_irradiance = poa_irradiance * (1 - loss_soiling) * (1 - loss_external_shading)
##self-shading have three irradiance loss factors, which reprensent the irradiance loss level for poa_direct, poa_sky_diffuse and poa_ground_diffuse
#Sky diffuse self-shading factor, Ground-reflected diffuse self-shading, Linear self-shading factor 



#lists some module specification  
#calculates the cell temperature   
#Yingli_module_Imp = 8.39
#Yingli_module_Vmp = 29.8
#Yingli_module_length = 1.640
#Yingli_module_width =  .99
#Yingli_short_circuit_current = 8.92
#Yingli_temperature_coefficient_short_circuit_current = 0.05  #unit %/C
#Yingli_module_efficiency = Yingli_module_Imp * Yingli_module_Vmp / Yingli_module_length / Yingli_module_width /1000
Yingli_module_efficiency = .153
#Determine the cell temperature
params = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_polymer']
cell_temp = pvlib.temperature.sapm_cell(
    poa_global=poa_irradiance['poa_global'],
    temp_air=data['temp_air'],
    wind_speed=data['wind_speed'],
    **params)

#an AOI correction F2 to adjust the direct beam irradiance to account for reflection losses
#refraction indexes for the glass, antireflection coating and air
n_glass = 1.526 
n_ar = 1.3
n_air = 1
F21 = loss_reflect(aoi, n_glass, n_ar, n_air)
F21[np.where(F21>1)[0]] = 1
#Determines the incidence angle modifiers for diffuse sky and ground-reflected irradiance
F22 = pvlib.iam.martin_ruiz_diffuse(surface_tilt)
poa_irradiance['poa_direct'] = F21 * poa_irradiance['poa_direct']
poa_irradiance['poa_sky_diffuse'] =  F22[0] * poa_irradiance['poa_sky_diffuse']
poa_irradiance['poa_ground_diffuse'] =  F22[1] * poa_irradiance['poa_ground_diffuse']
poa_irradiance['poa_global'] = poa_irradiance['poa_direct'] + poa_irradiance['poa_sky_diffuse'] + poa_irradiance['poa_ground_diffuse']
poa_irradiance.fillna(0, inplace=True)



#Calculates the SAPM spectral loss coefficient, F1.
Yingli_module_type='multisi'
#Spectral correction based on absolute airmass and precipitable water
F1 = pvlib.atmosphere.first_solar_spectral_correction(
    data['precipitable_water'], 
    airmass_abs, 
    Yingli_module_type)
F1.fillna(0, inplace=True)
poa_irradiance['poa_global'] = F1 * poa_irradiance['poa_global']

'''
#计算单串光伏组件数
N_mod =  950 / 37.6 / (1 - 0.32/100 * ( cell_temp.min()- 25) )
print(N_mod)

N_mod =  605 / 37.6 / (1 - 0.32/100 * ( cell_temp.max()- 25) )
print(N_mod)
#计算串联数
N_str = 1000000/250/20
print(N_str)
'''

#module information
cec_modules = pvlib.pvsystem.retrieve_sam('cecmod')
module = cec_modules['Yingli_Energy__China__YL250P_29b']
#Calculates five parameter values for the single diode equation 
IL, I0, Rs, Rsh, nNsVth = pvlib.pvsystem.calcparams_cec(
    effective_irradiance=poa_irradiance['poa_global'],
    temp_cell=cell_temp,
#    alpha_sc=Yingli_temperature_coefficient_short_circuit_current/100*Yingli_short_circuit_current, #unit: A/K
    alpha_sc=0.00385,    #unit: A/K   
    a_ref=1.585228,     #unit: V
    I_L_ref=8.798402,   #unit: A
    I_o_ref=2.63E-10,   #unit: A
    R_sh_ref=432.474701,#unit: Ohm
    R_s=0.413368,       #unit: Ohm
    Adjust=5.836602,    #unit: %
    EgRef=1.121,        #unit: eV
    dEgdT=-0.0002677)   #unit:1/K
#Solve the single-diode equation to obtain the maximum power in watts.
curve_info = pvlib.pvsystem.singlediode(
    photocurrent=IL,
    saturation_current=I0,
    resistance_series=Rs,
    resistance_shunt=Rsh,
    nNsVth=nNsVth,
    ivcurve_pnts=100,
    method='lambertw')

#Scales the voltage, current, and power 
modules_per_string = 20        
strings_per_inverter = 200
dc =pd.DataFrame(columns=['p_mp'], data=curve_info['p_mp'] * modules_per_string * strings_per_inverter )
dc['v_mp'] = curve_info['v_mp'] * modules_per_string



#DC snow losses
dc_snow_losses = 0.03
dc['p_mp'] = dc['p_mp'] * (1 - dc_snow_losses)
#DC electrical losses(mismatch:2%, wiring:2%, connections:0.5, nameplate rating 1.5, others: 3)
dc_electrical_losses = 1- (1-0.02) * (1-0.02) * (1-0.005) * (1-0.005) * (1-0.03)
dc['p_mp'] = dc['p_mp'] * (1 - dc_electrical_losses) 



#inverter information
cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
#inverter = sapm_inverters['TMEIC__PVL_L0833GR']   # acceesing the database when konwing the exact the inverter name: way 1
inverter_true_mode = cec_inverters.T.index.str.startswith('TMEIC') & cec_inverters.T.index.str.contains('PVL')  & cec_inverters.T.index.str.contains('L0833GR')
inverter = cec_inverters.T[inverter_true_mode].T.iloc[:,0]
#AC power output
ac = pvlib.inverter.sandia(
    dc['v_mp'], 
    dc['p_mp'], 
    inverter)

ac = ac.fillna((0))
#plots the PV simulated by the physical model chain
pv_module_chain = ac[0:8760]
pv_module_chain[pv_module_chain<0] = 0
pv_module_chain.index = range(8760)
sns.set_theme(style="darkgrid")
plt.figure(dpi=100, figsize=(12,6))
sns.lineplot(data=pv_module_chain)
print(pv_module_chain.mean())
pv_module_chain.to_csv(r'D:\Doctor\paper\first2022\code\edition3\pv_module_chain.csv', index=False)

####----------------Code to get the data needed for drawing heat map---------------------
pv_module_chain = pd.read_csv(r'D:\Doctor\paper\first2022\code\edition3\pv_module_chain.csv')
pv_module_chain = pv_module_chain.iloc[:,0].values
pv_module_chain_reshape = pv_module_chain.reshape(24,-1,order='F').T/1000000
#pd.DataFrame(pv_module_chain_reshape).to_csv(r'D:\Doctor\paper\first2022\pv_module_chain_reshape.csv')
pv_module_chain_daily = np.sum(pv_module_chain_reshape, axis=1)
'''
#plots the traditional PV
yingli_temperature_coefficient_of_maximum_power = -0.42/100  #unit:%/C
yingli_maximum_power = 250  #unit:W
yingli_T_noct = 46 #unit:C
inverter_efficiency = 0.975 #https://www.energysage.com/solar-inverters/tmeic/2866/pvl-l0833gr/

#poa_irradiance['poa_global']
cell_temp_traditional = data['temp_air'].values + poa_irradiance['poa_global'].values/800 * (yingli_T_noct - 20)
pv_tradional = yingli_maximum_power * modules_per_string * strings_per_inverter * poa_irradiance['poa_global'].values\
 /1000 * (1 + yingli_temperature_coefficient_of_maximum_power * (cell_temp_traditional - 25) ) * inverter_efficiency
pv_tradional[pv_tradional<0] = 0
pv_tradional[pv_tradional>833000] = 833000
sns.set_theme(style="darkgrid")
plt.figure(dpi=100, figsize=(12,6))
sns.lineplot(data=pv_tradional)

print((pv_tradional.mean()-pv_module_chain.mean())/pv_module_chain.mean())
pv_tradional = pd.Series(pv_tradional)
pv_tradional.to_csv(r'D:\Doctor\paper\first2022\code\edition3\pv_tradional.csv', index=False)
'''