# -*- coding: utf-8 -*-
"""
Created on Wed Jun 29 08:44:46 2022
author: Guoming Yang
School of Electrical Engineering and Automation
Harbin Institute of Technology
email: yangguoming1995@gmail.com
"""
import pvlib
import numpy as np
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pvlib.iotools.psm3 import parse_psm3
import warnings
warnings.filterwarnings('ignore')

pv_module_chain = pd.read_csv(r'D:\Doctor\paper\first2022\code\edition3\pv_module_chain.csv')
pv_module_chain = pv_module_chain.iloc[:,0].values


#reads the TMY data downloaded from NSRDB for Harbin  
with open('D:/Doctor/paper/first2022/code/edition3/harbin_tmy_data.csv', 'r') as f:
    data, metadata =  pvlib.iotools.psm3.parse_psm3(f,map_variables=True)
lat = metadata['latitude']
lon = metadata['longitude']
times = data.index

 

#acquires the information relating to the solar position
solpos = pvlib.solarposition.get_solarposition(
    times, 
    lat, 
    lon,
    temperature=data['temp_air'])

#gets the surface tilt and surface azimuth, systemtype == 'fixed':
axis_tilt = 46
axis_azimuth = 175
surface_tilt = axis_tilt
surface_azimuth = axis_azimuth

#Determines total in-plane irradiance and its beam, sky diffuse and ground reflected components using the default model:'isotropic'
poa_irradiance = pvlib.irradiance.get_total_irradiance(
    surface_tilt, 
    surface_azimuth, 
    solpos['apparent_zenith'],
    solpos['azimuth'], 
    data['dni'], 
    data['ghi'], 
    data['dhi'])
poa_irradiance.fillna(0, inplace=True)

#plots the traditional PV
yingli_temperature_coefficient_of_maximum_power = -0.42/100  #unit:%/C
yingli_maximum_power = 250  #unit:W
yingli_T_noct = 46 #unit:C
inverter_efficiency = 0.975 #https://www.energysage.com/solar-inverters/tmeic/2866/pvl-l0833gr/
modules_per_string = 20        
strings_per_inverter = 200
#poa_irradiance['poa_global']
cell_temp_traditional = data['temp_air'].values + poa_irradiance['poa_global'].values/800 * (yingli_T_noct - 20)
pv_tradional = yingli_maximum_power * modules_per_string * strings_per_inverter * poa_irradiance['poa_global'].values\
 /1000 * (1 + yingli_temperature_coefficient_of_maximum_power * (cell_temp_traditional - 25) ) * inverter_efficiency
pv_tradional[pv_tradional<0] = 0
pv_tradional[pv_tradional>833000] = 833000
sns.set_theme(style="darkgrid")
plt.figure(dpi=100, figsize=(12,6))
sns.lineplot(data=pv_tradional)
print(pv_tradional.mean())
pv_tradional = pd.Series(pv_tradional)
pv_tradional.to_csv(r'D:\Doctor\paper\first2022\code\edition3\pv_tradional.csv', index=False)



####----------------Code to get the data needed for drawing heat map---------------------
pv_tradional = pd.read_csv(r'D:\Doctor\paper\first2022\code\edition3\pv_tradional.csv')
pv_tradional = pv_tradional.iloc[:,0].values
pv_tradional_reshape  = pv_tradional.reshape(24,-1,order='F').T/1000000
#pd.DataFrame(pv_tradional_reshape).to_csv(r'D:\Doctor\paper\first2022\pv_tradional_reshape.csv')
pv_tradional_daily = np.sum(pv_tradional_reshape, axis=1)


'''
#plots the PV simulated by the physical model chain
pv_module_chain[pv_module_chain<0] = 0
sns.set_theme(style="darkgrid")
plt.figure(dpi=100, figsize=(12,6))
sns.lineplot(data=pv_module_chain)
print(pv_module_chain.mean())
print(pv_module_chain_daily.mean())

print((pv_tradional.mean()-pv_module_chain.mean())/pv_module_chain.mean())
'''
