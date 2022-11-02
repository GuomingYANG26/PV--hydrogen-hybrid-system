# -*- coding: utf-8 -*-
"""
@author: YANG Guoming
"""

# PV power geneartion modeling considering actual physical process via PVLIB 
import pvlib
import numpy as np
from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from pvlib.iotools.psm3 import parse_psm3
import dill
import warnings
warnings.filterwarnings('ignore')

'''
#reads the TMY data downloaded from NSRDB for Harbin  
with open('D:/Doctor/paper/first2022/code/edition3/harbin_tmy_data.csv', 'r') as f:
    data, metadata =  pvlib.iotools.psm3.parse_psm3(f, map_variables=True)
lat = metadata['latitude']
lon = metadata['longitude']
times = data.index

def surface_orientaion (axis_tilt, axis_azimuth):
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
    #----------------------change-------------------------
    axis_tilt = axis_tilt
    axis_azimuth = axis_azimuth
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
    #Yingli_module_efficiency = .153
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
    #DC electrical losses(mismatch:2%, wiring:2%, connections:0.5%, nameplate rating 1.5%, others: 3%)
    dc_electrical_losses = 1- (1-0.02) * (1-0.02) * (1-0.005) * (1-0.005) * (1-0.03)
    dc['p_mp'] = dc['p_mp'] * (1 - dc_electrical_losses) 



    #inverter information
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    #inverter = sapm_inverters['TMEIC__PVL_L0833GR']   # acceesing the database when konwing the exact the inverter name: way 1
    inverter_true_mode = sapm_inverters.T.index.str.startswith('TMEIC') & sapm_inverters.T.index.str.contains('PVL')  & sapm_inverters.T.index.str.contains('L0833GR')
    inverter = sapm_inverters.T[inverter_true_mode].T.iloc[:,0]
    #AC power output
    ac = pvlib.inverter.sandia(
        dc['v_mp'], 
        dc['p_mp'], 
        inverter)
    ac = ac.fillna((0))
    return sum(ac)/8760/1000000*100

def optimize_capacity_factor (tilt, azimuth):
    cf               = []       #光伏板不同方位时的光伏电站的容量因子
    max_cf           = 0        #最大容量因子值
    max_axis_tilt    = 0        #最大容量因子对应的光伏板倾角
    max_axis_azimuth = 0        #最大容量因子对应的光伏板朝向
    for axis_tilt in tilt:
        print(axis_tilt)
        for axis_azimuth in azimuth:
            cf0 = surface_orientaion(axis_tilt, axis_azimuth)   #调用surface_orientaion函数求解光伏板在不同位置时光伏电站对应的容量因子
            cf.append(cf0)   
            if cf0 >= max_cf:    #判断光伏板当前位置下光伏电站的容量因子是否最大，若是则储存对应数据
                max_cf = cf0
                max_axis_tilt = axis_tilt
                max_axis_azimuth = axis_azimuth
    #将cf,tilt,azimuths重排成矩阵形式
    cf = np.reshape(cf, [len(tilt), len(azimuth)])
    tilts = tilt
    for i in range(len(azimuth)-1):
        tilts = np.column_stack((tilts,tilt))
    azimuths = azimuth
    for i in range(len(tilt)-1):
        azimuths = np.column_stack((azimuths,azimuth))
    azimuths = azimuths.T
    return (max_axis_tilt, max_axis_azimuth, max_cf, tilts, azimuths, cf)


#程序正式开始，定义tilt,azimuth的范围
tilt = np.arange(0, 91, 1)
azimuth =np.arange (90, 271, 1)
(max_axis_tilt, max_axis_azimuth, max_cf, tilts, azimuths, cf) = optimize_capacity_factor (tilt, azimuth)



# 保存运行的全部数据
transposition_filename = 'D:/Doctor/paper/first2022/code/edition3/contour_map.pkl'
dill.dump_session(transposition_filename)

'''
#导入保存的数据
import matplotlib.pyplot as plt
import dill
transposition_filename = 'D:/Doctor/paper/first2022/code/edition3/contour_map.pkl'
dill.load_session(transposition_filename)

#画图
plt.close()
#85mm对应85/25.4inch，额外×1.2是考虑保存时图片去除空白后图片仍为85mm，这个参数需要手动调
f, ax = plt.subplots(1, 1, figsize=(85/25.4*1.4,85/25.4*1.5/3*1.46), dpi=600,
                     subplot_kw=dict(projection='polar'))
plt.rc('font',family='Times New Roman', size=7)
#plt.rcParams['font.sans-serif'] = ['Times New Roman'] 
latitude = lat
### CF
revenues = cf
opt_tilt = max_axis_tilt
opt_azimuth = max_axis_azimuth
### Contour
#pp = ax.contourf(np.radians(azimuths), tilts, revenues,cmap=plt.get_cmap('YlGnBu_r'))
pp = ax.contourf(np.radians(azimuths), tilts, revenues)
### Max point
ax.scatter(np.radians(opt_azimuth), opt_tilt, c='r', s=10, marker='*')
### Latitude line
ax.plot(np.radians(np.linspace(90,270,100)),     
        latitude * np.ones(100),
        c='k', lw=1.5, dashes=[3,3])
### Axis format
ax.set_theta_direction(-1)
ax.set_theta_zero_location('N')
ax.set_thetamin(270)

ax.set_thetamax(90)
ax.set_xticks(np.radians(np.linspace(90, 270, 7))) 
ax.set_yticks(np.linspace(0, 90, 7))  
ax.set_yticklabels(['{}°'.format(int(i)) for i in np.linspace(0, 90, 7)])   
cbar = plt.colorbar(pp, orientation='vertical', shrink=0.5, pad=0.08, aspect=10, ax=ax)
cbar.ax.tick_params(labelsize='medium')

ax.tick_params(axis='both', pad=.5, labelsize='medium')
ax.scatter(np.radians(opt_azimuth), opt_tilt, c='r', s=20, marker='*', zorder=3)

#plt.tight_layout()
plt.title("Capacity factor (%)", fontsize=7, y=0.82, x=0.5)

plt.show()
plt.savefig("D:/Doctor/paper/first2022/code/edition3/contourmap.pdf", bbox_inches='tight', pad_inches = 0)
#plt.savefig("D:/Doctor/paper/first2022/code/edition3/contourmap.pdf", bbox_inches='tight')








