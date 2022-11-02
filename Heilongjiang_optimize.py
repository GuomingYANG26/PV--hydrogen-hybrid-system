# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 10:20:13 2022
author: Guoming Yang
School of Electrical Engineering and Automation
Harbin Institute of Technology
email: yangguoming1995@gmail.com
"""
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import pvlib
#--------------------------data input----------------------------------
#1€=1.05$  1RMB=0.15$
euro_to_dollar = 1.05 #exchange rate of Euro against US dollar
RMB_to_dollar = 0.15  #exchange rate of RMB against US dollar
N_variable = 8760     #Number of hours per year
#electricity price
electricity_price = pd.read_csv(r'D:\Doctor\paper\first2022\2020_electriciy_price.csv')
electricity_price = electricity_price.iloc[:,0].values
electricity_price = euro_to_dollar*electricity_price/1000
#PV power
#model chain
#model_chain_pv = pd.read_csv(r'D:\Doctor\paper\first2022\pv_module_chain.csv')
#conventional pv
#model_chain_pv = pd.read_csv(r'D:\Doctor\paper\first2022\pv_tradional.csv')
#pv_modeling = model_chain_pv.iloc[:,0].values/1000
hydrogen_price = 9   #hydrogen sale price
lambda_air1 = 1      #the support level for PV from government
lambda_air2 = 1      #the support level for hydrogen from government
sigma_HFV = 110       #fuel economy of HFC

#emission amounts and penalty coefficients of the pollutants emitted by coal-fired thermal power generators
mu_co2, mu_so2, mu_nox0 = 86.4725/1000, 3.9446/1000, 3.09383/1000
delta_co2, delta_so2, delta_nox = 0.0035, 0.923, 1.23
#emission amounts and penalty coefficients of the pollutants emitted by gasoline-fueled vehicles
mu_co, mu_nox1, mu_hc = 1000/1000/1000, 60/1000/1000, 170/1000/1000
delta_co, delta_nox, delta_hc = .07, 1.23, .3895

rated_power_of_unit_compressor = 5   #unit: kW
#The economic and technical parameters of each component, including the unit cost, the ratio of O&M cost, lifetime and efficiency of compressors, transformer, electrolyzers, hydrogen tanks and HFCs
compressor_invest, compressor_OM_ratio, compressor_lifetime = 712*rated_power_of_unit_compressor, 0.01, 20              #integer,5 denotes the rated power of a single compressor
trans_invest, trans_OM_ratio, trans_lifetime, trans_efficiency = 400*euro_to_dollar, 0.05, 20, 0.98
electrolyzer_invest, electrolyzer_OM_ratio, electrolyzer_lifetime, electrolyzer_efficiency = 1182, 0.05, 8, 0.73
tanks_invest, tanks_OM_ratio, tanks_lifetime = 295*euro_to_dollar, 0.01, 20              #note that the unit is $/kg, m^3
hydrogen_fuel_cell_invest, hydrogen_fuel_cell_OM_ratio, hydrogen_fuel_cell_lifetime, hydrogen_fuel_cell_efficiency = 46, 0.05, 5, 0.47

tao = 0.10    #discount rate
standar_pressure = 1                #standard atmospheric pressure
hydrogen_high_heating_value = 39    #hydrogen high heating value
compressor_pressure = 200           #working pressure of compressor
compressor_pressure_ref = 350       #reference working pressure of compressor
compressor_power_comsumption_rate = 2.1     #the hourly power consumption when compressors compress per kilogram of hydrogen under standard working pressure
utilization_rate = 0.95       #PV power utilization rate
mass_volume_fraction_H2 = 30 #kg/m^3 mass volume fraction of hydrogen0.03kg/L
delt_time = 1 #time interval
#PV plant
module_price, module_OM_ratio, module_lifetime, module_capacity = 0.37, 0.02, 25, 1000000         #note that the unit is $/W.  unit cost, the ratio of O&M, lifetime, rated power of PV modules
inverter_price, inverter_OM_ratio, inverter_lifetime, inverter_capacity = 0.07, 0.02, 25, 833000   #note that the unit is $/W.  unit cost, the ratio of O&M, lifetime, rated power of inverter


xi_module = tao*(1+tao)**module_lifetime/((1+tao)**module_lifetime - 1)
xi_inverter = tao*(1+tao)**inverter_lifetime/((1+tao)**inverter_lifetime - 1)
xi_compressor = tao*(1+tao)**compressor_lifetime /((1+tao)**compressor_lifetime  - 1)
xi_trans = tao*(1+tao)**trans_lifetime/((1+tao)**trans_lifetime - 1)
xi_electrolyzer = tao*(1+tao)**electrolyzer_lifetime/((1+tao)**electrolyzer_lifetime - 1)
xi_tanks = tao*(1+tao)**tanks_lifetime/((1+tao)**tanks_lifetime - 1)
xi_hydrogen_fuel_cell = tao*(1+tao)**hydrogen_fuel_cell_lifetime/((1+tao)**hydrogen_fuel_cell_lifetime - 1)


def Heilongjiang_pv(data_path, in_path):

    
    #PV power
    #model chain
    model_chain_pv = pd.read_csv(data_path)
    pv_modeling = model_chain_pv.iloc[:,0].values/1000
    with open(in_path, 'r') as f:
        data, metadata =  pvlib.iotools.psm3.parse_psm3(f,map_variables=True)
    ghi = data['ghi'].mean()
    # --------------------------------construct model------------------------------------
    photo_hydrogen=gp.Model("photo_hydrogen")

    #--------------------------------define variables--------------------------------------
    N1 = photo_hydrogen.addVar(vtype=GRB.INTEGER, name='number of the compressor')
    X2 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS, name='capacity of the transformer')
    X3 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS, name='capacity of the electrolyzer')
    X4 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS, name='capacity of the hydrogen tanks')
    X5 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY,vtype=GRB.CONTINUOUS, name='capacity of the hydrogen fuel cell')
    Demand_elec = {}
    P_elec = {}
    P_comp = {}
    V_tanks = {}
    Demand_sale = {}
    Demand_fuel_cell = {}
    P_fuel_cell = {}
    P_pv_net = {}
    P_pv_curt = {}
    for i in range(N_variable):
        Demand_elec[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='hydrogen production of the electrolyzer')
        P_elec[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='power consumption of the electrolyzer')
        P_comp[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='power consumption of the compressor')
        V_tanks[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='hydrogen storage level of the hydrogen tanks')
        Demand_sale[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='the amount of hydrogen directly sold to the hydrogen market')
        Demand_fuel_cell[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='the amount of hydrogen for regeneration')
        P_fuel_cell[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='power generated by the hydrogen fuel cell')
        P_pv_net[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='grid-connected power of the PV plant')
        P_pv_curt[i] = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='curtailment power of the PV plant')



    R1 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='hydrogen sale revenue')
    R2 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='electricity sale revenue')
    R3 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='environmental benefits')
    C1 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='annual investment cost')
    C2 = photo_hydrogen.addVar(lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name='annual O&M cost')

    # -----------------------------Constraints---------------------------------------
    #constraints of electrolyzers
    photo_hydrogen.addConstrs((Demand_elec[t] == P_elec[t]*electrolyzer_efficiency/hydrogen_high_heating_value for t in range(N_variable)), \
                              name='operation constraint of the electrolyzer 1')
    photo_hydrogen.addConstrs((P_elec[t] <= X3 for t in range(N_variable)), name='operation constraint of the electrolyzer 2')

    #constraints of compressors
    photo_hydrogen.addConstrs((P_comp[t] == compressor_power_comsumption_rate * Demand_elec[t] * math.log(compressor_pressure/standar_pressure) / math.log(compressor_pressure_ref/standar_pressure)\
                               for t in range(N_variable)), name='operation constraint of the compressor 1')
    photo_hydrogen.addConstrs((P_comp[t] <= N1*rated_power_of_unit_compressor for t in range(N_variable)), name='operation constraint of the compressor 2')    

    #constraints of hydrogen tanks
    photo_hydrogen.addConstrs((mass_volume_fraction_H2*V_tanks[t+1] == mass_volume_fraction_H2*V_tanks[t] + delt_time * (Demand_elec[t] - Demand_sale[t] - Demand_fuel_cell[t]) \
                               for t in range(N_variable-1)), name='hydrogen balance of the hydrogen tanks')
    photo_hydrogen.addConstrs((V_tanks[t] <= X4 for t in range(N_variable)), name='operation constraint of the hydrogen tanks')    
    photo_hydrogen.addConstr((V_tanks[0] == 0.5*X4), name='initial hydrogen storage level')
    photo_hydrogen.addConstr(mass_volume_fraction_H2*V_tanks[0] == mass_volume_fraction_H2*V_tanks[8759] + delt_time * (Demand_elec[8759] - Demand_sale[8759] - Demand_fuel_cell[8759]), name='terminal hydrogen storage level') 

    #constraints of hydrogen fuel cells
    photo_hydrogen.addConstrs((P_fuel_cell[t] == Demand_fuel_cell[t] * hydrogen_fuel_cell_efficiency * hydrogen_high_heating_value \
                               for t in range(N_variable)), name='operation constraint of the hydrogen fuel cell 1')   
    photo_hydrogen.addConstrs((P_fuel_cell[t] <= X5 for t in range(N_variable)), name='operation constraint of the hydrogen fuel cell 2') 

    #constraint of the PV plant
    photo_hydrogen.addConstrs((P_pv_net[t] + P_pv_curt[t] + P_elec[t] + P_comp[t] == pv_modeling[t] \
                               for t in range(N_variable)), name='operation constraint of the PV plants') 
    #grid-connected power constraint
    photo_hydrogen.addConstrs((P_pv_net[t] + P_fuel_cell[t] <= X2 for t in range(N_variable)), name='Grid-connected power constraint') 

    #constraint of the PV power utilization rate
    photo_hydrogen.addConstr((sum(P_pv_curt[t] for t in range(N_variable)) <= (1 - utilization_rate) * sum(pv_modeling[t] for t in range(N_variable))), \
                              name='utilization rate of the PV power') 
        
    # --------------------------Objective function--------------------------------
    #annual hydrogen sale revenue
    photo_hydrogen.addConstr((R1 == sum(hydrogen_price*Demand_sale[t] for t in range(N_variable))), name='revenue from hydrogen sale')
    #annual electricity sale revnue
    photo_hydrogen.addConstr((R2 == sum(trans_efficiency * electricity_price[t] * (P_pv_net[t] + P_fuel_cell[t]) \
                                        for t in range(N_variable))), name='revenue from electricity sale')
    #annual environmental benefits
    photo_hydrogen.addConstr((R3 == sum(lambda_air1 * trans_efficiency * (P_pv_net[t] + P_fuel_cell[t]) * (mu_co2*delta_co2+mu_so2*delta_so2+mu_nox0*delta_nox) for t in range(N_variable))\
                                        + sum(lambda_air2 * Demand_sale[t] * sigma_HFV *  (mu_co*delta_co + mu_nox1*delta_nox + mu_hc*delta_hc) for t in range(N_variable))), name='environmental revenue')

    #equivalent annual investment cost

    photo_hydrogen.addConstr((C1 == xi_module*module_price*module_capacity + xi_inverter*inverter_price*inverter_capacity \
                              + xi_compressor*compressor_invest*N1 + xi_trans*trans_invest*X2 \
                              + xi_electrolyzer*electrolyzer_invest*X3 + xi_tanks*tanks_invest*X4 \
                              + xi_hydrogen_fuel_cell*hydrogen_fuel_cell_invest*X5), name='investment cost of the components')
        
    #annual operation and maintenance cost   
    photo_hydrogen.addConstr((C2 == module_OM_ratio*module_price*module_capacity + inverter_OM_ratio*inverter_price*inverter_capacity \
                              + compressor_OM_ratio*compressor_invest*N1 + trans_OM_ratio*trans_invest*X2 \
                              + electrolyzer_OM_ratio*electrolyzer_invest*X3 + tanks_OM_ratio*tanks_invest*X4 \
                              + hydrogen_fuel_cell_OM_ratio*hydrogen_fuel_cell_invest*X5), name='O&M cost of the components')    

    photo_hydrogen.setObjective(R1+R2+R3-C1-C2, GRB.MAXIMIZE)

    # Optimize model
    photo_hydrogen.optimize()
    obj = photo_hydrogen.objVal
    return obj, X3.x, X2.x, ghi
#the municipality in the heilongjiang province
total_city = ["harbin","qitaihe","daxinganling","heihe","qiqihar","daqing","suihua","yichun","mudanjiang",\
        "shuangyashan","jiamusi","jixi","hegang"]
parent_directory = 'D:/Doctor/paper/first2022/data'
heilongjiang_obj = pd.DataFrame()
heilongjiang_electrolyzer = pd.DataFrame()
heilongjiang_transformer = pd.DataFrame()
heilongjiang_ghi = pd.DataFrame()
 


#calling the function "Heilongjiang_pv" to get the results under different municipality
for i in range(len(total_city)):
    city = total_city[i]
    data_path = parent_directory + str('/') + str('pv') + str('/') + city + str('_sapm_pv_modeling')  + str('.csv')
    in_path = parent_directory + str('/') + str('tmy') + str('/') + city + str('_tmy_data')  + str('.csv')
    print('---------------------------------------------------')
    print(city)
    heilongjiang_obj.loc[0, city],  heilongjiang_electrolyzer.loc[0, city], heilongjiang_transformer.loc[0, city],\
    heilongjiang_ghi.loc[0, city]= Heilongjiang_pv(data_path, in_path)





    

