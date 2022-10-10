# -*- coding: utf-8 -*-
"""
Created on Tue Jun 22 20:13:39 2021

@author: Cesar
"""

#Loading modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from IPython import get_ipython
import statsmodels.api as sm

#selecciono el grafico en Terminal (inline) o en ventana emergente (qt5)
#get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('matplotlib', 'qt5')

#%% Elijo el directorio donde se encuentran los archivos que voy a analizar

os.chdir (r'C:\Users\Miguel\Desktop\Labo 3\Sexta entrega')
# print("Nombre del archivo completo con terminación .txt incluida")
# file = input()

# Nombre completo del archivo de datos que voy a analizar
file = 'simulacion con desacople variable'

R7 = 10        # [Ohm] - Es la R que utilizo para medir la corriente en el primario
frec = 1000     # [Hz] - Es la frecuencia de trabajo (fija)
w = 2*np.pi*frec # [rad/s] - Frecuencia angular de trabajo (fija)


#%% Importo los datos del archivo resultante de la simulación
data = np.loadtxt(file,dtype=float,delimiter = ',',skiprows= 1)
#data = np.loadtxt(file,dtype=float,delimiter = '\t',skiprows= 1)

#%% Extraigo los datos de Vi, Atenuación Vp y Atenuación V R7

R = data[:,0]      # Vi [V]
A_Vp = data[:,1]    # Atenuación de 
A_VR8 = data[:,3]   # Atenuación de V_R7

Vp = pow(10, (A_Vp/20))     # Cálulo Vp a partir de la Atenuación
VR8 = pow(10, (A_VR8/20))   # Cálculo de V_R7 a partir de la Atenuación

T= abs(VR8/Vp)/10

#%% Grafico Vi en fc de Ip para hacer una 1ra observación de los datos

plt.ion()
plt.close("all")
plt.figure(1,figsize= (11, 7))
plt.plot(R,T,".--k", label = 'Simulación')
plt.xlabel("R desacople(\u03A9)", fontsize = 17)
plt.ylabel("Transferencia", fontsize = 17)
#plt.title('Primera observación de los datos \n (unidos por segmentos rectos)', fontsize = 16)
plt.xscale('log')
plt.yscale('log')
#plt.grid('on')
plt.legend(loc='upper left')
plt.show()





