# -*- coding: utf-8 -*-
"""
Created on Tue Jul 11 21:46:43 2023

@author: beneg
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
import pickle
import pandas as pd
import fractions
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'qt5')   

os.chdir (r'C:\Users\beneg\OneDrive\Escritorio\Labo 6')

#%%

def measure_time(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        
        hours, rem = divmod(total_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours > 0:
            time_str += f"{int(hours)} hours "
        if minutes > 0:
            time_str += f"{int(minutes)} minutes "
        if seconds > 0 or (hours == 0 and minutes == 0):
            time_str += f"{seconds:.2f} seconds"

        print(f"Total time: {time_str}")
        return result
    return wrapper


def conservacion(Malla, conductividad_x, conductividad_y, i, j):
    
    """
    Update the matrix M based on conservation rules.

    Parameters:
    - M: numpy.ndarray
        The input matrix to be updated.
    - conductividad_x: float
        Scaling factor for the effect of the i-direction neighbors.
    - conductividad_y: float
        Scaling factor for the effect of the j-direction neighbors.
    - i: int
        Row index of the element to be updated.
    - j: int
        Column index of the element to be updated.

    Returns:
    - None
    
    """
    
    M = Malla
    
    M[i, j] = 2 * (conductividad_x + conductividad_y) * M[i, j]
    M[i + 1, j] = -conductividad_x * M[i + 1, j]
    M[i - 1, j] = -conductividad_x * M[i - 1, j]
    M[i, j + 1] = -conductividad_y * M[i, j + 1]
    M[i, j - 1] = -conductividad_y * M[i, j - 1]



def conservation_matrix(filas, columnas, conductividad_x, conductividad_y):
    
    """
    
    Generates the matrix used in the calculation with the given dimensions.

    Parameters:
        m (int): Number of rows in the matrix.
        n (int): Number of columns in the matrix.

    Returns:
        F (ndarray): Generated matrix of shape (m*n, m*n).

    Description:
        This function generates a matrix F with the dimensions m*n to be used in a calculation. The matrix is constructed
        by iterating over each position (i, j) in the matrix and applying a modification to the auxiliary matrix P based on
        the conservation equation. The modified P matrix is then flattened and stored as a row in the final matrix F.

    Notes:
        - The function assumes the availability of the 'conservacion' function, which modifies the P matrix in place
          based on the conservation equation.
          
    """
    m = filas
    n = columnas
    
    F = np.zeros((m*n, m*n)) # pre-allocate memory for F
    k = 0  
    
    for i in range(1,m+1):
        for j in range(1,n+1): 
            P = np.ones((m+2,n+2))
            # P[P != 1] = 1 # Cambio todos los lugares por 1 y asi vuelvo a la P original
            conservacion(P, conductividad_x, conductividad_y, i, j) # modify P in place
            # matriz con todos ceros menos los 5 lugares
            mask = (P == 1)
            # Set the locations where the mask is True (i.e., where the matrix has 1's) to 0
            P[mask] = 0
            #Condiciones de contorno
            if i - 1 < 1:
                P[i,j] += P[i - 1,j]
            if i + 1 > m:
                  P[i,j] += P[i + 1,j]
            if j - 1 < 1:
                P[i,j] += P[i,j - 1]
            if j + 1 > n:
                P[i,j] += P[i,j + 1]
            F[k] = P[1:m+1, 1:n+1].flatten() # store values in F
            k += 1

    return F

def contactos(M, i1, j1, i2, j2, Error = None):
    
    """
   Modify a matrix and create a right-hand side vector for contact conditions.

   Parameters:
   - M: numpy.ndarray
       The matrix to be modified.
   - i1, j1: int
       Coordinates of the first contact position.
   - i2, j2: int
       Coordinates of the second contact position.

   Returns:
   - M: numpy.ndarray
       The modified matrix with contact conditions.
   - b: numpy.ndarray
       The right-hand side vector for contact conditions.

   Note:
   - This function modifies the input matrix M to incorporate contact conditions at the specified positions.
   - The contact conditions are set by updating the corresponding elements of M and creating a right-hand side vector b.
   - The contact positions are given by the coordinates (i1, j1) and (i2, j2).
   - The value of 1 is assigned to the contact positions in the matrix M.
   - The value of 1 is assigned to the corresponding positions in the right-hand side vector b.
   - The value of -1 is assigned to the opposite contact position in the right-hand side vector b.
   - The modified matrix M and the right-hand side vector b are returned.

   """
   
    contacto_i1_j1= np.zeros(m*n)
    contacto_i2_j2= contacto_i1_j1.copy()
    b= contacto_i2_j2.copy()
    
    contacto_i1_j1[((n*(i1-1)) + (j1-1))] = 1
    contacto_i2_j2[((n*(i2-1)) + (j2-1))] = 1
    
    if Error == True:
    
        b[((n*(i1-1)) + (j1-1))] = 1 +  np.random.normal(0, 0.3)
        b[((n*(i2-1)) + (j2-1))]=-1 + np.random.normal(0, 0.3)
    
    else:
        
        b[((n*(i1-1)) + (j1-1))] = 1
        b[((n*(i2-1)) + (j2-1))]=-1
    
    
    M[((n*(i1-1)) + (j1-1)),:]= contacto_i1_j1
    M[((n*(i2-1)) + (j2-1)),:]= contacto_i2_j2
    
    
    return M,b

       


def GaussSeidel(A, b, x0, tol=1e-6, n=1000):
    
    """
    Solve a linear system using the Gauss-Seidel method.

    Parameters:
    - A: numpy.ndarray
        The coefficient matrix of the linear system.
    - b: numpy.ndarray
        The right-hand side vector of the linear system.
    - x0: numpy.ndarray
        The initial guess for the solution.
    - tol: float, optional
        Tolerance for convergence (default: 1e-6).
    - n: int, optional
        Maximum number of iterations (default: 1000).

    Returns:
    - numpy.ndarray
        The solution to the linear system.

    Note:
    - The Gauss-Seidel method is an iterative algorithm for solving linear systems.
    - The function solves the linear system Ax = b using the Gauss-Seidel method, starting from an initial guess x0.
    - The algorithm iteratively updates the solution x by solving equations of the form x_i = (b_i - sum(A_ij * x_j)) / A_ii.
    - The iterations continue until convergence is achieved (the difference between consecutive solutions is below tol)
      or the maximum number of iterations n is reached.

    """
    
    L = np.tril(A, k=-1)
    U = np.triu(A, k=1)
    D = np.diag(np.diag(A))
    x = x0
    for _ in range(n):
        x_prev = x.copy()
        x = np.linalg.solve(D + L, b - np.dot(U, x))
        error = np.linalg.norm(x - x_prev)
        if error < tol:
            return x
    return x



def corriente(malla_con_corriente, i, j,conductividad_x,conductividad_y):
    
    """
    Calculate the current at a specific position in the solution matrix.

    Parameters:
    - malla_con_corriente: numpy.ndarray
        The solution matrix.
    - i, j: int
        Coordinates of the position to calculate the current.

    Returns:
    - i_0: float
        The calculated current at the specified position.

    Note:
    - This function calculates the current at the specified position (i, j) in the solution matrix.
    - The solution matrix is assumed to have missing values (NaN) for positions outside the grid.
    - The current is calculated based on the neighboring values of the specified position.
    - The left, right, up, and down values are determined by the neighboring positions.
    - The current i_0 is calculated using the given values of conductividad_x and conductividad_y.
    - The calculated current i_0 is returned.

    """
    
    i = i - 1
    j = j - 1
    rows, cols = malla_con_corriente.shape

    # Calculate values of neighbors (assuming missing values are equal to central element)
    left = malla_con_corriente[i, j - 1] if j > 0 else malla_con_corriente[i, j]
    right = malla_con_corriente[i, j + 1] if j < cols - 1 else malla_con_corriente[i, j]
    up = malla_con_corriente[i - 1, j] if i > 0 else malla_con_corriente[i, j]
    down = malla_con_corriente[i + 1, j] if i < rows - 1 else malla_con_corriente[i, j]

    # Calculate i_0
    i_0 = -conductividad_x * (left - malla_con_corriente[i, j] + right - malla_con_corriente[i, j]) - conductividad_y * (up - malla_con_corriente[i, j] + down - malla_con_corriente[i, j])

    return i_0

def Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None):
    '''
    Calculo de la matriz con los contactos de R1
    '''
    i_R1, j_R1 = Contact_pairs[0]
    k_R1, l_R1 = Contact_pairs[1]
    
    matriz = np.asarray(conservation_matrix(m, n,conductividad_x, conductividad_y))
    
    if Error == True:
        
        matriz, b1 = contactos(matriz, i_R1, j_R1, k_R1, l_R1, Error = True)
        
    else:
        
        matriz, b1 = contactos(matriz, i_R1, j_R1, k_R1, l_R1, Error = None)
        
    x0 = np.zeros_like(b1)

    solucion = GaussSeidel(matriz, b1, x0)

    solucion = solucion.reshape((m, n))

    malla_contactos_x = solucion / corriente(solucion, i_R1, j_R1,conductividad_x,conductividad_y)
    
    return malla_contactos_x

def Matrix_Y_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None):
    '''
    Calculo de la matriz con los contactos de R2
    '''
    i_R2, j_R2 = Contact_pairs[2]
    k_R2, l_R2 = Contact_pairs[3]
    
    matriz = np.asarray(conservation_matrix(m, n,conductividad_x, conductividad_y))
    
    if Error == True:
        
        matriz, b1 = contactos(matriz, i_R2, j_R2, k_R2, l_R2, Error = True)
        
    else:
        
        matriz, b1 = contactos(matriz, i_R2, j_R2, k_R2, l_R2, Error = None)
        
    x0 = np.zeros_like(b1)

    solucion = GaussSeidel(matriz, b1, x0)

    solucion = solucion.reshape((m, n))

    malla_contactos_y = solucion / corriente(solucion, i_R2, j_R2,conductividad_x,conductividad_y)
    
    return malla_contactos_y

def Diferencia(malla_con_corriente,p1,q1,p2,q2):   
    
    """
    Calculate the difference between two positions in the solution matrix.

    Parameters:
    - malla_con_corriente: numpy.ndarray
        The solution matrix.
    - p1, q1: int
        Coordinates of the first position.
    - p2, q2: int
        Coordinates of the second position.

    Returns:
    - difference: float
        
    """
    
    return malla_con_corriente[p1-1,q1-1] - malla_con_corriente[p2-1,q2-1]


def Resistivity_x(Contact_pairs, m, n, conductividad_x, conductividad_y, Error = None):
    
    """
    Calculate the difference between two positions in the solution matrix for the R1 pair.

    """

    i1, j1 = Contact_pairs[0]
    i2, j2 = Contact_pairs[1]
    p1, p2 = Contact_pairs[4]
    q1, q2 = Contact_pairs[5]
    
    if Error == True:
        
        malla_contactos_x = Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = True)
        
    else:
        
        malla_contactos_x = Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None)
    
    dif = Diferencia(malla_contactos_x, p1, p2, q1, q2)
    
    return dif


def Resistivity_y(Contact_pairs, m, n, conductividad_x, conductividad_y, Error = None):
    
    """
    Calculate the difference between two positions in the solution matrix for the R2 pair.


    """

    i1, j1 = Contact_pairs[2]
    i2, j2 = Contact_pairs[3]
    p1, p2 = Contact_pairs[4]
    q1, q2 = Contact_pairs[5]

    if Error == True:

        malla_contactos_y = Matrix_Y_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = True)
    
    else:
        
        malla_contactos_y = Matrix_Y_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None)
    
    dif = Diferencia(malla_contactos_y, p1, p2, q1, q2)
    
    return dif


def R1_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None):
    X_resistivity = []
    
    if Error == True:
        
        matriz_contactos_R1 = Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = True)
    
    else:
        
        matriz_contactos_R1 = Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None)
    
    for t in range(1, int((n + 1) / 2)):
        p1, q1 = 1, int((n + 1) / 2) + t
        p2, q2 = m, int((n + 1) / 2) - t

        dif = Diferencia(matriz_contactos_R1, p1, q1, p2, q2)
        X_resistivity.append(dif)
    return X_resistivity

def R2_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None):
    Y_resistivity = []
    
    if Error == True:
        
        matriz_contactos_R2 = Matrix_Y_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = True)
    
    else:
        
        matriz_contactos_R2 = Matrix_Y_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None)
    
    for t in range(1, int((n + 1) / 2)):
        p1, q1 = 1, int((n + 1) / 2) + t
        p2, q2 = m, int((n + 1) / 2) - t

        dif = Diferencia(matriz_contactos_R2, p1, q1, p2, q2)
        Y_resistivity.append(dif)
    return Y_resistivity


def get_neighbors_positions(m, n, contacts, radius):
    matrix = np.ones((m,n))
    positions_list = []
    for i, j in contacts:
        i = i - 1
        j = j - 1
        positions = []
        nrows, ncols = matrix.shape

        for r in range(1, radius+1):
            for di in range(-r, r+1):
                for dj in range(-r, r+1):
                    if abs(di) + abs(dj) <= r:
                        ii = i + di
                        jj = j + dj
                        if ii >= 0 and ii < nrows and jj >= 0 and jj < ncols:
                            positions.append((ii, jj))

        chosen_position = random.choice(positions)
        if not isinstance(chosen_position, tuple) or len(chosen_position) != 2:
            chosen_position = (i+1, j+1)
        else:
            chosen_position = (chosen_position[0]+1, chosen_position[1]+1)
        positions_list.append(chosen_position)
    return positions_list   
  


def Anisotropy_Graph(df, m, n, figure=None): 
    
    x  = 2*np.arange(1,int((n + 1)/2))/m
    
    if figure is None:
        plt.figure(figure)
        
    plt.suptitle(f'Malla de {m}x{n}')
    plt.subplot(1,2,1)
    plt.title('Anisotropy')
    for index, row in df.iterrows():
        plt.plot(x,row['Anisotropy'],'-o',label=f"conductividades  = {row['Conductividad']}")
    plt.xlabel('Distancia entre contactos/ el ancho')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.title('Difference of Anisotropy')
    for index, row in df.iterrows():
        plt.plot(x,row['Difference of Anisotropy'],'-o',label=f"conductividades = {row['Conductividad']}")
    plt.xlabel('Distancia entre contactos/ el ancho')
        
    plt.legend() 

def Dif_Anisotropy_Graph(df, m, n, figure=None): 
    
    colors = ['red', 'green', 'blue']
    markers = ['o', 's', '^', '*', 'D', 'x']
    
    x  = 2*np.arange(1,int((n + 1)/2))/m
    
    if figure is None:
        plt.figure(figure)
        
    
    plt.title(f'Malla de {m}x{n}')
    for i in range(1, k + 1):
        for j, (index, row) in enumerate(df.iterrows()):
            color = colors[j % len(colors)]  # Choose color cyclically
            marker = markers[i - 1]  # Use the i-th marker
            plt.plot(x, row[f'Difference of Anisotropy for {fractions.Fraction(i / k).limit_denominator()}'], color=color, marker=marker, linestyle='-')
        
        plt.plot([], [], label=f'Difference of Anisotropy for {fractions.Fraction(i / k).limit_denominator()}', color='k', marker=markers[i-1], linestyle='', markersize=5)
    plt.xlabel('Distancia entre contactos/ el ancho')
    plt.ylabel('Diferencia de Anisotropia')
    plt.legend() 



class AnisotropyAnalyzer:
    
    
    @measure_time
    def Sweep_fix_borders( m, n, conductividades, Contact_pairs, Error = None):
        
        print(f'Malla de {m}x{n}')
        
        
        df_list = []  # List to store individual DataFrames
        with tqdm(total=len(conductividades), desc ="Sweeping conductivities") as pbar_h:
            
            for conductividad_x, conductividad_y in conductividades:
                
                if Error == True:
                
                    X_resistivity = R1_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = True)
                    Y_resistivity = R2_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = True)
               
                else:
                    
                    X_resistivity = R1_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None)
                    Y_resistivity = R2_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs, Error = None)
               
                # Create a DataFrame for the current iteration
                df_iteration = pd.DataFrame({
                    'Conductividad': [(conductividad_x, conductividad_y)],
                    'X resistivity': [np.array(X_resistivity)],
                    'Y resistivity': [np.array(Y_resistivity)]})
            
                # Append the DataFrame to the list
                df_list.append(df_iteration)
                
                pbar_h.update(1)
                
        # Concatenate all DataFrames in the list into a single DataFrame
        df_Concatenate = pd.concat(df_list)
        df_Concatenate.reset_index(drop=True, inplace=True)  # Reset index values
        
              
        df_Concatenate['Y resistivity / X resistivity'] = df_Concatenate['Y resistivity'] / df_Concatenate['X resistivity']       
        
        df_Concatenate['Anisotropy'] = (df_Concatenate['X resistivity'] - df_Concatenate['Y resistivity']) / (df_Concatenate['X resistivity'] + df_Concatenate['Y resistivity'])
                                
        
        df_Concatenate['Difference of Anisotropy'] = df_Concatenate['Anisotropy'].apply(lambda x: x - df_Concatenate.loc[0, 'Anisotropy'])

            
                              
        return df_Concatenate  
     
    @measure_time
    def Sweep_dif_borders(m, n, conductividades, k, Error = None): 
        
        print(f'Malla de {m}x{n}')
        # Primer par de contacto que va a variar segun el largo pero se 
        # quedan en el medio
        i_R1, j_R1 = 1, int((n + 1)/2)
        k_R1, l_R1 = m, int((n + 1)/2)
        
        
        Contact_pairs_R1 = [(i_R1,j_R1),(k_R1,l_R1)]
        
        df_list = []  # List to store individual DataFrames
        with tqdm(total=len(conductividades)*k, desc = f"Sweeping conductivities for {k} positions") as pbar_h:
        
            for conductividad_x, conductividad_y in conductividades:
                
                if Error == True:
                
                    X_resistivity = R1_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs_R1, Error = True)
                     
                else:
                    
                    X_resistivity = R1_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs_R1, Error = None)
                    
                df_iteration = pd.DataFrame({
                    'Conductividad': [(conductividad_x, conductividad_y)],
                    'X resistivity': [np.array(X_resistivity)]})
                
                '''
                Aca vario la posicion de los contactos desde el medio hasta afuera
                '''
                
                for i in range(1,k + 1):
                    # Segundo para de contactos que se moveran al centro 
                    i_R2, j_R2 = int((m + 1)/2), int((n + 1)/2) + i*int(((n + 1)/2)/k)#Me lo dejara fijo en la
                    k_R2, l_R2 = int((m + 1)/2), int((n + 1)/2) - i*int(((n + 1)/2)/k) #Matriz de 55
                    
                    Contact_pairs_R2 = [(i_R1,j_R1),(k_R1,l_R1),(i_R2,j_R2),(k_R2,l_R2)]
                    
                    if Error == True:
                        
                        Y_resistivity = R2_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs_R2, Error = True)
                    
                    else:
                        
                        Y_resistivity = R2_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs_R2, Error = None)
                    
                    columns_name = f'Y resistivity in {fractions.Fraction(i/k).limit_denominator()}'
                    
                    df_iteration[columns_name] = [np.array(Y_resistivity)]
                    
                    df_iteration[f'Anisotropy for {fractions.Fraction(i/k).limit_denominator()}'] = (df_iteration['X resistivity'] - df_iteration[columns_name]) / (df_iteration['X resistivity'] + df_iteration[columns_name])

                    pbar_h.update(1)
                    
                # Append the DataFrame to the list
                df_list.append(df_iteration)

                # Concatenate all DataFrames in the list into a single DataFrame
                df_Concatenate = pd.concat(df_list)
                df_Concatenate.reset_index(drop=True, inplace=True)  # Reset index values
                
                for i in range(1, k + 1):
                    
                    df_Concatenate[f'Difference of Anisotropy for {fractions.Fraction(i/k).limit_denominator()}'] \
                        = df_Concatenate[f'Anisotropy for {fractions.Fraction(i/k).limit_denominator()}'].apply(lambda x: x - df_Concatenate.loc[0, f'Anisotropy for {fractions.Fraction(i/k).limit_denominator()}'])
                
        return df_Concatenate


    @measure_time  
    def Fix_borders(m, n, conductividades, Contact_pairs, Error = None):
        
        print(f'Malla de {m}x{n}')
        
        R1 = []
        R2 = []
        Psi = []
        
        with tqdm(total=len(conductividades), desc ="Sweeping conductivities") as pbar_h:
        
        
            for conductividad_x, conductividad_y in conductividades:
                
                Psi.append((conductividad_x - conductividad_y)/(conductividad_x + conductividad_y))
                
                if Error == True:
                
                    R1.append(Resistivity_x(Contact_pairs, m, n, conductividad_x, conductividad_y, Error = True))
                    
                    R2.append(Resistivity_y(Contact_pairs, m, n, conductividad_x, conductividad_y, Error = True))
                    
                else:
                    
                    R1.append(Resistivity_x(Contact_pairs, m, n, conductividad_x, conductividad_y, Error = None))
                    
                    R2.append(Resistivity_y(Contact_pairs, m, n, conductividad_x, conductividad_y, Error = None))
                    
                pbar_h.update(1)
        
        # Zip the lists together
        zipped_list = list(zip(conductividades,Psi, R1, R2))
        
        # Create a DataFrame from the zipped list
        df = pd.DataFrame(zipped_list)
        
        # Set the column names of the DataFrame
        df.columns = ["conductividades","Psi", "R1", "R2"]
        
        df['Anisotropia'] = (df['R1'] - df['R2'])/(df['R1'] + df['R2'])
        
        df['Difference of Anisotropy'] = df['Anisotropia'] - df.loc[0,'Anisotropia']
        
        return df
        


           
#%% Barro los contactos de medicion para puntos de corriente fijo
Sweep_Anisotropy_fix_borders_dif_Lenghts = {}

m = 11
ns = [55] # Vario el largo
conductividades = [(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(1,5)]

for n in ns:
    # Par de contacto R1
    i_R1, j_R1 = 1, int((n + 1) / 2)
    k_R1, l_R1 = m, int((n + 1) / 2)

    # Par de contacto R2
    i_R2, j_R2 = int((m + 1) / 2), n
    k_R2, l_R2 = int((m + 1) / 2), 1
    
    Contact_pairs = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2)]
    
    
    
    Sweep_Anisotropy_fix_borders_dif_Lenghts[f'Lenght {n}'] \
        = AnisotropyAnalyzer.Sweep_fix_borders(m, n, conductividades, Contact_pairs, Error = None)
        
        
    Anisotropy_Graph(Sweep_Anisotropy_fix_borders_dif_Lenghts[f'Lenght {n}'], m, n,figure= None)

#%% Guardo los datos?

# Save the dictionary to a file
with open('Sweep_Anisotropy_fix_borders_dif_Lenghts_all_Anisotropy.pkl', 'wb') as file:
    pickle.dump(Sweep_Anisotropy_fix_borders_dif_Lenghts, file)
    
    
    
# Load the dictionary from the file
# with open('Sweep_Anisotropy_fix_borders_dif_Lenghts.pkl', 'rb') as file:
#     Sweep_Anisotropy_fix_borders_dif_Lenghts = pickle.load(file)

#%%Barro los contactos de medicion para puntos de corriente que varian a lo largo de la muestra


conductividades = [(1,1), (2,1), (1,2)]

m = 11

k = 6 #Cantidad de puntos que voy a queres barrer

Sweep_Anisotropy_dif_borders_dif_Lenghts = {}


ns = [55,67,77,89,99,111] # Vario el largo

for n in ns:
    
    Sweep_Anisotropy_dif_borders_dif_Lenghts[f'Lenght {n}'] = AnisotropyAnalyzer.Sweep_dif_borders(m, n, conductividades, k, Error = None)
    
    Dif_Anisotropy_Graph(Sweep_Anisotropy_dif_borders_dif_Lenghts[f'Lenght {n}'], m, n,figure=None)

#%% Guardo los datos?
# Save the dictionary to a file
with open('Sweep_Anisotropy_dif_borders_dif_Lenghts.pkl', 'wb') as file:
    pickle.dump(Sweep_Anisotropy_dif_borders_dif_Lenghts, file)

#%% Barros varios Psi para varios tamaños


i = 6
k = 6
t = int(m/2)

# conductividades = [(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(1,5)]


Anisotropy_dif_Psi = {}

ns = [55,67,77,89,99,111]
for n in ns:
    
    i_R1, j_R1 = 1, int((n + 1)/2)

    k_R1, l_R1 = m, int((n + 1)/2)

    i_R2, j_R2 = int((m + 1)/2), int((n + 1)/2) + i*int(((n + 1)/2)/k)

    k_R2, l_R2 = int((m + 1)/2), int((n + 1)/2) - i*int(((n + 1)/2)/k)

    p1, q1 = 1, int((n + 1)/2) + t

    p2, q2 = m, int((n + 1)/2) - t


    Contact_pairs = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2),(p1, q1),(p2, q2)]
    
    
    Anisotropy_dif_Psi[f'Lenght {n}'] = AnisotropyAnalyzer.Fix_borders(m, n, conductividades, Contact_pairs, Error = None)
    
#%% Guardo los datos?
# Save the dictionary to a file
with open('Anisotropy_dif_Psi_dif_lenght_zoom.pkl', 'wb') as file:
    pickle.dump(Anisotropy_dif_Psi, file)

#%% Barros varios Psi para un tamaño con contactos random

Random_Anisotropy_dif_Lenght = {}
Pairs_dif_Lenght = {}

m= 11
ns = [55,67,77,89,99,111]
t = int(m/2)
R = 10
radius = 1

# conductividades = [(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(1,5)]

Lista_fina = np.linspace(1,2,5)

conductividades = []

for i in range(len(Lista_fina)-1,-1,-1):
    conductividades.append((Lista_fina[i],1))


for i in range(1, len(Lista_fina)):
    conductividades.append((1,Lista_fina[i]))

start_time = time.time()
for n in ns:
    

    i_R1, j_R1 = 1, int((n + 1)/2)
    
    k_R1, l_R1 = m, int((n + 1)/2)
    
    i_R2, j_R2 = int((m + 1)/2), n
    
    k_R2, l_R2 = int((m + 1)/2), 1
    
    p1, q1 = 1, int((n + 1)/2) + t
    
    p2, q2 = m, int((n + 1)/2) - t
    
    Pares = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2),(p1, q1),(p2, q2)]
    contacts = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2)]
    
    Pairs = {'Ideal' : Pares}
    
    Random_Anisotropy = {}
    
        
    for o in range(1, R + 1):
        
        print(f' Iteration {o}')
        
        Contactos_random = get_neighbors_positions(m, n, contacts, radius)
        Contactos_random.append(Pares[4])
        Contactos_random.append(Pares[5])
        
      
        # Assign the Series to the DataFrame column
        Pairs[f'Random contacts {o}'] = Contactos_random
        
        Random_Anisotropy[f'Random contacts {o}'] = AnisotropyAnalyzer.Fix_borders(m, n, conductividades, Contactos_random, Error = None)
        
    Random_Anisotropy_dif_Lenght[f'Lenght {n}'] = Random_Anisotropy
    Pairs_dif_Lenght[f'Lenght {n}'] = Pairs

end_time = time.time()

total_time = end_time - start_time

# Format the total_time as a string in hours, minutes, and seconds
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

time_str = ""
if hours > 0:
    time_str += f"{int(hours)} hours "
if minutes > 0:
    time_str += f"{int(minutes)} minutes "
if seconds > 0 or (hours == 0 and minutes == 0):
    time_str += f"{seconds:.2f} seconds"

print(f"Total time: {time_str}")
    
    #%% Guardo los datos?
# Save the dictionary to a file
with open('Random_Anisotropy_dif_Lenght_radius_1.pkl', 'wb') as file:
    pickle.dump(Random_Anisotropy_dif_Lenght, file)

with open('Pairs_dif_Lenght_radius_1.pkl', 'wb') as file:
    pickle.dump(Pairs_dif_Lenght, file)
#%%
# Load the dictionary from the file
with open('Anisotropy_dif_Psi_dif_lenght.pkl', 'rb') as file:
    Anisotropy_dif_Psi_dif_lenght = pickle.load(file)
    
#%%
# Anisotropy_With_Noise = {}
# Load the dictionary from the file    
with open('Anisotropy_With_Noise.pkl', 'rb') as file:
    Anisotropy_With_Noise = pickle.load(file)
    
#%%   
m= 11
ns = [111]
t = int(m/2) 

conductividades = [(1,1),(2,1),(3,1),(4,1),(5,1),(1,2),(1,3),(1,4),(1,5)]

start_time = time.time()

for n in ns:
    
    i_R1, j_R1 = 1, int((n + 1)/2)
    
    k_R1, l_R1 = m, int((n + 1)/2)
    
    i_R2, j_R2 = int((m + 1)/2), n
    
    k_R2, l_R2 = int((m + 1)/2), 1
    
    p1, q1 = 1, int((n + 1)/2) + t
    
    p2, q2 = m, int((n + 1)/2) - t
    
    Contact_pairs = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2),(p1, q1),(p2, q2)]
    
    Iterations = {}
    
    for o in range(50):
        
        print(f'Iteration {o}')
        
        Iterations[f'Iteration {o}'] = AnisotropyAnalyzer.Fix_borders(m, n, conductividades, Contact_pairs, Error = True)
        
    
    Anisotropy_With_Noise[f'Lenght {n}'] = Iterations    
    
end_time = time.time()

total_time = end_time - start_time

# Format the total_time as a string in hours, minutes, and seconds
hours, rem = divmod(total_time, 3600)
minutes, seconds = divmod(rem, 60)

time_str = ""
if hours > 0:
    time_str += f"{int(hours)} hours "
if minutes > 0:
    time_str += f"{int(minutes)} minutes "
if seconds > 0 or (hours == 0 and minutes == 0):
    time_str += f"{seconds:.2f} seconds"

print(f"Total time: {time_str}")
    
#%%
with open('Anisotropy_With_Noise.pkl', 'wb') as file:
    pickle.dump(Anisotropy_With_Noise, file)
    
    
#%%

@measure_time
def Montgomery(m,n,conductividades):
    
    i_R1, j_R1 = 1, 1
    k_R1, l_R1 = m, 1
    
    i_R2, j_R2 = 1, n #Me lo dejara fijo en la
    k_R2, l_R2 = m, n 
    
    for conductividad_x, conductividad_y in conductividades:
        
        # df_iteration = pd.DataFrame({
        #     'Conductividad': [(conductividad_x, conductividad_y)]})
        
        Contact_pairs_R1 = [(5,5),(5,5),(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2)]
        
        Y_resistivity = Resistivity_y(Contact_pairs_R1, m, n, conductividad_x, conductividad_y)
        
        Contact_pairs_R2 = [(i_R1, j_R1),(i_R2, j_R2),(5,5),(5,5),(k_R1, l_R1),(k_R2, l_R2)]
        
        X_resistivity = Resistivity_x(Contact_pairs_R2, m, n, conductividad_x, conductividad_y)
        
        return np.array(X_resistivity)/np.array(Y_resistivity)
#%%  Esta mal la funcion, esta calculando matrices de mas. Eso es por la definicion de 
# Y_resistivity y X_Resisitivity

Montgomery_dif_anisotrpo = []

m = 11
ns = [11,23,33,45,55]

conductividades = [(1,1)]

for n in ns:
    Montgomery_dif_anisotrpo.append(Montgomery(m, n, conductividades))
#%%

plt.close('all')

x = np.array(ns)/m
y = Montgomery_dif_anisotrpo

plt.figure(figsize = (12,8))
plt.grid(color='black',ls='dotted',lw=0.5) 
plt.plot(x,y,color = sns.color_palette('rocket', 1)[0])
plt.xlabel('Largo/Ancho',fontsize=24)
plt.ylabel('R2/ R1',fontsize=24)
plt.yscale('log')


