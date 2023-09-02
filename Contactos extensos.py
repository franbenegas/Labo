# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 20:52:30 2023

@author: beneg
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import random
import csv
import pickle
import pandas as pd
from scipy.optimize import curve_fit
import math
from threading import Thread
from numba import njit
import fractions
get_ipython().run_line_magic('matplotlib', 'qt5')   

#%%

def measure_time(func):
    
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        total_time = end_time - start_time
        if total_time >= 3600:
            total_time = total_time / 3600
            print(f"Total time: {total_time:.2f} hours")
        elif total_time >= 60:
            total_time = total_time / 60
            print(f"Total time: {total_time:.2f} minutes")
        else:
            print(f"Total time: {total_time:.2f} seconds")
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

def contacts_within_area(m, n, chosen_position: tuple, area_size: int) -> list:
    """
    Generate a list of coordinates within a specified area around a chosen position.

    Parameters:
    - m, n: int
        The dimensions of the grid or matrix.
    - chosen_position: tuple
        A tuple containing the coordinates (i, j) of the chosen position.
    - area_size: int
        The size of the area around the chosen position to consider.

    Returns:
    - positions_list: list of tuples
        A list of tuples containing coordinates (i, j) of positions within the specified area.

    Note:
    - This function calculates the coordinates of positions within the area defined by the chosen position and area_size.
    - The coordinates are represented as tuples (i, j).
    - The boundaries of the area are constrained by the dimensions of the grid or matrix (m, n).
    - The function returns a list of tuples, where each tuple represents a position within the specified area.
    - The positions are sorted in row-major order, starting from the top-left corner of the area.

    Example:
    >>> contacts_within_area(5, 5, (3, 3), 1)
    [(2, 2), (2, 3), (2, 4), (3, 2), (3, 3), (3, 4), (4, 2), (4, 3), (4, 4)]
    """
    contacts_within_area = []
    
    i, j = chosen_position
    i = i - 1
    j = j - 1
    
    rows, cols = m, n
    # Calculate the boundaries of the area around the chosen position
    i_min = max(0, i - area_size)
    i_max = min(rows - 1, i + area_size)
    j_min = max(0, j - area_size)
    j_max = min(cols - 1, j + area_size)
        
    for x in range(i_min, i_max + 1):
        for y in range(j_min, j_max + 1):
            contacts_within_area.append((x + 1, y + 1))
    
    return contacts_within_area

def contactos(m,n,M, contact_points):
    """
    Modify a matrix and create a right-hand side vector for contact conditions.

    Parameters:
    - M: numpy.ndarray
        The matrix to be modified.
    - contact_points: list of list of tuples
        A list containing two sublists of tuples, where each sublist represents a set of contact points.
        The first sublist contains tuples representing positive contact points, and the second sublist contains tuples representing negative contact points.

    Returns:
    - M: numpy.ndarray
        The modified matrix with contact conditions.
    - b: numpy.ndarray
        The right-hand side vector for contact conditions.

    Note:
    - This function modifies the input matrix M to incorporate contact conditions at the specified positions.
    - The contact conditions are set based on the provided list of contact points.
    - Positions in both sublists are assigned the value '1' in the matrix M to indicate contact.
    - Positions in the first sublist are assigned the value '1' in the right-hand side vector b.
    - Positions in the second sublist are assigned the value '-1' in the right-hand side vector b.
    - The modified matrix M and the right-hand side vector b are returned.
    """
    b = np.zeros(m*n)
    
    for index, positions in enumerate(contact_points):
        for i, j in positions:
            M[i-1, j-1] = 1
            b[(i-1) * n + (j-1)] = 1 if index == 0 else -1
    
    return M, b

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

def corriente(malla_con_corriente, chosen_positions, conductividad_x, conductividad_y):
    """
    Calculate the total current at specified positions.

    Parameters:
    - malla_con_corriente: numpy.ndarray
        The solution matrix.
    - chosen_positions: list of tuples
        A list containing tuples with coordinates (i, j) of the chosen positions.
    - conductividad_x, conductividad_y: float
        The conductivity values.

    Returns:
    - total_current: float
        The total current calculated at the specified positions.

    Note:
    - This function calculates the total current at the specified positions in the solution matrix.
    - It calculates the current for each specified position.
    - The total current is returned as the sum of currents at all specified positions.
    """
    total_current = 0.0

    for chosen_position in chosen_positions:
        i, j = chosen_position
        i = i - 1
        j = j - 1
        rows, cols = malla_con_corriente.shape

        # Calculate values of neighbors (assuming missing values are equal to central element)
        left = malla_con_corriente[i, j - 1] if j > 0 else malla_con_corriente[i, j]
        right = malla_con_corriente[i, j + 1] if j < cols - 1 else malla_con_corriente[i, j]
        up = malla_con_corriente[i - 1, j] if i > 0 else malla_con_corriente[i, j]
        down = malla_con_corriente[i + 1, j] if i < rows - 1 else malla_con_corriente[i, j]

        # Calculate i_0 for the current position
        i_0 = -conductividad_x * (left - malla_con_corriente[i, j] + right - malla_con_corriente[i, j]) - conductividad_y * (up - malla_con_corriente[i, j] + down - malla_con_corriente[i, j])

        # Add the current for the current position to the total current
        total_current += i_0

    return total_current


def Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs):
    '''
    Calculo de la matriz con los contactos de R1
    '''

    contacts_points = Contact_pairs[:2]
    position = Contact_pairs[0]

    matriz = np.asarray(conservation_matrix(m, n,conductividad_x, conductividad_y))
    matriz, b1 = contactos(m,n,matriz, contacts_points)
    x0 = np.zeros_like(b1)

    solucion = GaussSeidel(matriz, b1, x0)

    solucion = solucion.reshape((m, n))

    malla_contactos_x = solucion / corriente(solucion, position,conductividad_x,conductividad_y)
    
    return malla_contactos_x


def Matrix_Y_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs):
    '''
    Calculo de la matriz con los contactos de R2
    '''

    contacts_points = Contact_pairs[2:4]
    position = Contact_pairs[2]
    
    matriz = np.asarray(conservation_matrix(m, n,conductividad_x, conductividad_y))
    matriz, b1 = contactos(m,n,matriz, contacts_points)
    x0 = np.zeros_like(b1)

    solucion = GaussSeidel(matriz, b1, x0)

    solucion = solucion.reshape((m, n))

    malla_contactos_y = solucion / corriente(solucion, position,conductividad_x,conductividad_y)
    
    return malla_contactos_y


def Diferencia(malla_con_corriente, positions_lists):
    """
    Calculate the difference between two lists of positions in the solution matrix.

    Parameters:
    - malla_con_corriente: numpy.ndarray
        The solution matrix.
    - positions_lists: list of list of tuples
        A list of lists, where each inner list represents a list of positions with coordinates as tuples.

    Returns:
    - difference: float
        The calculated difference between the total values of the last two lists of positions in the solution matrix.
    """
    if len(positions_lists) < 2:
        raise ValueError("The 'positions_lists' parameter must contain at least two lists of positions.")
    
    total_value_1 = sum([malla_con_corriente[i - 1, j - 1] for i, j in positions_lists[-2]])
    total_value_2 = sum([malla_con_corriente[i - 1, j - 1] for i, j in positions_lists[-1]])

    difference = total_value_1 - total_value_2
    
    return difference


def Resistivity_x(Contact_pairs, m, n, conductividad_x, conductividad_y):
    
    """
    Calculate the difference between two positions in the solution matrix for the R1 pair.

    """
   
    position_pairs = Contact_pairs[4:]
    
    malla_contactos_x = Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs)
    
    dif = Diferencia(malla_contactos_x,  position_pairs)
    
    return dif


def Resistivity_y(Contact_pairs, m, n, conductividad_x, conductividad_y):
    
    """
    Calculate the difference between two positions in the solution matrix for the R2 pair.


    """

    position_pairs = Contact_pairs[4:]

    malla_contactos_y = Matrix_Y_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs)
    
    dif = Diferencia(malla_contactos_y, position_pairs)
    
    return dif


def R1_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs, area_size):
    X_resistivity = []
    matriz_contactos_R1 = Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs)
    
    for t in range(1, int((n + 1) / 2)):
        p1, q1 = 1, int((n + 1) / 2) + t
        p2, q2 = m, int((n + 1) / 2) - t

        position_pairs = [(p1, q1), (p2, q2)]        
        
        Contact_pairs[-2] = contacts_within_area(m, n, position_pairs[0], area_size)
        Contact_pairs[-1] = contacts_within_area(m, n, position_pairs[1], area_size)
        
        dif = Diferencia(matriz_contactos_R1, Contact_pairs)
        X_resistivity.append(dif)
    return X_resistivity

def R2_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs, area_size):
    Y_resistivity = []
    matriz_contactos_R2 = Matrix_Y_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs)
    
    for t in range(1, int((n + 1) / 2)):
        p1, q1 = 1, int((n + 1) / 2) + t
        p2, q2 = m, int((n + 1) / 2) - t
        
        position_pairs = [(p1, q1), (p2, q2)] 
        
        Contact_pairs[-2] = contacts_within_area(m, n, position_pairs[0], area_size)
        Contact_pairs[-1] = contacts_within_area(m, n, position_pairs[1], area_size)

        dif = Diferencia(matriz_contactos_R2, Contact_pairs)
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

def Dif_Anisotropy_Graph(df, m, n, k, figure=None): 
    
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
    def Sweep_fix_borders(m, n, conductividades, Contact_pairs, area_size):
        
        print(f'Malla de {m}x{n}')
        
        Contact_pairs2 = []
        
        for k in Contact_pairs:
            Contact_pairs2.append(contacts_within_area(m, n, k, area_size))
        
        
        df_list = []  # List to store individual DataFrames
        with tqdm(total=len(conductividades), desc ="Sweeping conductivities") as pbar_h:
            
            for conductividad_x, conductividad_y in conductividades:
            
                X_resistivity = R1_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs2, area_size)
                Y_resistivity = R2_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs2, area_size)
               
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
    def Sweep_dif_borders(m, n, conductividades, k, area_size): 
        
        print(f'Malla de {m}x{n}')
        # Primer par de contacto que va a variar segun el largo pero se 
        # quedan en el medio
        i_R1, j_R1 = 1, int((n + 1)/2)
        k_R1, l_R1 = m, int((n + 1)/2)
        
        
        Contact_pairs_R1 = [(i_R1,j_R1),(k_R1,l_R1)]
        
        Contact_pairs_R1[0] = contacts_within_area(m, n, Contact_pairs_R1[0], area_size)
        Contact_pairs_R1[1] = contacts_within_area(m, n, Contact_pairs_R1[1], area_size)
        
        df_list = []  # List to store individual DataFrames
        with tqdm(total=len(conductividades)*k, desc = f"Sweeping conductivities for {k} positions") as pbar_h:
        
            for conductividad_x, conductividad_y in conductividades:
                    
                X_resistivity = R1_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs_R1, area_size)
                     
                
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
                    
                    Contact_pairs_R2[2] = contacts_within_area(m, n, Contact_pairs_R2[2], area_size)
                    Contact_pairs_R2[3] = contacts_within_area(m, n, Contact_pairs_R1[3], area_size)
                    
                    Y_resistivity = R2_Sweep(m, n, conductividad_x, conductividad_y, Contact_pairs_R2, area_size)
                    
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
    def Fix_borders(m, n, conductividades, Contact_pairs, area_size):
        
        print(f'Malla de {m}x{n}')
        
        R1 = []
        R2 = []
        Psi = []
        
        with tqdm(total=len(conductividades), desc ="Sweeping conductivities") as pbar_h:
        
        
            for conductividad_x, conductividad_y in conductividades:
                
                Psi.append((conductividad_x - conductividad_y)/(conductividad_x + conductividad_y))
                
                R1.append(Resistivity_x(Contact_pairs, m, n, conductividad_x, conductividad_y, area_size))
                
                R2.append(Resistivity_y(Contact_pairs, m, n, conductividad_x, conductividad_y, area_size))
        
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
    
    

#%%
Sweep_Anisotropy_fix_borders_dif_Lenghts = {}

m = 11
ns = [111] # Vario el largo
conductividades = [(1,1), (2,1), (1,2)]
area_size = 0 # No esta dando lo mismo con area_size = 0, deberia dar igual

for n in ns:
    # Par de contacto R1
    i_R1, j_R1 = 1, int((n + 1) / 2)
    k_R1, l_R1 = m, int((n + 1) / 2)

    # Par de contacto R2
    i_R2, j_R2 = int((m + 1) / 2), n
    k_R2, l_R2 = int((m + 1) / 2), 1
    
    Contact_pairs = [(i_R1, j_R1),(k_R1, l_R1),(i_R2, j_R2),(k_R2, l_R2)]
    
    
    
    Sweep_Anisotropy_fix_borders_dif_Lenghts[f'Lenght {n}'] \
        = AnisotropyAnalyzer.Sweep_fix_borders(m, n, conductividades, Contact_pairs, area_size)
        
        
    Anisotropy_Graph(Sweep_Anisotropy_fix_borders_dif_Lenghts[f'Lenght {n}'], m, n,figure= None)
#%%

# # Replace these values with your specific parameters
# m = 11
# n = 11
# conductividad_x = 1.0
# conductividad_y = 1.0
# Pares = [(6, 1), (6, 11)]
# Contact_pairs = []
# for k in Pares:
#     Contact_pairs.append(contacts_within_area(m, n, k, 1))

# # print(Contact_pairs)
# # Calculate the matrix using Matrix_X_contacts
# malla_contactos_x = Matrix_X_contacts(m, n, conductividad_x, conductividad_y, Contact_pairs)