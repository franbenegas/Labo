# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 11:40:18 2023

@author: LBT
"""

import numpy as np
import matplotlib.pyplot as plt
from IPython import get_ipython
import time
import os
import copy
import pandas as pd
import fractions
from tqdm import tqdm
import random as rd
from scipy.sparse import csc_matrix
import pickle
import seaborn as sns

from scipy.sparse.linalg import spsolve
get_ipython().run_line_magic('matplotlib', 'qt5')

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


def Distribucion_dominios(filas: int, columnas: int, pesos: tuple, plot = None) -> np.ndarray:
    """
    Generate a matrix representing the distribution of domains with random values of 1 or -1 based on given weights.
    
    Args:
        filas (int): Number of rows in the matrix.
        columnas (int): Number of columns in the matrix.
        pesos (tuple): A tuple containing the weights for choosing 1 and -1 values respectively.
        plot (bool, optional): If True, plot the generated matrix. Defaults to None.
    
    Returns:
        numpy.ndarray: Matrix representing the distribution of domains.
    """
    # Generate the matrix of domains with random values of 1 or -1 based on given weights
    # Matriz_dominios = np.asarray([rd.choices([1, -1], weights=pesos, k=columnas) for i in range(filas)])
    Matriz_dominios = np.asarray(rd.choices([1, -1], weights=pesos, k=columnas*filas)).reshape((filas, columnas))
    
    # Plot the matrix if plot is set to True
    if plot is True:
        # plt.title()
        plt.matshow(Matriz_dominios, cmap='Blues', aspect='auto')
        plt.colorbar(label='Sigma Value')
        plt.title(f'Pesos = {pesos}')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
    
    return Matriz_dominios


def Interaccion_col(Matriz_dominios: np.ndarray) -> np.ndarray:
    
    diff_cols = np.diff(Matriz_dominios,axis=1)
    
    diff_cols[np.where((diff_cols == 2))] = 100 
    diff_cols[np.where((diff_cols == -2))] = -100 
    
    #Aca decido las islas de conductivdad
    # Iterate over each row
    for row_idx in range(diff_cols.shape[0]):
        row_values = diff_cols[row_idx]
        # Find the index of the first non-zero value
        non_zero_idx = np.argmax(row_values != 0)
        non_zero_idx_last = np.argmax(row_values[::-1] != 0)
        
        # Handle the case where -2 occurs first
        if row_values[non_zero_idx] == -100:
            row_values[:non_zero_idx] = sigma_columnas
            
        
        # Handle the case where 2 occurs first
        elif row_values[non_zero_idx] == 100:
            row_values[:non_zero_idx] = sigma_filas
            row_values[len(row_values)-non_zero_idx_last:] = sigma_filas
        
            
        #Me mira como termina cada fila a ver que onda
        if row_values[::-1][non_zero_idx_last] == -100:
            row_values[len(row_values)-non_zero_idx_last:] = sigma_filas
            
        elif row_values[::-1][non_zero_idx_last] == 100:
              row_values[len(row_values)-non_zero_idx_last:] = sigma_columnas
             
               
        # Initialize variables to store indices
        indices_between_minus_100 = []
        indices_between_100_minus_100 = []
        
        # Initialize variables to track the index of the last occurrence of -2 and 2
        last_minus_100_idx = None
        last_100_idx = None
        
        # Iterate over each index and value in the row
        for idx, value in enumerate(row_values):
            # Check if the current value is -2
            if value == -100:
                # Record the index of the last occurrence of -2
                last_minus_100_idx = idx
            # Check if the current value is 2
            elif value == 100:
                # Record the index of the last occurrence of 2
                last_100_idx = idx
            # Check if the last occurrence of -2 and 2 have been recorded
            if last_minus_100_idx is not None and last_100_idx is not None:
                # Add the indices between -2 and 2 to the list
                indices_between_minus_100.extend([i for i in range(last_minus_100_idx + 1, last_100_idx) if i not in indices_between_minus_100])
                # Add the indices between 2 and -2 to the list
                indices_between_100_minus_100.extend([i for i in range(last_100_idx + 1, last_minus_100_idx) if i not in indices_between_100_minus_100])
        
        
        row_values[indices_between_100_minus_100] = sigma_columnas
        row_values[indices_between_minus_100] = sigma_filas
        
        if not np.any((row_values != 0)):
            # Fill the entire column with sigma_col
            if np.any((Matriz_dominios[row_idx] == 1)):
                diff_cols[row_idx] = sigma_columnas
                
            else:
            
                diff_cols[row_idx] = sigma_filas
            
    return diff_cols


def Interaccion_filas(Matriz_dominios: np.ndarray) -> np.ndarray:
    
    diff_rows = np.diff(Matriz_dominios,axis=0)
    
    diff_rows[np.where((diff_rows == 2))] = 100 
    diff_rows[np.where((diff_rows == -2))] = -100
    
    # Iterate over each column
    for col_idx in range(diff_rows.shape[1]):
        col_values = diff_rows[:, col_idx]
        # Find the index of the first non-zero value
        non_zero_idx = np.argmax(col_values != 0)
        non_zero_idx_last = np.argmax(col_values[::-1] != 0)
        
        # Handle the case where -2 occurs first
        if col_values[non_zero_idx] == -100:
            col_values[:non_zero_idx] = sigma_filas
            
        # Handle the case where 2 occurs first
        elif col_values[non_zero_idx] == 100:
            col_values[:non_zero_idx] = sigma_columnas
            col_values[len(col_values)-non_zero_idx_last:] = sigma_columnas
            
        
        #Me mira como termina cada fila a ver que onda
        if col_values[::-1][non_zero_idx_last] == -100:
            col_values[len(col_values)-non_zero_idx_last:] = sigma_columnas
            
        elif col_values[::-1][non_zero_idx_last] == 100:
            col_values[len(col_values)-non_zero_idx_last:] = sigma_filas
               
        # Initialize variables to store indices
        indices_between_minus_100 = []
        indices_between_100_minus_100 = []
        
        # Initialize variables to track the index of the last occurrence of -2 and 2
        last_minus_100_idx = None
        last_100_idx = None
        
        # Iterate over each index and value in the column
        for idx, value in enumerate(col_values):
            # Check if the current value is -2
            if value == -100:
                # Record the index of the last occurrence of -2
                last_minus_100_idx = idx
            # Check if the current value is 2
            elif value == 100:
                # Record the index of the last occurrence of 2
                last_100_idx = idx
            # Check if the last occurrence of -2 and 2 have been recorded
            if last_minus_100_idx is not None and last_100_idx is not None:
                # Add the indices between -2 and 2 to the list
                indices_between_minus_100.extend([i for i in range(last_minus_100_idx + 1, last_100_idx) if i not in indices_between_minus_100])
                # Add the indices between 2 and -2 to the list
                indices_between_100_minus_100.extend([i for i in range(last_100_idx + 1, last_minus_100_idx) if i not in indices_between_100_minus_100])
        
        # Use the indices to update the values
        col_values[indices_between_100_minus_100] = sigma_filas
        col_values[indices_between_minus_100] = sigma_columnas
        
        if not np.any((col_values != 0)):
            # Fill the entire column with sigma_filas
            # diff_rows[:, col_idx] = sigma_filas
            
            if np.any((Matriz_dominios[:, col_idx] == 1)):
                diff_rows[:, col_idx] = sigma_filas
                
            else:
            
                diff_rows[:, col_idx] = sigma_columnas

    return diff_rows


def Matriz_de_interacciones(filas:int, columnas:int ,Matriz_dominios, conductividades:tuple, pesos:tuple, plot_matriz_interaccion = None, plot_diff_col = None, plot_diff_rows = None) -> np.ndarray:
    """
    Calculate the interaction matrix representing the interaction between domains based on given conductivities.

    Args:
        filas (int): Number of rows in the matrix.
        columnas (int): Number of columns in the matrix.
        Matriz_dominios (numpy.ndarray): Matrix representing the distribution of domains.
        conductividades (tuple): A tuple containing the conductivities for rows and columns respectively.
        plot_matriz_interaccion (bool, optional): If True, plot the interaction matrix. Defaults to None.
        plot_diff_col (bool, optional): If True, plot the difference matrix for columns. Defaults to None.
        plot_diff_rows (bool, optional): If True, plot the difference matrix for rows. Defaults to None.

    Returns:
        numpy.ndarray: Interaction matrix.
    """
    
    sigma_filas, sigma_columnas = conductividades
    
    sigma_mono = 0.5 * ((1 / sigma_filas) + (1 / sigma_columnas))
    
    if pesos == (0, 100):
        sigma_filas, sigma_columnas = sigma_columnas, sigma_filas
         
    
    Matriz_interaccion = np.zeros((2 * filas - 1, 2 * columnas - 1))
    
    diff_cols = Interaccion_col(Matriz_dominios)
    diff_rows = Interaccion_filas(Matriz_dominios)

    #Calculo la martriz entera

        
    for i in range(filas):
        if 2 * i + 1 < 2 * filas:
            Matriz_interaccion[2 * i, ::2] = Matriz_dominios[i, :]
            Matriz_interaccion[2 * i, 1::2] = diff_cols[i, :]

        
    for i in range(columnas):
        if 2 * i + 1 < 2 * columnas:
            Matriz_interaccion[1::2, 2 * i] = diff_rows[:, i]
            
    if sigma_filas == sigma_columnas:
        Matriz_interaccion[np.where(abs(Matriz_interaccion)==100)] = sigma_filas #LO CAMBIO POR LA PARED
    
    else: 
        Matriz_interaccion[np.where(abs(Matriz_interaccion)==100)] = sigma_mono
    
    #todos los plots
    
    if plot_diff_col is True:
        plt.matshow(diff_cols,cmap='Blues')
        plt.colorbar(label='Sigma Value')
        plt.title('diff_cols')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
    

    if plot_diff_rows is True:
        plt.matshow(diff_rows,cmap='Blues')
        plt.colorbar(label='Sigma Value')
        plt.title('diff_rows')
        plt.xlabel('Columns')
        plt.ylabel('Rows')


    if plot_matriz_interaccion is True:
        plt.matshow(Matriz_interaccion, cmap='Blues', aspect='auto')
        plt.title('Matriz de interaccion')
        plt.colorbar(label='Sigma Value')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
    
    
    return Matriz_interaccion
    

def Matriz_de_conservacion(filas: int, columnas: int, Matriz_interaccion, plot_matriz_conservacion=None) -> np.ndarray:
    """
    Calculate the conservation matrix representing conservation of charge in each position.

    Args:
        filas (int): Number of rows in the matrix.
        columnas (int): Number of columns in the matrix.
        Matriz_interaccion (numpy.ndarray): Interaction matrix representing the interaction between domains.
        plot_matriz_conservacion (bool, optional): If True, plot the conservation matrix. Defaults to None.

    Returns:
        numpy.ndarray: Conservation matrix.
    """
    # Calculate interactions between columns, right and left neighbors
    vec_costado_derecho = Matriz_interaccion[::2, 1::2]
    vec_costado_derecho = np.concatenate((vec_costado_derecho, np.zeros(filas)[:, np.newaxis]), axis=1)
    vec_costado_derecho = vec_costado_derecho.flatten()
    vecinos_derecha = np.diag(vec_costado_derecho[:-1], k=1)
    vecinos_izquierda = np.diag(vec_costado_derecho[:-1], k=-1)
    
    # Calculate interactions between rows, top and bottom neighbors
    vec_arriba = Matriz_interaccion[1::2, ::2].flatten()
    vecinos_abajo = np.diag(vec_arriba, k=columnas)
    vecinos_arriba = np.diag(vec_arriba, k=-columnas)
    
    # Calculate diagonal, elements of the original grid
    vec_costado_alargado_1 = np.append(np.zeros(columnas), vec_arriba)
    vec_costado_alargado_2 = np.append(vec_arriba, np.zeros(columnas))
    vec_diag_diferente_alargado_1 = np.append([0], vec_costado_derecho[:-1])
    vec_diag_diferente_alargado_2 = vec_costado_derecho
    vec_diagonal = -(vec_costado_alargado_1 + vec_costado_alargado_2 + vec_diag_diferente_alargado_1 + vec_diag_diferente_alargado_2)
    elementos_de_malla = np.diag(vec_diagonal)
    
    # Total conservation across all positions
    Matriz_conservacion = vecinos_derecha + vecinos_izquierda + elementos_de_malla + vecinos_abajo + vecinos_arriba
    
    if plot_matriz_conservacion is True:
        plt.matshow(Matriz_conservacion, cmap='Blues', aspect='auto')
        plt.title('Matriz de conservaciones')
        plt.colorbar(label='Sigma Value')
    
    return Matriz_conservacion

def Contactos_circulares(filas: int, columnas: int, posicion: tuple, radius: int) -> list:
    """
    Generate a list of positions within a circular region around a given position.
    
    Args:
        filas (int): Number of rows in the matrix.
        columnas (int): Number of columns in the matrix.
        posicion (tuple): Tuple representing the position (i, j) around which the circular region is defined.
        radius (int): Radius of the circular region.

    Returns:
        list: List of positions within the circular region.
    """
    matrix = np.ones((filas, columnas))
    
    if radius == 0:
        return [posicion]
    
    i, j = posicion

    i = i - 1
    j = j - 1
    
    positions = []
    nrows, ncols = matrix.shape

    for r in range(1, radius + 1):
        for di in range(-r, r + 1):
            for dj in range(-r, r + 1):
                if abs(di) + abs(dj) <= r:
                    ii = i + di
                    jj = j + dj
                    if 0 <= ii < nrows and 0 <= jj < ncols:
                        positions.append((ii + 1, jj + 1))

    return positions
 

def Contactos_barra(filas: int, col_deseada: int) -> list:
    """
    Generate a list of positions representing a bar across a desired column.

    Args:
        filas (int): Number of rows in the matrix.
        col_deseada (int): Desired column index.
    Returns:
        list: List of positions forming a bar across the desired column.
    """
    Contactos = []
    
    for i in range(filas):
        Contactos.append((i + 1, col_deseada))
    
    return Contactos


def contactos_extensos(matriz_conservacion, Contactos_extensos_Izquierda: list, Contactos_extensos_Derecha: list, Error: bool = None) -> tuple:
    """
    Apply extensive contacts to the conservation matrix.

    Args:
        matriz_conservacion (np.ndarray): Conservation matrix.
        Contactos_extensos_Izquierda (list): List of positions for extensive contacts on the left side.
        Contactos_extensos_Derecha (list): List of positions for extensive contacts on the right side.
        Error (bool, optional): Flag indicating whether to add noise to the extensive contacts. Defaults to None.

    Returns:
        tuple: Tuple containing the updated conservation matrix and the b vector.
    """
    b = np.zeros(filas * columnas)
    
    for punto_izquierdo in Contactos_extensos_Izquierda:
        contacto_i1_j1 = np.zeros(filas * columnas)
        i1, j1 = punto_izquierdo
        contacto_i1_j1[(columnas * (i1 - 1)) + (j1 - 1)] = 1
        
        if Error == True:
            b[(columnas * (i1 - 1)) + (j1 - 1)] = 1 + np.random.normal(0, 0.3)
        else:
            b[(columnas * (i1 - 1)) + (j1 - 1)] = 1
        
        matriz_conservacion[(columnas * (i1 - 1)) + (j1 - 1), :] = contacto_i1_j1
    
    for punto_derecho in Contactos_extensos_Derecha:
        contacto_i2_j2 = np.zeros(filas * columnas)
        i2, j2 = punto_derecho
        contacto_i2_j2[(columnas * (i2 - 1)) + (j2 - 1)] = 1
        
        if Error == True:
            b[(columnas * (i2 - 1)) + (j2 - 1)] = -1 + np.random.normal(0, 0.3)
        else:
            b[(columnas * (i2 - 1)) + (j2 - 1)] = -1
        
        matriz_conservacion[(columnas * (i2 - 1)) + (j2 - 1), :] = contacto_i2_j2
    
    return matriz_conservacion, b



def corriente_extensa(malla_con_corriente, Contactos: list, Matriz_interaccion) -> float:
    """
    Calculate the total current flowing through extensive contacts in a mesh.

    Args:
    - malla_con_corriente (numpy.ndarray): The mesh with currents.
    - Contactos (list): A list of positions where extensive contacts are located.
    - Matriz_interaccion (numpy.ndarray): The interaction matrix representing how elements in the mesh interact with each other.

    Returns:
    - float: The total current flowing through the extensive contacts.

    """

    # Initialize a variable to store the total current
    i_suma = 0
    
    # Loop over each contact position in Contactos
    for par_corriente_pos in Contactos:
        
        # Extract the row and column indices (i and j) from the contact position
        i, j = par_corriente_pos
        
        # Adjust indices to match array indexing (starts from 0)
        i = i - 1
        j = j - 1
        
        # Calculate the number of rows and columns in the malla_con_corriente mesh
        rows, cols = malla_con_corriente.shape
    
        # Calculate the values of neighboring elements (left, right, up, down) around the current position.
        # If the neighbor is at the boundary, take the current element itself.
        left = malla_con_corriente[i, j - 1] if j > 0 else malla_con_corriente[i, j]
        right = malla_con_corriente[i, j + 1] if j < cols - 1 else malla_con_corriente[i, j]
        up = malla_con_corriente[i - 1, j] if i > 0 else malla_con_corriente[i, j]
        down = malla_con_corriente[i + 1, j] if i < rows - 1 else malla_con_corriente[i, j]
        
        # Extract the interaction values of neighboring elements (left_interaction, right_interaction, up_interaction, down_interaction)
        # from the Matriz_interaccion. If the neighbor is at the boundary, set the interaction value to 0.
        left_interaction = Matriz_interaccion[2*i, 2*j - 1] if j > 0 else 0
        right_interaction = Matriz_interaccion[2*i, 2*j + 1] if j < cols - 1 else  0
        up_interaction =  Matriz_interaccion[2*i - 1, 2*j] if i > 0 else 0 
        down_interaction =  Matriz_interaccion[2*i + 1, 2*j] if i < rows - 1 else 0
        
        # Calculate i_0, which represents the current flowing into the current element
        # due to its interaction with neighboring elements.
        i_0 = - left_interaction * (left - malla_con_corriente[i, j]) \
              - right_interaction * (right - malla_con_corriente[i, j]) \
              - up_interaction * (up - malla_con_corriente[i, j]) \
              - down_interaction * (down - malla_con_corriente[i, j])
        
        # Update i_suma by adding i_0
        i_suma += i_0
    
    # Return the total current i_suma
    return i_suma


def Diferencia_extensa(malla_con_corriente, Contactos_tension_Izquierda:list, Contactos_tension_Derecha:list) -> float:
    """
    Calculate the potential difference between extensive contacts in a mesh.

    Args:
    - malla_con_corriente (numpy.ndarray): The mesh with currents.
    - Contactos_tension_Izquierda (list): A list of positions for extensive contacts on the left side.
    - Contactos_tension_Derecha (list): A list of positions for extensive contacts on the right side.

    Returns:
    - float: The potential difference between the left and right extensive contacts.
    """

    # Initialize variables to store potential differences at left and right extensive contacts
    V_izquierda = 0
    V_derecha = 0
    
    # Loop over each left extensive contact position
    for punto_izquierdo in Contactos_tension_Izquierda:
        # Extract row and column indices (p1, q1) from the contact position
        p1, q1 = punto_izquierdo
        # Adjust indices to match array indexing (starts from 0)
        p1, q1 = p1 - 1, q1 - 1
        # Add the potential at the left extensive contact to V_izquierda
        V_izquierda += malla_con_corriente[p1, q1]
    
    # Loop over each right extensive contact position
    for punto_derecho in Contactos_tension_Derecha:
        # Extract row and column indices (p2, q2) from the contact position
        p2, q2 = punto_derecho
        # Adjust indices to match array indexing (starts from 0)
        p2, q2 = p2 - 1, q2 - 1
        # Add the potential at the right extensive contact to V_derecha
        V_derecha += malla_con_corriente[p2, q2]
    
    # Calculate the potential difference between left and right extensive contacts
    diferencia = V_izquierda - V_derecha
    
    # Return the potential difference
    return diferencia   


def Malla_total(filas:int, columnas:int, Matriz_dominios, conductividades:tuple, Contactos_corriente_entrada:list, Contactos_corriente_salida:list, plot_matriz_interaccion = None, plot_matriz_conservacion = None, diferencia_corriente = None, normalizar = None,plot_contactos = None) -> np.ndarray:
    
    # Calculate interaction matrix and conservation matrix
    if plot_matriz_interaccion == True:
        Matriz_interaccion = Matriz_de_interacciones(filas, columnas, Matriz_dominios, conductividades, pesos, plot_matriz_interaccion=True)
    else: 
        Matriz_interaccion = Matriz_de_interacciones(filas, columnas, Matriz_dominios, conductividades, pesos, plot_matriz_interaccion=None)
        
    if plot_matriz_conservacion == True:   
        Matriz_conservacion = Matriz_de_conservacion(filas, columnas, Matriz_interaccion, plot_matriz_conservacion=True)
        
    else:
        Matriz_conservacion = Matriz_de_conservacion(filas, columnas, Matriz_interaccion, plot_matriz_conservacion=None)
    
    # Apply extensive contacts to conservation matrix and solve for currents
    Matriz_contactos, Condiciones = contactos_extensos(Matriz_conservacion, Contactos_corriente_entrada, Contactos_corriente_salida)
    
    Matriz_contactos = csc_matrix(Matriz_contactos)
    Malla_con_corriente = spsolve( Matriz_contactos, Condiciones)
    Malla_con_corriente = Malla_con_corriente.reshape((filas,columnas))
    
    if diferencia_corriente == True:
        corriente_entrada = corriente_extensa(Malla_con_corriente, Contactos_corriente_entrada, Matriz_interaccion)
        corriente_salida = corriente_extensa(Malla_con_corriente, Contactos_corriente_salida, Matriz_interaccion)
        
        dif_corriente = corriente_entrada + corriente_salida
        
        print("Corriente de entrada: %.2f" %  corriente_entrada)
        print("Diferencia de corriente: %.2f" %  dif_corriente)
    
    # Normalize the mesh by dividing by the total current
    if normalizar == False:
        Malla_normalizada = Malla_con_corriente
        
    else:
        Malla_normalizada = Malla_con_corriente / corriente_extensa(Malla_con_corriente, Contactos_corriente_entrada, Matriz_interaccion)
    
    
    # Plot the contacts if required
    if plot_contactos is True:
        plt.figure()
        plt.title('Malla cargada')
        plt.contourf(Malla_normalizada, cmap='Blues')
        plt.colorbar()
        
        # Plot input contacts
        for point in Contactos_corriente_entrada:
            plt.scatter(point[1]-1, point[0]-1, color='blue', marker='x')  # Adjust the color and marker style as needed
        
        # Plot output contacts
        for point in Contactos_corriente_salida:
            plt.scatter(point[1]-1, point[0]-1, color='blue', marker='x')
    
    return Malla_normalizada


# def Malla_cargada(filas:int, columnas:int, Matriz_dominios, conductividades:tuple, Contactos_corriente_entrada:list, Contactos_corriente_salida:list, plot_contactos = None) -> np.ndarray:
#     """
#     Simulate a loaded mesh with given parameters.

#     Args:
#     - filas (int): Number of rows in the mesh.
#     - columnas (int): Number of columns in the mesh.
#     - Matriz_dominios (numpy.ndarray): Matrix representing the domains in the mesh.
#     - conductividades (tuple): Tuple containing the conductivities for rows and columns.
#     - Contactos_corriente_entrada (list): List of positions for current input contacts.
#     - Contactos_corriente_salida (list): List of positions for current output contacts.
#     - plot_contactos (bool, optional): Whether to plot the contacts. Defaults to None.

#     Returns:
#     - numpy.ndarray: The normalized mesh with currents.
#     """

#     # Calculate interaction matrix and conservation matrix
#     Matriz_interaccion = Matriz_de_interacciones(filas, columnas, Matriz_dominios, conductividades, pesos, plot_matriz_interaccion=None)
#     Matriz_conservacion = Matriz_de_conservacion(filas, columnas, Matriz_interaccion, plot_matriz_conservacion=None)
    
#     # Apply extensive contacts to conservation matrix and solve for currents
#     Matriz_contactos, Condiciones = contactos_extensos(Matriz_conservacion, Contactos_corriente_entrada, Contactos_corriente_salida)
    
#     Matriz_contactos = csc_matrix(Matriz_contactos)
#     Malla_con_corriente = spsolve( Matriz_contactos, Condiciones)
#     Malla_con_corriente = Malla_con_corriente.reshape((filas,columnas))
    
#     # Normalize the mesh by dividing by the total current
#     Malla_normalizada = Malla_con_corriente / corriente_extensa(Malla_con_corriente, Contactos_corriente_entrada, Matriz_interaccion)
    
#     dif_corriente = corriente_extensa(Malla_con_corriente, Contactos_corriente_entrada, Matriz_interaccion) + corriente_extensa(Malla_con_corriente, Contactos_corriente_salida, Matriz_interaccion)
    
#     # print("%.2f" %  dif_corriente)
#     # print("%.2f" %  corriente_extensa(Malla_con_corriente, Contactos_corriente_entrada, Matriz_interaccion))
    
#     # Plot the contacts if required
#     if plot_contactos is True:
#         plt.figure()
#         plt.contourf(Malla_normalizada, cmap='Blues')
#         plt.colorbar()
        
#         # Plot input contacts
#         for point in Contactos_corriente_entrada:
#             plt.scatter(point[1]-1, point[0]-1, color='blue', marker='x')  # Adjust the color and marker style as needed
        
#         # Plot output contacts
#         for point in Contactos_corriente_salida:
#             plt.scatter(point[1]-1, point[0]-1, color='blue', marker='x')
    
#     return Malla_normalizada

# @measure_time

def Seis_contactos(filas:int, columnas:int, Matriz_dominios, conductividades: tuple, radio :int, plot_eje_largo=None, plot_eje_corto=None) -> float:
    """
    Simulate six contacts scenario with given parameters.

    Args:
    - filas (int): Number of rows in the mesh.
    - columnas (int): Number of columns in the mesh.
    - Matriz_conservacion (numpy.ndarray): Conservation matrix.
    - conductividades (tuple): Tuple containing the conductivities for rows and columns.
    - radio (int): Radius for circular contacts.
    - plot_eje_largo (bool, optional): Whether to plot the long axis. Defaults to None.
    - plot_eje_corto (bool, optional): Whether to plot the short axis. Defaults to None.

    Returns:
    - float: The ratio of differences between long and short axes.
    """

    # Define circular contacts for the long axis
    Contactos_corriente_entrada_circular_eje_largo = Contactos_circulares(filas, columnas, (int(filas/2) + 1,1), radius = radio) 
    Contactos_corriente_salida_circular_eje_largo = Contactos_circulares(filas, columnas, (int(filas/2) + 1,columnas), radius = radio) 
    
    # Define circular contacts for the short axis
    Contactos_corriente_entrada_circular_eje_corto = Contactos_circulares(filas, columnas, (1,int(columnas/2) + 1), radius = radio) 
    Contactos_corriente_salida_circular_eje_corto = Contactos_circulares(filas, columnas, (filas,int(columnas/2) + 1), radius = radio) 
    
    # Define circular contacts for tension
    Contactos_tension_entrada_circular = Contactos_circulares(filas, columnas, (filas,int(columnas/2) - int(filas/2)), radius = radio) 
    Contactos__tension_salida_circular = Contactos_circulares(filas, columnas, (1,int(columnas/2) + int(filas/2)), radius = radio)
    
    # Calculate loaded meshes for the long and short axes
    
    Matriz_cargada_eje_largo = Malla_total(filas, columnas, Matriz_dominios, conductividades, Contactos_corriente_entrada_circular_eje_largo, Contactos_corriente_salida_circular_eje_largo,normalizar=True)
    # Matriz_cargada_eje_largo = Malla_cargada(filas, columnas, Matriz_dominios, conductividades, Contactos_corriente_entrada_circular_eje_largo, Contactos_corriente_salida_circular_eje_largo)
    diff_largo = abs(Diferencia_extensa(Matriz_cargada_eje_largo, Contactos_tension_entrada_circular, Contactos__tension_salida_circular))
    # print(f"voltaje en el largo = {diff_largo}")
    
    Matriz_cargada_eje_corto = Malla_total(filas, columnas, Matriz_dominios, conductividades, Contactos_corriente_entrada_circular_eje_corto, Contactos_corriente_salida_circular_eje_corto,normalizar=True)
    # Matriz_cargada_eje_corto = Malla_cargada(filas, columnas, Matriz_dominios, conductividades, Contactos_corriente_entrada_circular_eje_corto, Contactos_corriente_salida_circular_eje_corto)
    diff_corto = abs(Diferencia_extensa(Matriz_cargada_eje_corto, Contactos_tension_entrada_circular, Contactos__tension_salida_circular))
    # print(f"voltaje en el corto = {diff_corto}")
    # Calculate ratio of differences between long and short axes
    # Ratio = (diff_corto - diff_largo)/ (diff_corto + diff_largo)
    
    # Plot the long axis if required
    if plot_eje_largo is True:
        plt.figure()
        plt.title(f'Conductividad: {conductividades}, Peso: {pesos}')
        plt.contourf(Matriz_cargada_eje_largo, cmap='Blues')
        plt.colorbar()
        
        # Plot current input contacts for the long axis
        for point in Contactos_corriente_entrada_circular_eje_largo:
            plt.scatter(point[1]-1, point[0]-1, color='blue', marker='o')  # Adjust the color and marker style as needed
        
        # Plot current output contacts for the long axis
        for point in Contactos_corriente_salida_circular_eje_largo:
            plt.scatter(point[1]-1, point[0]-1, color='blue', marker='x')
        
        # Plot tension input contacts for the long axis
        for point in Contactos_tension_entrada_circular:
            plt.scatter(point[1]-1, point[0]-1, color='red', marker='o')  # Adjust the color and marker style as needed
        
        # Plot tension output contacts for the long axis
        for point in Contactos__tension_salida_circular:
            plt.scatter(point[1]-1, point[0]-1, color='red', marker='x')
        
    
    # Plot the short axis if required
    if plot_eje_corto is True:
        plt.figure()
        plt.title(f'Conductividad: {conductividades}, Peso: {pesos}')
        plt.contourf(Matriz_cargada_eje_corto, cmap='Blues')
        plt.colorbar()
        
        # Plot current input contacts for the short axis
        for point in Contactos_corriente_entrada_circular_eje_corto:
            plt.scatter(point[1]-1, point[0]-1, color='blue', marker='o')  # Adjust the color and marker style as needed
        
        # Plot current output contacts for the short axis
        for point in Contactos_corriente_salida_circular_eje_corto:
            plt.scatter(point[1]-1, point[0]-1, color='blue', marker='x')
        
        # Plot tension input contacts for the short axis
        for point in Contactos_tension_entrada_circular:
            plt.scatter(point[1]-1, point[0]-1, color='red', marker='o')  # Adjust the color and marker style as needed
        
        # Plot tension output contacts for the short axis
        for point in Contactos__tension_salida_circular:
            plt.scatter(point[1]-1, point[0]-1, color='red', marker='x')
            
            
    return diff_corto, diff_largo


#%%#################################           Pruebas      ################################
#%% Malla con contactos barra

columnas = 41  # columnas
filas = 21 #filas      
sigma_filas = 1 # Este va entre las filas
sigma_columnas = 2 #Este es el que va en entre las columnas
conductividades = (sigma_filas, sigma_columnas)
Dominios_A = 24
pesos = (Dominios_A,100 - Dominios_A)
Matriz_dominios = Distribucion_dominios(filas, columnas, pesos, plot=True)

Contactos_corriente_entrada_barra = Contactos_barra(filas, 1)  
Contactos_corriente_salida_barra = Contactos_barra(filas, columnas)  


Contactos_tension_entrada_barra = Contactos_barra(filas, int(columnas/2) - int(filas/2))  
Contactos_tension_salida_barra = Contactos_barra(filas, int(columnas/2) + int(filas/2))  
    

Malla_barra = Malla_total(filas, columnas, Matriz_dominios, conductividades, Contactos_corriente_entrada_barra, Contactos_corriente_salida_barra,plot_contactos=True,plot_matriz_interaccion=True,diferencia_corriente=True)

dif_barra = Diferencia_extensa(Malla_barra, Contactos_tension_entrada_barra, Contactos_tension_salida_barra)

for point in Contactos_tension_entrada_barra:
    plt.scatter(point[1]-1, point[0]-1, color='red', marker='o')  # Adjust the color and marker style as needed
for point in Contactos_tension_salida_barra:
    plt.scatter(point[1]-1, point[0]-1, color='red', marker='x')

#%% Malla con contactos redondos
columnas =101  # columnas
filas = 51 #filas      
sigma_filas = 1 # Este va entre las filas
sigma_columnas = 2 #Este es el que va en entre las columnas
conductividades = (sigma_filas, sigma_columnas)
Dominios_A = 0
pesos = (Dominios_A,100 - Dominios_A)
Matriz_dominios = Distribucion_dominios(filas, columnas, pesos,plot=True)
radio = 0


Contactos_corriente_entrada_circular = Contactos_circulares(filas, columnas, (int(filas/2) + 1,1), radius = radio) 
Contactos_corriente_salida_circular = Contactos_circulares(filas, columnas, (int(filas/2) + 1,columnas), radius = radio) 


Contactos_tension_entrada_circular = Contactos_circulares(filas, columnas, (int(filas/2) + 1,int(columnas/2) - int(filas/4)), radius = radio) 
Contactos_tension_salida_circular = Contactos_circulares(filas, columnas, (int(filas/2) + 1,int(columnas/2) + int(filas/4)), radius = radio)


Malla_circular = Malla_total(filas, columnas, Matriz_dominios, conductividades, Contactos_corriente_entrada_circular, Contactos_corriente_salida_circular,plot_contactos=True,plot_matriz_interaccion=True,diferencia_corriente=True)

dif_circular = Diferencia_extensa(Malla_circular, Contactos_tension_entrada_circular, Contactos_tension_salida_circular)

for point in Contactos_tension_entrada_circular:
    plt.scatter(point[1]-1, point[0]-1, color='red', marker='o')  # Adjust the color and marker style as needed
for point in Contactos_tension_salida_circular:
    plt.scatter(point[1]-1, point[0]-1, color='red', marker='x')


#%% Metodo de los 6 contactos

columnas = 41  # columnas
filas = 21 #filas      
sigma_filas = 1 # Este va entre las filas
sigma_columnas = 3 #Este es el que va en entre las columnas
conductividades = (sigma_filas, sigma_columnas)
Dominios_A = 70
pesos = (Dominios_A,100 - Dominios_A)
Matriz_dominios = Distribucion_dominios(filas, columnas, pesos, plot=True)



Seis_contactos(filas, columnas, Matriz_dominios, conductividades,radio=0,plot_eje_corto=True,plot_eje_largo=True)


#%% Variacion de sigmas para contactos barra
columnas = 41  # columnas
filas = 21 #filas      

Contactos_corriente_entrada_barra = Contactos_circulares(filas, columnas, (int(filas/2)+1,1), radius = radio)#Contactos_barra(filas, 1)  
Contactos_corriente_salida_barra = Contactos_circulares(filas, columnas, (int(filas/2)+1,columnas), radius = radio)#Contactos_barra(filas, columnas)  


Contactos_tension_entrada_barra = Contactos_circulares(filas, columnas, (int(filas/2) +1,int(columnas/2) - int(filas/4)), radius = radio)#Contactos_barra(filas, int(columnas/2) - int(filas/2))  
Contactos_tension_salida_barra = Contactos_circulares(filas, columnas, (int(filas/2) + 1,int(columnas/2) + int(filas/4)), radius = radio)#Contactos_barra(filas, int(columnas/2) + int(filas/2))


radio = 0
# Contactos_tension_entrada_barra = Contactos_circulares(filas, columnas, (int(filas/2),int(columnas/2) - int(filas/2)), radius = radio) 
# Contactos_tension_salida_barra = Contactos_circulares(filas, columnas, (int(filas/2),int(columnas/2) + int(filas/2)), radius = radio)

mediciones = []

Columnas = ["Proporcion de Dominios","Resistencia eje largo","STD eje largo","Conductividad"]
df_barra = pd.DataFrame(columns=Columnas)

start_time = time.time()

for i in range(1,4):
    
    sigma_filas = 1 # Este va entre las filas
    sigma_columnas = i #Este es el que va en entre las columnas
    conductividades = (sigma_filas, sigma_columnas)

    Dominios_y_mediciones = []
    Promedios = []
    Desviaciones = []
    
    with tqdm(total=101*50) as pbar_h:
        for Dominios_A in range(0,101):
            
            pesos = (Dominios_A,100 - Dominios_A)
            
            
            Iteraciones = np.zeros(50)
            
            for _ in range(50):
                
                Matriz_dominios = Distribucion_dominios(filas, columnas, pesos)
                
                Malla_barra = Malla_total(filas, columnas, Matriz_dominios, conductividades, Contactos_corriente_entrada_barra, Contactos_corriente_salida_barra)
                
                dif_barra = Diferencia_extensa(Malla_barra, Contactos_tension_entrada_barra, Contactos_tension_salida_barra)
            
                Iteraciones[_] = dif_barra
                
                # Iteraciones[_] = Seis_contactos(filas, columnas, Matriz_dominios, conductividades, radio)
                
                pbar_h.update(1)
                
            Promedio_eje_largo = np.mean(Iteraciones)
            Desviacion_eje_largo = np.std(Iteraciones)
            
            # Dominios_y_mediciones.append(Dominios_A)
            # Promedios.append(Promedio)
            # Desviaciones.append(Desviacion)
            
            # Create a temporary DataFrame with the values
            temp_df = pd.DataFrame([[Dominios_A, Promedio_eje_largo,Desviacion_eje_largo, conductividades]], columns=Columnas)

            # Check if temp_df contains any empty or all-NA entries
            if not temp_df.isnull().values.all():
                # Concatenate the temporary DataFrame with the main DataFrame
                df_barra = pd.concat([df_barra, temp_df], ignore_index=True)
    # mediciones.append(Promedios)
    
    
    
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

plt.figure(figsize=(14,7))
plt.grid()
sns.scatterplot(data=df_barra, x="Proporcion de Dominios", y="Resistencia eje largo", hue="Conductividad",style="Conductividad")

#%%

# Save the dictionary to a file
with open('Simlaciones barra puntuales', 'wb') as file:
    pickle.dump(df_barra, file)
#%% Variacion de sigmas para seis contactos

columnas = 101  # columnas
filas = 51 #filas      
radio=0
Columnas = ["Proporcion de Dominios","Resistencia eje corto","STD eje corto", "Resistencia eje largo","STD eje largo","Conductividad"]
df_seis_contactos = pd.DataFrame(columns=Columnas)

start_time = time.time()
for i in range(1,4):
    
    sigma_filas = 1 # Este va entre las filas
    sigma_columnas = i #Este es el que va en entre las columnas
    conductividades = (sigma_filas, sigma_columnas)

    
    
    with tqdm(total=101*50) as pbar_h:
        for Dominios_A in range(0,101):
            
            pesos = (Dominios_A,100 - Dominios_A)
            
            
            # Iteraciones = np.zeros(50)
            Resistencia_eje_corto = np.zeros(50)
            Resistencia_eje_largo = np.zeros(50)
            
            for _ in range(50):
                
                Matriz_dominios = Distribucion_dominios(filas, columnas, pesos)
                
                res_corta, res_larga = Seis_contactos(filas, columnas, Matriz_dominios, conductividades, radio)
                
                Resistencia_eje_corto[_] = res_corta
                Resistencia_eje_largo[_] = res_larga
                pbar_h.update(1)
                
        
            Promedio_eje_corto = np.mean(Resistencia_eje_corto)
            Promedio_eje_largo = np.mean(Resistencia_eje_largo)
            Desviacion_eje_corto = np.std(Resistencia_eje_corto)
            Desviacion_eje_largo = np.std(Resistencia_eje_largo)
            
            # Create a temporary DataFrame with the values
            temp_df = pd.DataFrame([[Dominios_A, Promedio_eje_corto, Desviacion_eje_corto, Promedio_eje_largo,Desviacion_eje_largo, conductividades]], columns=Columnas)

            # Check if temp_df contains any empty or all-NA entries
            if not temp_df.isnull().values.all():
                # Concatenate the temporary DataFrame with the main DataFrame
                df_seis_contactos = pd.concat([df_seis_contactos, temp_df], ignore_index=True)
            
              
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
    # mediciones.append(Promedios)
    
#%%

df_seis_contactos[r'$\frac{R_{corto}-R_{largo}}{R_{corto}+R_{largo}}$'] = (df_seis_contactos["Resistencia eje corto"] - df_seis_contactos["Resistencia eje largo"]) / (df_seis_contactos["Resistencia eje corto"] + df_seis_contactos["Resistencia eje largo"])
plt.figure(figsize=(14,7))
plt.grid()
sns.scatterplot(data=df_seis_contactos, x="Proporcion de Dominios", y=r'$\frac{R_{corto}-R_{largo}}{R_{corto}+R_{largo}}$', hue="Conductividad",style="Conductividad")
#%%

plt.figure(figsize=(14,7))
plt.title('Normalizado')
plt.xlabel('Proporcion de Dominios de tipo A')
plt.ylabel(r'$\frac{R_{corto}-R_{largo}}{R_{corto}+R_{largo}}$', fontsize=14)
plt.grid()
# plt.vlines(x= 50, ymin = -0.9, yma    x=0.2,linestyles="dashed",color="k")

for i in range(0,3):
    plt.plot(mediciones[i],label = f'Conductividad = (1,{i + 1})')
    plt.legend(fancybox=True, shadow=True)
#%%

# Save the dictionary to a file
with open('Simlaciones seis contactos', 'wb') as file:
    pickle.dump(df_seis_contactos, file)
    