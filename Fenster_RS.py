# -*- coding: utf-8 -*-
"""
Created on Wed Oct 31 09:59:09 2018

@author: schmoll
"""

import numpy as np

def Rechteck(N):
    x = np.ones(N)
    Skalierung = 1
    FensterName = 'Rechteck'
    return (x, Skalierung, FensterName)

def Hann(N):
    # Haversine-Funktion (Hann-Fenster)
    x = 1/2 * (1 - np.cos(2*np.pi*(np.arange(0,N) / (N-1))))
    Skalierung = 1/0.5
    FensterName = 'Hann'
    return (x, Skalierung, FensterName)

def Hamming(N):
    x = 0.54 - 0.46*(np.cos(2*np.pi*(np.arange(0,N) / (N-1))))
    Skalierung = 1/0.54
    FensterName = 'Hamming'
    return (x, Skalierung, FensterName)

def Blackman(N):
    alpha = 0.16
    x = (1-alpha)/2 - 1/2*(np.cos(2*np.pi*(np.arange(0,N) / (N-1)))) + \
                  alpha/2*(np.cos(4*np.pi*(np.arange(0,N) / (N-1))))
    Skalierung = 1/((1-alpha)/2)
    FensterName = 'Blackman'
    return (x, Skalierung, FensterName)

def FlatTop(N):
    n = np.arange(0,N)
    x =   0.21557895\
        - 0.41663158*(np.cos(2*np.pi*(n / (N-1))))\
        + 0.277263158*(np.cos(4*np.pi*(n / (N-1))))\
        - 0.083578947*(np.cos(6*np.pi*(n / (N-1))))\
        + 0.006947368*(np.cos(8*np.pi*(n / (N-1))))
    Skalierung = 1/0.21557895
    FensterName = 'FlatTop'
    return (x, Skalierung, FensterName)

def KaiserBessel(N):
    n = np.arange(0,N)
    x =   0.4021\
        - 0.4986*(np.cos(2*np.pi*(n / (N-1))))\
        + 0.0981*(np.cos(4*np.pi*(n / (N-1))))\
        - 0.0012*(np.cos(6*np.pi*(n / (N-1))))
    Skalierung = 1/0.4021
    FensterName = 'KaiserBessel'
    return (x, Skalierung, FensterName)

def Gauss(N):
    sigma = 0.4
    n = np.arange(0,N)
    x =   np.exp(-1/2*((n-N/2)/(sigma*N/2))**2)
    Skalierung = 1/0.5
    FensterName = 'Gauß'
    return (x, Skalierung, FensterName)

def Dreieck(N):
    n = np.arange(0,N)
    x = np.zeros(N)   # Alles auf 0
    x[:N//2] = 2*n[:N//2]/N
    x[N//2:] = 2 - 2*n[N//2:]/N
    Skalierung = 1/0.5
    FensterName = 'Dreieck'
    return (x, Skalierung, FensterName)