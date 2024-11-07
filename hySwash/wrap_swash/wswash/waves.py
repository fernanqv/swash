#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def energy_spectrum(hs, tp, gamma, duration):
    '''
    spec: Dataset with vars for the whole partitions
    S(f,dir) = S(f) * D(dir)
    S(f) ----- D(dir)
    Meshgrid de x freqs - dir - z energy
    '''

    # Defining frequency series - tend length
    freqs = np.linspace(0.02, 1, duration)

    S = []
    fp = 1/tp

    for f in freqs:
        if f <= fp:
            sigma = 0.07
        if f > fp:
            sigma = 0.09

        Beta = (0.06238/(0.23+0.0336*gamma-0.185*(1.9+gamma)**-1))*(1.094-0.01915*np.log(gamma))
        Sf = Beta * (hs**2) * (tp**-4) * (f**-5)*np.exp(-1.25*(tp*f)**-4)*gamma**(np.exp((-(tp*f-1)**2)/(2*sigma**2)))
        S.append(Sf)

    return (S)

def series_Jonswap(waves):
    '''
    Generate surface elevation from PSD df = 1/tendc

    waves - dictionary
              T       - Period (s)
              H       - Height (m)
              gamma   - Jonswap spectrum  peak parammeter
              warmup  - spin up time (s)
              deltat  - delta time (s)
              tendc   - simulation period (s)

    returns 2D numpy array with series time and elevation
    '''

    # waves properties
    hs = waves['H']
    tp = waves['T']
    gamma = waves['gamma']
    warmup = waves['warmup']
    deltat = waves['deltat']
    tendc = waves['tendc']

    # series duration
    # TODO: puede que haya algun problema en esta suma
    duration = tendc + int(warmup)
    time = np.arange(0, duration, deltat)

    # series frequency
    freqs = np.linspace(0.02, 1, duration)
    delta_f = freqs[1] - freqs[0]

    # calculate energy spectrum
    S = energy_spectrum(hs, tp, gamma, duration)

    # series elevation
    teta = np.zeros((len(time)))

    # calculate aij
    for f in range(len(freqs)):
        ai = np.sqrt(S[f] * 2 * delta_f)
        eps = np.random.rand() * (2*np.pi)

        # calculate elevation
        teta = teta + ai * np.cos(2*np.pi*freqs[f] * time + eps)

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:, 0] = time
    series[:, 1] = teta

    return series

def series_regular_monochromatic(waves):
    '''
    Generates monochromatic regular waves series

    waves - dictionary
              T      - Period (s)
              H      - Height (m)
              WL     - Water level (m)
              warmup - spin up time (s)
              deltat - delta time (s)
              tendc  - simulation period (s)

    returns 2D numpy array with series time and elevation
    '''

    # waves properties
    T = waves['T']
    H = waves['H']
    WL = waves['WL']
    warmup = waves['warmup']
    deltat = waves['deltat']
    tendc = waves['tendc']

    # series duration
    duration = tendc + int(warmup)
    time = np.arange(0, duration, deltat)

    # series elevation
    teta = (H/2) * np.cos((2*np.pi/T)*time)

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:, 0] = time
    series[:, 1] = teta

    return series

def series_regular_bichromatic(waves):
    '''
    Generates bichromatic regular waves series

    waves - dictionary
              T1     - Period component 1 (s)
              T2     - Period component 2 (s)
              H      - Height (m)
              WL     - Water level (m)
              warmup - spin up time (s)
              deltat - delta time (s)
              tendc  - simulation period (s)
    '''

    # waves properties
    T1 = waves['T1']
    T2 = waves['T2']
    H = waves['H']
    WL = waves['WL']
    warmup = waves['warmup']
    deltat = waves['deltat']
    tendc = waves['tendc']

    # series duration
    duration = tendc + int(warmup)
    time = np.arange(0, duration, deltat)

    # series elevation
    teta = (H/2) * np.cos((2*np.pi/T1)*time) + (H/2) * np.cos((2*np.pi/T2)*time)

    # generate series dataframe
    series = np.zeros((len(time), 2))
    series[:,0] = time
    series[:,1] = teta

    return series

def waves_dispersion(T, h):
    'Solve the wave dispersion relation'

    L1 = 1
    L2 = ((9.81*T**2)/2*np.pi) * np.tanh(h*2*np.pi/L1)
    umbral = 1

    while umbral > 0.1:
        L2 = ((9.81*T**2)/(2*np.pi)) * np.tanh((h*2*np.pi)/L1)
        umbral = np.abs(L2-L1)
        L1 = L2

    L = L2
    k = (2*np.pi)/L
    c = np.sqrt(9.8*np.tanh(k*h)/k)

    return(L, k, c)


# TODO eliminar
def test_tutorial_commit(nada):
    print(nada)

