#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def gompertz(age, A, beta, gamma):
    """
    Modèle de croissance Gompertz.
    
    Paramètres:
    age : âge des arbres (en années)
    A : asymptote (production maximale)
    beta : paramètre de déplacement
    gamma : paramètre de taux de croissance
    """
    return A * np.exp(-beta * np.exp(-gamma * age))

def logistic_growth(age, K, r, t0):
    """
    Modèle de croissance logistique.
    """
    return K / (1 + np.exp(-r * (age - t0)))

def modified_exp_growth(age, a, b, c):
    """
    Modèle de croissance exponentielle modifiée.
    """
    return a * (1 - np.exp(-b * age)) ** c
