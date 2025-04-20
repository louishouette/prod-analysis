#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Fonction de croissance logistique (alternative à Gompertz)
def logistic_growth(age, K, r, t0):
    """
    Modèle de croissance logistique.
    
    Paramètres:
    age : âge des arbres (en années)
    K : capacité de charge (production maximale)
    r : taux de croissance intrinsèque
    t0 : point d'inflexion
    """
    return K / (1 + np.exp(-r * (age - t0)))

# Fonction de croissance exponentielle modifiée (alternative à Gompertz)
def modified_exp_growth(age, a, b, c):
    """
    Modèle de croissance exponentielle modifiée.
    
    Paramètres:
    age : âge des arbres (en années)
    a : asymptote (production maximale)
    b : facteur d'échelle
    c : taux de croissance
    """
    return a * (1 - np.exp(-b * age)) ** c

# Fonction de croissance non-paramétrique basée sur splines (alternative à Gompertz)
def spline_growth(age, *coeffs):
    """
    Modèle de croissance basé sur des splines cubiques.
    
    Paramètres:
    age : âge des arbres (en années)
    coeffs : coefficients des splines
    """
    from scipy.interpolate import CubicSpline
    knots = np.linspace(1, 12, len(coeffs))
    cs = CubicSpline(knots, coeffs)
    return cs(age)

def fit_alternative_growth(rampup_data, model_type='logistic'):
    """
    Ajuste un modèle de croissance alternatif à la courbe de montée en production.
    
    Paramètres:
    rampup_data : DataFrame avec les données de montée en production
    model_type : type de modèle ('logistic', 'modified_exp', 'spline')
    
    Retourne:
    Paramètres optimaux et métrique d'ajustement
    """
    # Extraction des données
    ages = rampup_data['Age'].values
    productions = rampup_data['Production au plant (g)'].values
    
    if model_type == 'logistic':
        # Paramètres initiaux pour logistique
        K_init = productions.max() * 1.2  # asymptote légèrement supérieure au max observé
        r_init = 0.8  # taux de croissance initial
        t0_init = 5.0  # point d'inflexion estimé
        
        try:
            params, pcov = curve_fit(
                logistic_growth, 
                ages, 
                productions, 
                p0=[K_init, r_init, t0_init],
                bounds=([0, 0, 0], [200, 5, 12])
            )
            
            # Calcul de R²
            K_opt, r_opt, t0_opt = params
            productions_pred = logistic_growth(ages, K_opt, r_opt, t0_opt)
            
            ss_tot = np.sum((productions - np.mean(productions)) ** 2)
            ss_res = np.sum((productions - productions_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return K_opt, r_opt, t0_opt, r_squared
            
        except Exception as e:
            print(f"Erreur lors de l'ajustement du modèle logistique: {e}")
            return None, None, None, None
            
    elif model_type == 'modified_exp':
        # Paramètres initiaux pour exponentielle modifiée
        a_init = productions.max() * 1.2
        b_init = 0.3
        c_init = 1.5
        
        try:
            params, pcov = curve_fit(
                modified_exp_growth, 
                ages, 
                productions, 
                p0=[a_init, b_init, c_init],
                bounds=([0, 0, 0], [200, 2, 5])
            )
            
            # Calcul de R²
            a_opt, b_opt, c_opt = params
            productions_pred = modified_exp_growth(ages, a_opt, b_opt, c_opt)
            
            ss_tot = np.sum((productions - np.mean(productions)) ** 2)
            ss_res = np.sum((productions - productions_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return a_opt, b_opt, c_opt, r_squared
            
        except Exception as e:
            print(f"Erreur lors de l'ajustement du modèle exponentiel modifié: {e}")
            return None, None, None, None
            
    elif model_type == 'spline':
        # Pour les splines, on utilise directement les valeurs observées comme coefficients initiaux
        try:
            # Utilisation de 5 coefficients pour les splines
            knots = np.linspace(min(ages), max(ages), 5)
            coeffs_init = np.interp(knots, ages, productions)
            
            params, pcov = curve_fit(
                lambda x, *coeffs: spline_growth(x, *coeffs), 
                ages, 
                productions, 
                p0=coeffs_init
            )
            
            # Calcul de R²
            productions_pred = spline_growth(ages, *params)
            
            ss_tot = np.sum((productions - np.mean(productions)) ** 2)
            ss_res = np.sum((productions - productions_pred) ** 2)
            r_squared = 1 - (ss_res / ss_tot)
            
            return params, r_squared
            
        except Exception as e:
            print(f"Erreur lors de l'ajustement des splines: {e}")
            return None, None
    
    else:
        print(f"Type de modèle '{model_type}' non reconnu")
        return None, None, None, None

def project_alternative_growth(params, max_age=12, model_type='logistic'):
    """
    Projette la courbe de croissance selon le modèle alternatif choisi.
    
    Paramètres:
    params : paramètres du modèle
    max_age : âge maximum pour la projection
    model_type : type de modèle ('logistic', 'modified_exp', 'spline')
    
    Retourne:
    DataFrame avec les âges et productions projetées
    """
    ages = np.arange(1, max_age + 1)
    
    if model_type == 'logistic':
        K_opt, r_opt, t0_opt = params[0:3]
        productions = logistic_growth(ages, K_opt, r_opt, t0_opt)
    
    elif model_type == 'modified_exp':
        a_opt, b_opt, c_opt = params[0:3]
        productions = modified_exp_growth(ages, a_opt, b_opt, c_opt)
    
    elif model_type == 'spline':
        productions = spline_growth(ages, *params[0])
    
    else:
        print(f"Type de modèle '{model_type}' non reconnu")
        return None
    
    return pd.DataFrame({'Age': ages, 'Production au plant (g)': productions})
