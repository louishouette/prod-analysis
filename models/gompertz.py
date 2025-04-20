#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import pymc as pm
import arviz as az

# Fonction de croissance Gompertz
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

def fit_ramp_up_curve(rampup_data):
    """
    Ajuste le modèle Gompertz à la courbe de montée en production.
    Retourne les paramètres optimaux et les métriques d'ajustement.
    """
    # Extraction des données
    ages = rampup_data['Age'].values
    productions = rampup_data['Production au plant (g)'].values
    
    # Estimation initiale des paramètres
    A_init = productions.max()  # asymptote ~ production maximale
    beta_init = 10.0  # valeur initiale typique
    gamma_init = 0.5  # valeur initiale typique
    
    # Ajustement de la courbe
    try:
        params, pcov = curve_fit(
            gompertz, 
            ages, 
            productions, 
            p0=[A_init, beta_init, gamma_init],
            bounds=([0, 0, 0], [200, 100, 1])
        )
        
        # Extraction des paramètres optimaux
        A_opt, beta_opt, gamma_opt = params
        
        # Calcul de R²
        productions_pred = gompertz(ages, A_opt, beta_opt, gamma_opt)
        ss_tot = np.sum((productions - np.mean(productions)) ** 2)
        ss_res = np.sum((productions - productions_pred) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return A_opt, beta_opt, gamma_opt, r_squared
        
    except Exception as e:
        print(f"Erreur lors de l'ajustement de la courbe: {e}")
        return None, None, None, None

def build_bayesian_model(data, gamma_fixed, A_mean=None, beta_mean=None):
    """
    Construction du modèle bayésien hiérarchique pour les paramètres spécifiques à chaque parcelle.
    Utilise l'Âge Brut comme variable d'âge.
    """
    # Création d'une liste unique des parcelles
    parcels = data['Parcelle'].unique()
    n_parcels = len(parcels)
    
    # Préparation des données pour le modèle
    model_data = {}
    model_data['age'] = data['Age'].values
    model_data['production'] = data['Production au plant (g)'].values
    model_data['parcel_idx'] = pd.Categorical(data['Parcelle']).codes
    
    # Valeurs par défaut pour les priors
    if A_mean is None:
        A_mean = 60.0
    if beta_mean is None:
        beta_mean = 15.0
    
    # Construction du modèle
    with pm.Model() as hierarchical_model:
        # Hyperpriors pour les paramètres de parcelle
        A_mu = pm.Normal('A_mu', mu=A_mean, sigma=20.0)
        A_sigma = pm.HalfNormal('A_sigma', sigma=10.0)
        
        beta_mu = pm.Normal('beta_mu', mu=beta_mean, sigma=5.0)
        beta_sigma = pm.HalfNormal('beta_sigma', sigma=2.0)
        
        # Paramètres spécifiques aux parcelles
        A = pm.Normal('A', mu=A_mu, sigma=A_sigma, shape=n_parcels)
        beta = pm.Normal('beta', mu=beta_mu, sigma=beta_sigma, shape=n_parcels)
        
        # Paramètre gamma fixé basé sur la courbe de montée en production
        # (on utilise gamma_fixed directement, pas besoin de pm.Deterministic)
        
        # Modèle de vraisemblance
        sigma = pm.HalfNormal('sigma', sigma=10.0)
        
        # Prédictions du modèle
        mu = pm.Deterministic(
            'mu', 
            A[model_data['parcel_idx']] * 
            pm.math.exp(-beta[model_data['parcel_idx']] * 
                        pm.math.exp(-gamma_fixed * model_data['age']))
        )
        
        # Vraisemblance (distribution des observations)
        obs = pm.Normal('obs', mu=mu, sigma=sigma, observed=model_data['production'])
    
    return hierarchical_model, model_data, parcels

def sample_posterior(model, cores=1, chains=2, tune=1000, draws=1000):
    """
    Échantillonnage de la distribution postérieure du modèle bayésien.
    """
    with model:
        # Initialisation du sampler NUTS
        trace = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            return_inferencedata=True
        )
    
    return trace

def project_future_production(trace, model_data, parcels, gamma_fixed, max_age=12):
    """
    Projette la production future pour chaque parcelle en fonction de l'âge.
    Utilise les échantillons de la distribution postérieure.
    """
    # Extraction des échantillons postérieurs
    A_samples = az.extract(trace, var_names="A").to_numpy()
    beta_samples = az.extract(trace, var_names="beta").to_numpy()
    
    # Calcul des moyennes des échantillons postérieurs pour chaque parcelle
    A_mean = A_samples.mean(axis=0)
    beta_mean = beta_samples.mean(axis=0)
    
    # Projection de la production pour chaque parcelle et âge
    ages = np.arange(1, max_age + 1)
    projections = {}
    
    for i, parcel in enumerate(parcels):
        growth_curve = [gompertz(age, A_mean[i], beta_mean[i], gamma_fixed) for age in ages]
        projections[parcel] = growth_curve
    
    # Conversion en DataFrame pour faciliter l'analyse
    proj_df = pd.DataFrame(projections, index=ages)
    proj_df.index.name = 'Age'
    
    return proj_df
