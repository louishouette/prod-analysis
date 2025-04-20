#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings

# Ignorer les avertissements pour une sortie plus propre
warnings.filterwarnings('ignore')

def load_data(file_path):
    """
    Charge les données à partir d'un chemin de fichier spécifié.
    Utilise l'âge Brut (sans pénalité) comme variable fondamentale pour les prévisions,
    conformément aux analyses antérieures qui ont démontré que le facteur de pénalité
    n'était pas scientifiquement fondu.
    """
    try:
        # Chargement avec gestion de l'encodage UTF-8 explicite
        data = pd.read_csv(file_path, sep=';', index_col=None, encoding='utf-8')
        data = data.reset_index(drop=True)
        
        # Normalisation des noms de colonnes pour éviter les problèmes d'encodage
        # Convertir tous les noms de colonnes en 'ascii' (normalisation)
        normalized_columns = {}
        for col in data.columns:
            # Utilisation d'une version simplifiée pour les colonnes problématiques
            if 'Esp' in col:
                normalized_columns[col] = 'Espece'  # Suppression de l'accent
            elif 'Pénali' in col:
                normalized_columns[col] = 'Penalite'  # Suppression de l'accent
        
        # Application des renommages si nécessaire
        if normalized_columns:
            data = data.rename(columns=normalized_columns)
        
        # Traitement de 'Age Brut'
        if 'Age Brut' in data.columns:
            data = data.rename(columns={'Age Brut': 'Age'})
            print("Note: 'Age Brut' renommé en 'Age' pour les analyses.")
        elif 'Age' not in data.columns:
            raise ValueError(f"Ni 'Age Brut' ni 'Age' n'est présent dans {file_path}")
            
        # Vérification des colonnes requises
        if 'Saison' in data.columns:
            required_cols = ['Parcelle', 'Espece', 'Production au plant (g)']  # 'Espece' sans accent
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Colonnes manquantes dans {file_path}: {', '.join(missing)}")
        elif 'Age' in data.columns:
            if 'Production au plant (g)' not in data.columns:
                raise ValueError("La colonne 'Production au plant (g)' est manquante dans le fichier de ramp-up")
        else:
            raise ValueError(f"Format de fichier non reconnu: {file_path}")
            
        # Vérification des index dupliqués
        if data.index.duplicated().any():
            print(f"ALERTE: Index dupliqués détectés après chargement de {file_path}, correction...")
            data = data.reset_index(drop=True)
            
        print(f"Données chargées avec succès depuis {file_path}. Shape: {data.shape}")
        return data
        
    except Exception as e:
        print(f"Erreur lors du chargement des données depuis {file_path}: {str(e)}")
        raise

def prepare_data(data):
    """
    Prépare les données pour l'analyse en utilisant l'âge Brut (sans pénalité).
    Réinitialise l'index à chaque étape de filtrage pour éviter tout problème d'index dupliqué.
    """
    # Vérification des colonnes obligatoires
    required_cols = ['Saison', 'Parcelle', 'Espece', 'Age', 'Production au plant (g)']
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        raise ValueError(f"Colonnes manquantes dans les données: {', '.join(missing_cols)}")
    
    # Réinitialisation de l'index pour éviter les problèmes d'index dupliqué
    data = data.reset_index(drop=True)
    print(f"[DEBUG] Index avant toute opération: unique={data.index.is_unique}, duplicated={data.index.duplicated().sum()}, type={type(data.index)}")
    
    # Identification des colonnes dupliquées (même nom, différentes données)
    print(f"[DEBUG] Colonnes: {data.columns.tolist()}")
    print(f"[DEBUG] Types de colonnes: {data.dtypes}")
    
    column_names = data.columns.tolist()
    duplicate_columns = [name for name in column_names if column_names.count(name) > 1]
    
    if duplicate_columns:
        print(f"[DEBUG] Colonnes dupliquées détectées: {duplicate_columns}, suppression...")
        # Garder uniquement la première occurrence de chaque colonne dupliquée
        data = data.loc[:, ~data.columns.duplicated()]
    
    # Copie de sécurité pour ne pas modifier les données d'origine
    valid_data = data.copy()
    valid_data = valid_data.reset_index(drop=True)
    print(f"[DEBUG] Index avant filtrage âge: unique={valid_data.index.is_unique}, duplicated={valid_data.index.duplicated().sum()}")
    
    # Filtrer les arbres d'âge strictement positif
    valid_data = valid_data[valid_data['Age'] > 0]
    print(f"Filtre par âge: {len(valid_data)} lignes conservées sur {len(data)}")
    
    # Réinitialisation de l'index
    valid_data = valid_data.reset_index(drop=True)
    print(f"[DEBUG] Index avant dropna: unique={valid_data.index.is_unique}, duplicated={valid_data.index.duplicated().sum()}")
    
    # Filtrer les observations où la production au plant est disponible
    valid_data = valid_data.dropna(subset=['Production au plant (g)'])
    print(f"Filtre par production non-manquante: {len(valid_data)} lignes conservées sur {len(data)}")
    
    # Réinitialisation de l'index
    valid_data = valid_data.reset_index(drop=True)
    print(f"[DEBUG] Index avant filtrage production positive: unique={valid_data.index.is_unique}, duplicated={valid_data.index.duplicated().sum()}")
    
    # Filtrer les observations où la production au plant est positive
    valid_data = valid_data[valid_data['Production au plant (g)'] >= 0]
    print(f"Filtre par production positive: {len(valid_data)} lignes conservées sur {len(data)}")
    
    # Réinitialisation finale de l'index
    valid_data = valid_data.reset_index(drop=True)
    print(f"[DEBUG] Index final: unique={valid_data.index.is_unique}, duplicated={valid_data.index.duplicated().sum()}")
    
    print(f"Données prêparées avec succès: {len(valid_data)} lignes valides sur {len(data)} au total")
    
    # Message spécial concernant l'âge Brut
    print("\nNote importante: Utilisation de l'âge Brut comme variable fondamentale pour")
    print("grouper les lots et estimer la progression de la production, conformément aux")
    print("conclusions des analyses antérieures qui ont démontré que le facteur de pénalité")
    print("appliqué à l'âge des arbres n'était pas scientifiquement fondé.\n")
    
    return valid_data


def setup_directories():
    """
    Créer les répertoires nécessaires pour les sorties
    """
    # Répertoires pour les graphiques
    plot_dirs = [
        'generated/plots',
        'generated/plots/ramp_up',
        'generated/plots/models',
        'generated/plots/comparison'
    ]
    
    # Répertoires pour les données
    data_dirs = [
        'generated/data',
        'generated/data/projections',
        'generated/data/comparison'
    ]
    
    # Création des répertoires
    for directory in plot_dirs + data_dirs:
        os.makedirs(directory, exist_ok=True)
    
    print("Répertoires de sortie créés avec succès.")
