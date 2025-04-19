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
    Utilise l'Âge Brut (sans pénalité) comme variable fondamentale pour les prévisions.
    Réinitialise l'index pour éviter tout problème d'index dupliqué.
    """
    try:
        data = pd.read_csv(file_path, sep=';', index_col=None)
        data = data.reset_index(drop=True)
        if 'Saison' in data.columns:
            if 'Age Brut' in data.columns:
                data = data.rename(columns={'Age Brut': 'Age'})
                print("Note: 'Age Brut' renommé en 'Age' pour les analyses.")
            elif 'Age' not in data.columns:
                raise ValueError(f"Ni 'Age Brut' ni 'Age' n'est présent dans {file_path}")
            required_cols = ['Parcelle', 'Espèce', 'Production au plant (g)']
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Colonnes manquantes dans {file_path}: {', '.join(missing)}")
        elif 'Age' in data.columns:
            if 'Production au plant (g)' not in data.columns:
                raise ValueError("La colonne 'Production au plant (g)' est manquante dans le fichier de ramp-up")
        else:
            raise ValueError(f"Format de fichier non reconnu: {file_path}")
        if data.index.duplicated().any():
            print(f"ALERTE: Index dupliqués détectés après chargement de {file_path}, correction...")
            data = data.reset_index(drop=True)
        print(f"Données chargées avec succès depuis {file_path}. Shape: {data.shape}")
        return data
    except Exception as e:
        print(f"Erreur lors du chargement des données depuis {file_path}: {str(e)}")
        raise

    """
    Charge les données à partir d'un chemin de fichier spécifié.
    Utilise l'Âge Brut (sans pénalité) comme variable fondamentale pour les prévisions,
    conformément aux analyses antérieures qui ont démontré que le facteur de pénalité
    n'était pas scientifiquement fondé.
    
    Paramètres:
    -----------
    file_path : chemin vers le fichier CSV à charger
    
    Retourne:
    ---------
    DataFrame pandas contenant les données chargées
    """
    try:
        # Forcer la réinitialisation de l'index pour éviter tout problème avec des index dupliqués
        data = pd.read_csv(file_path, sep=';', index_col=None)
        
        # Vérifier si c'est un fichier de production ou de ramp-up
        if 'Saison' in data.columns:
            # Fichier de production
            # Vérifier si on doit utiliser 'Age Brut' ou 'Age'
            if 'Age Brut' in data.columns:
                data = data.rename(columns={'Age Brut': 'Age'})
                print("Note: 'Age Brut' renommé en 'Age' pour les analyses.")
            elif 'Age' not in data.columns:
                raise ValueError(f"Ni 'Age Brut' ni 'Age' n'est présent dans {file_path}")
            
            # Vérifier les autres colonnes requises
            required_cols = ['Parcelle', 'Espèce', 'Production au plant (g)'] 
            missing = [col for col in required_cols if col not in data.columns]
            if missing:
                raise ValueError(f"Colonnes manquantes dans {file_path}: {', '.join(missing)}")
        
        elif 'Age' in data.columns:
            # Fichier de ramp-up (courbe de référence)
            if 'Production au plant (g)' not in data.columns:
                raise ValueError("La colonne 'Production au plant (g)' est manquante dans le fichier de ramp-up")
            
        else:
            raise ValueError(f"Format de fichier non reconnu: {file_path}")
        
        print(f"Données chargées avec succès depuis {file_path}. Shape: {data.shape}")
        return data
        
    except Exception as e:
        print(f"Erreur lors du chargement des données depuis {file_path}: {str(e)}")
        raise

def prepare_data(data):
    """
    Prépare les données pour l'analyse en utilisant l'Âge Brut (sans pénalité).
    Réinitialise l'index à chaque étape de filtrage pour éviter tout problème d'index dupliqué.
    """
    try:
        df = data.copy()
        # Diagnostic avancé
        print(f"[DEBUG] Index avant toute opération: unique={df.index.is_unique}, duplicated={df.index.duplicated().sum()}, type={type(df.index)}")
        print(f"[DEBUG] Colonnes: {df.columns.tolist()}")
        print(f"[DEBUG] Types de colonnes: {df.dtypes}")
        # Colonnes dupliquées ?
        duplicated_cols = df.columns[df.columns.duplicated()].tolist()
        if duplicated_cols:
            print(f"[DEBUG] Colonnes dupliquées détectées: {duplicated_cols}, suppression...")
            df = df.loc[:, ~df.columns.duplicated()]
        # Forcer un index RangeIndex simple
        if not isinstance(df.index, pd.RangeIndex):
            print(f"[DEBUG] Conversion de l'index en RangeIndex")
            df.index = pd.RangeIndex(len(df))
        if not df.index.is_unique:
            print("[DEBUG] Correction d'index dupliqué AVANT tout filtrage...")
            df = df.reset_index(drop=True)
        initial_rows = len(df)
        if 'Age' not in df.columns or 'Production au plant (g)' not in df.columns:
            missing_cols = []
            if 'Age' not in df.columns: missing_cols.append('Age')
            if 'Production au plant (g)' not in df.columns: missing_cols.append('Production au plant (g)')
            raise ValueError(f"Colonnes manquantes pour la préparation des données: {', '.join(missing_cols)}")
        # Filtrage âge > 0
        print(f"[DEBUG] Index avant filtrage âge: unique={df.index.is_unique}, duplicated={df.index.duplicated().sum()}")
        if not df.index.is_unique:
            print(f"[DEBUG] Index dupliqué AVANT filtrage âge: {df.index[df.index.duplicated()].tolist()}")
            df.index = pd.RangeIndex(len(df))
        df = df[df['Age'] > 0]
        df.index = pd.RangeIndex(len(df))
        print(f"Filtre par âge: {len(df)} lignes conservées sur {initial_rows}")

        # Filtrage production non manquante
        print(f"[DEBUG] Index avant dropna: unique={df.index.is_unique}, duplicated={df.index.duplicated().sum()}")
        if not df.index.is_unique:
            print(f"[DEBUG] Index dupliqué AVANT dropna: {df.index[df.index.duplicated()].tolist()}")
            df.index = pd.RangeIndex(len(df))
        df = df.dropna(subset=['Production au plant (g)'])
        df.index = pd.RangeIndex(len(df))
        print(f"Filtre par production non-manquante: {len(df)} lignes conservées sur {initial_rows}")

        # Filtrage production >= 0
        print(f"[DEBUG] Index avant filtrage production positive: unique={df.index.is_unique}, duplicated={df.index.duplicated().sum()}")
        if not df.index.is_unique:
            print(f"[DEBUG] Index dupliqué AVANT filtrage production positive: {df.index[df.index.duplicated()].tolist()}")
            df.index = pd.RangeIndex(len(df))
        df = df[df['Production au plant (g)'] >= 0]
        df.index = pd.RangeIndex(len(df))
        print(f"Filtre par production positive: {len(df)} lignes conservées sur {initial_rows}")

        print(f"[DEBUG] Index final: unique={df.index.is_unique}, duplicated={df.index.duplicated().sum()}")
        print(f"Données préparées avec succès: {len(df)} lignes valides sur {initial_rows} au total")
        return df
    except Exception as e:
        print(f"Erreur lors de la préparation des données: {str(e)}")
        raise

    """
    Prépare les données pour l'analyse en utilisant l'Âge Brut (sans pénalité).
    
    Cette fonction implante la décision stratégique d'utiliser l'Âge Brut comme variable 
    fondamentale pour grouper les lots et estimer la progression de la production, 
    conformément aux conclusions des analyses antérieures qui ont démontré que le 
    facteur de pénalité appliqué à l'âge des arbres n'était pas fondé.
    
    Paramètres:
    -----------
    data : DataFrame de données brutes
    
    Retourne:
    ---------
    DataFrame de données filtrées et préparées
    """
    try:
        # Créer une copie pour éviter de modifier les données originales
        # et réinitialiser l'index pour éviter tout problème
        df = data.copy().reset_index(drop=True)
        
        # Appliquer les filtres pour les données valides
        initial_rows = len(df)
        
        # 1. S'assurer que toutes les colonnes nécessaires existent
        if 'Age' not in df.columns or 'Production au plant (g)' not in df.columns:
            missing_cols = []
            if 'Age' not in df.columns: missing_cols.append('Age')
            if 'Production au plant (g)' not in df.columns: missing_cols.append('Production au plant (g)')
            raise ValueError(f"Colonnes manquantes pour la préparation des données: {', '.join(missing_cols)}")
        
        # 2. Filtrer les données en utilisant des méthodes plus simples pour éviter les problèmes d'indexation
        # Étape 1: Filtrer par âge > 0
        df = df[df['Age'] > 0].copy()
        print(f"Filtre par âge: {len(df)} lignes conservées sur {initial_rows}")
        
        # Étape 2: Supprimer les lignes avec production manquante
        df = df.dropna(subset=['Production au plant (g)'])
        print(f"Filtre par production non-manquante: {len(df)} lignes conservées sur {initial_rows}")
        
        # Étape 3: Filtrer les productions >= 0
        df = df[df['Production au plant (g)'] >= 0].copy()
        print(f"Filtre par production positive: {len(df)} lignes conservées sur {initial_rows}")
        
        print(f"Données préparées avec succès: {len(df)} lignes valides sur {initial_rows} au total")
        return df
        
    except Exception as e:
        print(f"Erreur lors de la préparation des données: {str(e)}")
        raise

def setup_directories():
    """Crée les répertoires nécessaires pour les sorties"""
    directories = [
        'generated/data/projections',
        'generated/plots/production_projections',
        'generated/plots/model_diagnostics',
        'generated/plots/model_comparisons',
        'generated/plots/precocity',
        'generated/plots/truffle_weight',
        'generated/plots/truffle_quantity',
        'generated/plots/total_production',
        'generated/plots/correlations',
        'generated/plots/planting_effects'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        
    print("Répertoires de sortie créés avec succès.")
