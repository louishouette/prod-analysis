import os
import matplotlib.pyplot as plt
import pandas as pd
import arviz as az

def run_bayesian_hierarchical_model(data, rampup_data, output_dir='generated', max_age=12, verbose=True):
    """
    Exécute le pipeline du modèle bayésien hiérarchique :
    - Ajuste la courbe de ramp-up
    - Construit et fit le modèle bayésien
    - Projette la production future
    - Sauvegarde les résultats et graphiques
    """
    from models.gompertz.model import fit_ramp_up_curve, build_bayesian_model, sample_posterior, project_future_production
    
    # Vérifier si les dataframes ont le bon format (multiple colonnes)
    # Si une seule colonne et délimiteur potentiellement incorrect, recharger avec le bon délimiteur
    if isinstance(data, pd.DataFrame) and data.shape[1] == 1:
        try:
            # La première colonne contient en fait toutes les données avec délimiteur ';'
            first_col = data.columns[0]
            # Récupérer le contenu comme texte et le recharger avec le bon délimiteur
            csv_content = '\n'.join([first_col] + data[first_col].tolist())
            data = pd.read_csv(pd.io.common.StringIO(csv_content), delimiter=';')
            if verbose:
                print(f"Correction du format CSV avec délimiteur ';' réussie pour 'data'.")
        except Exception as e:
            if verbose:
                print(f"Attention: Impossible de corriger le format CSV de 'data': {e}")
    
    if isinstance(rampup_data, pd.DataFrame) and rampup_data.shape[1] == 1:
        try:
            # La première colonne contient en fait toutes les données avec délimiteur ';'
            first_col = rampup_data.columns[0]
            # Récupérer le contenu comme texte et le recharger avec le bon délimiteur
            csv_content = '\n'.join([first_col] + rampup_data[first_col].tolist())
            rampup_data = pd.read_csv(pd.io.common.StringIO(csv_content), delimiter=';')
            if verbose:
                print(f"Correction du format CSV avec délimiteur ';' réussie pour 'rampup_data'.")
        except Exception as e:
            if verbose:
                print(f"Attention: Impossible de corriger le format CSV de 'rampup_data': {e}")

    plots_dir = os.path.join(output_dir, 'plots', 'bayesian_hierarchical')
    projections_dir = os.path.join(output_dir, 'data', 'projections')
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(projections_dir, exist_ok=True)

    if verbose:
        print("Ajustement de la courbe de montée en production...")
    A_fixed, beta_fixed, gamma_fixed, r_squared = fit_ramp_up_curve(rampup_data)
    if verbose:
        print(f"Paramètres ramp-up : A={A_fixed:.2f}, beta={beta_fixed:.2f}, gamma={gamma_fixed:.3f}, R²={r_squared:.3f}")
        print("Construction du modèle bayésien hiérarchique...")
    hierarchical_model, model_data, parcels = build_bayesian_model(data, gamma_fixed, A_mean=A_fixed, beta_mean=beta_fixed)
    
    if verbose:
        print("Échantillonnage de la postérieure (PyMC)...")
    trace = sample_posterior(hierarchical_model, chains=2, tune=1000, draws=1000)
    
    if verbose:
        print("Sauvegarde du résumé ArviZ...")
    summary = az.summary(trace)
    summary_path = os.path.join(projections_dir, 'bayesian_hierarchical_summary.csv')
    summary.to_csv(summary_path)
    
    if verbose:
        print(f"Projection de la production future par parcelle et âge...")
    proj_df = project_future_production(trace, model_data, parcels, gamma_fixed, max_age=max_age)
    proj_path = os.path.join(projections_dir, 'bayesian_hierarchical_projections.csv')
    proj_df.to_csv(proj_path)
    
    if verbose:
        print("Génération des graphiques de projection...")
    plt.figure(figsize=(14, 8))
    for parcel in proj_df.columns:
        plt.plot(proj_df.index, proj_df[parcel], marker='o', label=parcel)
    plt.title('Projection future par parcelle (Bayésien hiérarchique)')
    plt.xlabel('Âge (années)')
    plt.ylabel('Production au plant (g)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plot_proj_path = os.path.join(plots_dir, 'projection_par_parcelle.png')
    try:
        plt.savefig(plot_proj_path, dpi=300)
        if verbose:
            print(f"Graphique de projection sauvegardé dans {plot_proj_path}")
    except Exception as e:
        if verbose:
            print(f"Erreur: Impossible de sauvegarder le graphique de projection: {e}")
    plt.close()
    
    # Génération des traceplots (ArviZ)
    if verbose:
        print("Génération des traceplots (ArviZ)...")
    az.plot_trace(trace, var_names=["A_mu", "A_sigma", "beta_mu", "beta_sigma", "sigma"])
    plt.tight_layout()
    traceplot_path = os.path.join(plots_dir, 'arviz_traceplots.png')
    try:
        plt.savefig(traceplot_path, dpi=300)
        if verbose:
            print(f"Traceplots sauvegardés dans {traceplot_path}")
    except Exception as e:
        if verbose:
            print(f"Erreur: Impossible de sauvegarder les traceplots: {e}")
    plt.close()
    
    # Génération du pairplot (ArviZ)
    if verbose:
        print("Génération du pairplot (ArviZ)...")
    az.plot_pair(trace, var_names=["A_mu", "beta_mu", "sigma"], kind='kde', marginals=True)
    plt.tight_layout()
    pairplot_path = os.path.join(plots_dir, 'arviz_pairplot.png')
    try:
        plt.savefig(pairplot_path, dpi=300)
        if verbose:
            print(f"Pairplot sauvegardé dans {pairplot_path}")
    except Exception as e:
        if verbose:
            print(f"Erreur: Impossible de sauvegarder le pairplot: {e}")
    plt.close()
    
    if verbose:
        print("Rapport bayésien hiérarchique généré avec succès.")
    return summary, proj_df, trace
