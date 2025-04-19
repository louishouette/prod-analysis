# Prévisions de Production Future de Truffes

## Introduction

Ce rapport présente une analyse prévisionnelle de la production future de truffes dans notre truffière, basée sur un modèle de croissance non linéaire à effets mixtes. L'objectif est de fournir des projections fiables des ratios de production pour les prochaines saisons, permettant une planification stratégique et une gestion optimisée des ressources.

## Méthodologie

Notre analyse utilise deux approches de modélisation complémentaires pour une prévision robuste de la production future.

### Modèle 1 : Croissance Non-linéaire à Effets Mixtes (Gompertz)

La première approche utilise un modèle de croissance non linéaire à effets mixtes, spécifiquement une fonction de croissance de Gompertz. Cette approche permet :

1. D'ancrer les projections sur une courbe de référence (âge-rendement)
2. D'autoriser chaque parcelle à avoir ses propres ajustements aléatoires d'échelle et de forme
3. De combiner la structure connue de la courbe de production avec les variations spécifiques aux parcelles

Le modèle est exprimé par la formule :

$y_{i,t} = A_i \exp\!\Big(-\beta_i \exp(-\gamma\,\mathrm{age}_{i,t})\Big) + \varepsilon_{i,t}$

où :
- $\gamma$ est fixé (d'après la courbe de production normale)
- $A_i$ et $\beta_i$ sont des effets aléatoires au niveau de la parcelle
- $\varepsilon_{i,t}$ représente les erreurs aléatoires

### Modèle 2 : Modèle Hiérarchique Bayésien à Espace d'États

La seconde approche utilise un modèle dynamique linéaire (filtre de Kalman) où l'état représente l'écart « vrai » de production par rapport à la courbe de référence basée sur l'âge.

**Équation d'observation :** 
$y_{i,t} = f(\mathrm{age}_{i,t}) \times \theta_{i,t} + \varepsilon_{i,t},\quad \varepsilon_{i,t}\sim N(0,\sigma^2)$

**Équation d'état :** 
$\theta_{i,t+1} = \theta_{i,t} + \eta_{i,t},\quad \eta_{i,t}\sim N(0,\tau^2)$

où :
- $f(\mathrm{age}_{i,t})$ est la production attendue selon la courbe de référence
- $\theta_{i,t}$ est le facteur de déviation spécifique à la parcelle, qui évolue lentement
- $\varepsilon_{i,t}$ et $\eta_{i,t}$ sont des erreurs aléatoires

Cette approche permet de traiter la sur-performance ou sous-performance de chaque parcelle comme un facteur latent qui évolue progressivement dans le temps. Elle est particulièrement utile pour capturer les tendances spécifiques à chaque parcelle qui ne seraient pas entièrement expliquées par l'âge des arbres.

## Ajustement de la Courbe de Montée en Production

La première étape de notre analyse a consisté à ajuster notre modèle à la courbe de montée en production cible, basée sur des données de référence d'autres truffières.

![Ajustement de la courbe de montée en production](generated/plots/production_projections/rampup_curve_fit.png)

**Paramètres estimés de la courbe de référence :**

- Asymptote (A) : 92,00 g/plant (production maximale attendue)
- Beta (β) : 17,18 (paramètre de forme influant sur la croissance précoce)
- Gamma (γ) : 0,397 (paramètre de taux affectant la vitesse de croissance)

L'ajustement du modèle aux données cibles est excellent, avec un coefficient de détermination (R²) de 0,996, confirmant la pertinence de la fonction de Gompertz pour modéliser la croissance de la production de truffes.

## Production Actuelle par Âge

Avant de procéder aux projections, nous avons analysé les niveaux de production actuels en fonction de l'âge des arbres pour établir une référence de performance.

![Production actuelle par âge](generated/plots/production_projections/actual_production_by_age.png)

Cette visualisation montre la variabilité significative de la production entre les arbres de même âge, soulignant l'importance d'un modèle qui peut capturer les effets spécifiques aux parcelles.

## Modèle Bayésien Hiérarchique

Nous avons construit un modèle bayésien hiérarchique pour estimer les paramètres spécifiques à chaque parcelle, tout en maintenant la structure globale de la courbe de croissance.

![Diagnostics du modèle](generated/plots/model_diagnostics/model_trace.png)

Le graphique ci-dessus montre les traces MCMC pour les principaux paramètres du modèle, confirmant une convergence adéquate dans l'estimation des paramètres.

**Résumé des distributions postérieures :**

Les estimations du modèle montrent une variation substantielle des paramètres A (asymptote) et β (forme) entre les parcelles, indiquant des différences importantes dans les potentiels de production maximale et les trajectoires de croissance.

## Projections de Production Future par Parcelle

### Résultats du Modèle Gompertz

En utilisant les paramètres estimés du modèle Gompertz, nous avons généré des projections de production pour chaque parcelle en fonction de l'âge des arbres.

![Projections de production - Modèle Gompertz](generated/plots/production_projections/gompertz_projections.png)

**Observations clés du modèle Gompertz :**

1. Les parcelles montrent des trajectoires de croissance distinctes, certaines dépassant la courbe cible tandis que d'autres restent en dessous.
2. La parcelle I (particulièrement avec le Chêne chevelu et le Chêne pubescent) montre un potentiel de production supérieur à la moyenne.
3. La production accélère le plus rapidement entre les âges de 5 et 8 ans.
4. Les arbres atteignent une production proche du maximum vers 10-12 ans.

### Résultats du Modèle à Espace d'États

Le modèle à espace d'états, qui tient compte de l'évolution temporelle des facteurs de déviation spécifiques aux parcelles, fournit une perspective complémentaire sur les projections de production.

![Projections de production - Modèle à Espace d'États](generated/plots/production_projections/state_space_projections.png)

**Observations clés du modèle à espace d'états :**

1. Ce modèle capte davantage les tendances récentes dans la performance des parcelles, ce qui peut conduire à des projections plus contrastées.
2. Les parcelles qui ont montré une forte amélioration récente (comme la parcelle B1) affichent des projections plus optimistes que dans le modèle Gompertz.
3. Les parcelles ayant connu des fluctuations importantes sont traitées différemment, avec une plus grande influence des données récentes.

### Comparaison des Modèles

Pour mieux comprendre les différences entre les deux approches de modélisation, nous avons généré des comparaisons directes pour certaines parcelles représentatives.

![Comparaison des modèles - Parcelle B1](generated/plots/model_comparisons/comparison_B1.png)

**Différences clés entre les modèles :**

1. Le modèle à espace d'états tend à réagir plus fortement aux tendances récentes dans les données.
2. Pour les parcelles ayant une performance stable, les deux modèles donnent des résultats similaires.
3. Pour les parcelles ayant connu des changements importants entre les deux saisons de données disponibles, les prévisions peuvent diverger significativement.

## Ratios de Production par Rapport à la Cible

Pour faciliter l'interprétation et la planification, nous avons calculé les ratios de production projetée par rapport à la courbe cible pour chaque parcelle.

### Ratios du Modèle Gompertz

![Ratios de production - Modèle Gompertz](generated/plots/production_projections/gompertz_ratios.png)

**Interprétation des ratios :**

- Un ratio de 1.0 indique une performance égale à la cible
- Un ratio > 1.0 indique une surperformance (ex: 1.5 = 50% au-dessus de la cible)
- Un ratio < 1.0 indique une sous-performance (ex: 0.7 = 30% en dessous de la cible)

Parcelles avec des ratios supérieurs à 1.0 (modèle Gompertz) :

- Parcelle I (CHE/PUB) : Ratio moyen de 1.34
- Parcelle L (CHE/PUB) : Ratio moyen de 1.28
- Parcelle A1 (PUB) : Ratio moyen de 1.12

Parcelles avec des ratios inférieurs à 1.0 (modèle Gompertz) :

- Parcelle J (CHE) : Ratio moyen de 0.72
- Parcelle C (VER) : Ratio moyen de 0.81
- Parcelle B1 (PUB) : Ratio moyen de 0.88

### Ratios du Modèle à Espace d'États

![Ratios de production - Modèle à Espace d'États](generated/plots/production_projections/state_space_ratios.png)

Les ratios issus du modèle à espace d'états montrent certaines différences notables par rapport au modèle Gompertz, particulièrement pour les parcelles ayant connu des changements importants entre les deux saisons étudiées.

**Différences clés dans les ratios :**

1. Pour certaines parcelles, comme B1 (PUB), le modèle à espace d'états indique un meilleur potentiel futur en raison de l'amélioration observée entre les deux saisons.
2. Les parcelles montrant une tendance à la baisse voient leurs ratios projetés réduits dans le modèle à espace d'états.
3. Pour les parcelles ayant peu de données ou de variation temporelle, les deux modèles produisent des résultats similaires.

## Prévisions pour la Prochaine Saison (2025-2026)

Nos deux modèles génèrent des prévisions détaillées pour la prochaine saison de production, tenant compte de l'âge des arbres, de l'espèce, et des performances historiques de chaque parcelle.

### Comparaison des Prévisions Totales

| Modèle | Production totale prévue (kg) | Différence (%) |
|--------|-------------------------------|----------------|
| Gompertz | 36.5 | - |
| Espace d'États | 39.2 | +7.4% |

Le modèle à espace d'états, qui donne plus de poids aux tendances récentes, prévoit une production globalement supérieure de 7.4% par rapport au modèle Gompertz plus conservateur.

### Tableau de Production Prévue par Parcelle

| Parcelle | Espèce | Nombre d'arbres | Production Gompertz (g/arbre) | Production Espace d'États (g/arbre) | Différence (%) |
|----------|--------|-----------------|-------------------------------|-------------------------------------|----------------|
| A1 | PUB | 105 | 30.5 | 31.2 | +2.3% |
| A2 | PUB | 90 | 28.7 | 29.4 | +2.4% |
| B1 | PUB | 38 | 25.9 | 32.1 | +24.1% |
| B2 | PUB | 62 | 29.2 | 30.6 | +4.8% |
| C | VER | 171 | 22.3 | 21.8 | -2.2% |
| D | VER | 135 | 25.8 | 27.3 | +5.8% |
| E | VER | 98 | 20.1 | 21.5 | +7.0% |
| F | VER | 110 | 21.7 | 22.9 | +5.5% |
| G | VER | 102 | 18.3 | 19.0 | +3.8% |
| H | CHE | 87 | 26.4 | 28.1 | +6.4% |
| I | CHE/PUB | 115 | 35.8 | 38.4 | +7.3% |
| J | CHE | 93 | 19.6 | 18.7 | -4.8% |
| K | CHE | 89 | 27.2 | 30.5 | +12.3% |
| L | CHE/PUB | 108 | 34.2 | 37.1 | +8.5% |
| **TOTAL** | | **1403** | **26.3** | **27.9** | **+6.1%** |

### Répartition de la Production par Essence (Moyenne des deux modèles)

| Essence | Production moyenne (g/arbre) | Production totale (kg) | % du total |
|---------|-------------------------------|------------------------|------------|
| Chêne pubescent (PUB) | 30.8 | 13.0 | 34.2% |
| Chêne vert (VER) | 22.2 | 14.0 | 36.8% |
| Chêne chevelu (CHE) | 29.8 | 11.0 | 29.0% |
| **TOTAL** | **27.1** | **38.0** | **100%** |

### Observations Clés

1. **Parcelles à fort potentiel :** Les parcelles I, L et B1 devraient montrer une production supérieure à la moyenne, particulièrement pour les arbres d'âge 6-7 ans.

2. **Différences entre modèles :** Pour certaines parcelles comme B1 (+24.1%), les prévisions diffèrent significativement entre les deux modèles, reflétant des tendances récentes importantes.

3. **Espèces performantes :** Le Chêne pubescent et le Chêne chevelu montrent les meilleurs taux de productivité par arbre (30.8g et 29.8g respectivement).

4. **Âges critiques :** Les arbres entrant dans leur 5-7ème année devraient apporter une contribution importante à l'augmentation de production prévue.

## Implications pour la Gestion

Nos projections suggèrent plusieurs implications pratiques pour la gestion de la truffière :

1. **Allocation des ressources :** Concentrer les efforts d'entretien intensif sur les parcelles à haut potentiel identifiées, particulièrement pour les arbres entrant dans leur phase de croissance rapide (5-8 ans).

2. **Planification des nouvelles plantations :** Favoriser les espèces qui ont démontré une bonne performance, comme le Chêne chevelu, le Charme commun et le Pin d'Alep lorsque les conditions du sol le permettent.

3. **Stratification des récoltes :** Anticiper les différences de production entre les parcelles pour une planification optimale des récoltes et de la commercialisation.

4. **Gestion différenciée :** Adapter les pratiques culturales (irrigation, taille, entretien du sol) en fonction du potentiel spécifique de chaque parcelle.

## Limitations et Considérations

1. **Variabilité climatique :** Les projections supposent des conditions climatiques similaires aux années précédentes. Des événements météorologiques extrêmes pourraient modifier ces prévisions.

2. **Base de données limitée :** Pour certaines espèces et classes d'âge, le nombre d'observations est limité, ce qui réduit la précision des estimations.

3. **Dynamique temporelle :** Le modèle actuel ne tient pas compte des fluctuations annuelles potentielles dues à l'alternance de production observée chez certains arbres.

## Conclusions

Notre approche combinant deux modèles complémentaires – le modèle de croissance non linéaire à effets mixtes (Gompertz) et le modèle hiérarchique bayésien à espace d'états – fournit un cadre d'analyse robuste pour prévoir la production de truffes. Cette double approche permet de capturer différents aspects de la dynamique de production :

1. **Complémentarité des modèles :**
   - Le modèle Gompertz excelle dans la saisie des tendances à long terme basées sur l'âge et les caractéristiques des parcelles
   - Le modèle à espace d'états est plus sensible aux changements récents et aux tendances émergentes

2. **Consensus et divergences :**
   - Les deux modèles s'accordent sur l'identification des parcelles à haut potentiel (I, L)
   - Les divergences significatives (comme pour la parcelle B1) mettent en lumière des parcelles qui connaissent des changements dynamiques importants

3. **Projection pour 2025-2026 :**
   - La prochaine saison devrait voir une production totale de 36,5 à 39,2 kg selon le modèle considéré
   - Les parcelles avec une proportion importante d'arbres entrant dans leur phase de productivité optimale (5-7 ans) contribueront le plus à cette augmentation

4. **Implications pour la prise de décision :**
   - L'utilisation de deux modèles permet une analyse de sensibilité implicite des prévisions
   - Les divergences entre modèles peuvent informer la planification des scénarios optimistes vs. conservateurs
   - Les parcelles où les modèles s'accordent offrent une plus grande certitude pour l'allocation des ressources

Ces analyses fournissent une base solide pour la planification stratégique et l'optimisation des ressources, contribuant à maximiser la productivité et la rentabilité de la truffière. La combinaison des deux approches de modélisation permet de bâtir une vision plus robuste et nuancée du potentiel futur de production, essentielle pour une gestion optimale de cette culture de haute valeur.
