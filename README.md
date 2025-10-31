# 🔍 Process Mining & AI Dashboard

**Dashboard Intelligent d'Analyse des Flux de Travail de Bugs** - Identifiez les goulots d'étranglement, optimisez les processus et prévoyez les risques de SLA.

---

## 📋 Table des Matières

1. [Vue d'Ensemble](#vue-densemble)
2. [Technologies Utilisées](#technologies-utilisées)
3. [Structure du Projet](#structure-du-projet)
4. [Fonctionnalités par Page](#fonctionnalités-par-page)
5. [Description des Fichiers](#description-des-fichiers)
6. [Installation](#installation)
7. [Utilisation](#utilisation)

---

## 🎯 Vue d'Ensemble

Ce projet est un **dashboard interactif de Process Mining** combiné à l'**Intelligence Artificielle** pour analyser et optimiser les processus de gestion des bugs dans le développement logiciel. Il permet de :

- **Visualiser** les flux de processus avec des cartes interactives
- **Analyser** les performances et identifier les goulots d'étranglement
- **Prédire** les temps de résolution des bugs avec des modèles ML
- **Prioriser** les catégories de bugs selon leur impact
- **Animer** les processus pour une meilleure compréhension visuelle

---

## 🛠️ Technologies Utilisées

### Framework Principal
- **Streamlit** (≥1.28.0) - Interface utilisateur web interactive

### Process Mining
- **pm4py** (≥2.7.0) - Bibliothèque de Process Mining pour l'analyse des logs d'événements

### Traitement de Données
- **pandas** (≥2.0.0) - Manipulation et analyse de données
- **numpy** (≥1.24.0) - Calculs numériques
- **openpyxl** (≥3.1.0) - Lecture/écriture de fichiers Excel

### Visualisation
- **plotly** (≥5.17.0) - Graphiques interactifs et animations
- **matplotlib** (≥3.7.0) - Graphiques statiques
- **seaborn** (≥0.12.0) - Visualisations statistiques
- **graphviz** (≥0.20.0) - Graphiques de réseau
- **networkx** (≥3.1) - Analyse de réseaux et graphes

### Machine Learning
- **scikit-learn** (≥1.3.0) - Modèles ML (Random Forest, Gradient Boosting, Regression Linéaire)

### Utilitaires
- **scipy** (≥1.11.0) - Outils scientifiques
- **pytz** (≥2023.3) - Gestion des fuseaux horaires

---

## 📁 Structure du Projet

```
DISCO/
├── app.py                          # Application principale Streamlit
├── requirements.txt                # Dépendances Python
├── Dockerfile                      # Configuration Docker
├── docker-compose.yml              # Orchestration Docker
├── setup.py                        # Configuration du package
├── run.bat                         # Script de lancement Windows
├── docker-run.bat                  # Script Docker Windows
│
├── utils/                          # Modules utilitaires
│   ├── __init__.py
│   ├── data_loader.py              # Chargement et validation des données
│   ├── process_mining.py           # Calculs de Process Mining (DFG, variants)
│   ├── metrics.py                  # Calcul des KPIs et statistiques
│   ├── visualizations.py           # Génération de graphiques (Process Map, Heatmap, etc.)
│   ├── feature_engineering.py      # Extraction de features pour ML
│   ├── ml_models.py                # Modèles ML de prédiction
│   ├── category_prioritization.py  # Priorisation des catégories de bugs
│   ├── animation.py                # Animation token replay
│   └── advanced_animation.py       # Animation avancée (Disco-style)
│
├── tests/                          # Tests unitaires
│   ├── __init__.py
│   └── test_data_loader.py         # Tests du module data_loader
│
├── models/                         # Modèles ML sauvegardés
│   └── .gitkeep
│
├── data/                           # Données d'exemple (optionnel)
│
├── exports/                        # Fichiers exportés (reports, etc.)
│
└── docs/                           # Documentation détaillée
    ├── INSTALLATION.md
    ├── USAGE_GUIDE.md
    ├── AI_FEATURES.md
    ├── ANIMATION_GUIDE.md
    └── ...
```

---

## 🎨 Fonctionnalités par Page

### 📊 **Page Principale - Dashboard KPI**

**Utilité** : Vue d'ensemble des indicateurs clés de performance

**Affiche** :
- **Total Bugs** : Nombre total de bugs uniques
- **Temps de Résolution Moyen** : Durée moyenne de résolution des bugs
- **Risque SLA %** : Pourcentage de bugs dépassant le seuil SLA (par défaut 24h)
- **Réouvertures** : Nombre et taux de bugs réouverts
- **Taux de Complétion** : Pourcentage de bugs fermés

**Fonctionnalités** :
- Métriques en temps réel
- Indicateurs visuels avec codes couleurs
- Mise en évidence du bug le plus lent

---

### 🗺️ **Onglet 1 : Process Map (Carte de Processus)**

**Utilité** : Visualisation du flux de processus avec indicateurs de performance

**Fonctionnalités** :
- **Graphe Directly-Follows (DFG)** : Représentation graphique des transitions entre activités
- **Nœuds rectangulaires** : Activités affichées dans des rectangles (au lieu de cercles)
- **Durées affichées** : 
  - Sur les **arcs** : Durée moyenne des transitions
  - Sur les **nœuds** : Temps moyen de traitement par activité
- **Code couleur** :
  - **Rouge** : Durée > 24h (seuil configurable)
  - **Bleu** : Durée ≤ 24h
- **Épaisseur des lignes** : Représente la fréquence des transitions
- **Animation optionnelle** : Tokens animés qui se déplacent le long des arcs
  - Vitesse variable selon la couleur (vert=rapide, orange=normal, rouge=lent)
  - Tokens alignés sur les lignes avec effet de flux continu

**Utilité** :
- Identifier visuellement les goulots d'étranglement
- Comprendre le flux de travail complet
- Détecter les activités critiques (rouges)

---

### 🔥 **Onglet 2 : Heatmap & Bottlenecks (Carte de Chaleur et Goulots d'Étranglement)**

**Utilité** : Identifier les activités lentes par catégorie de bugs

**Fonctionnalités** :
- **Heatmap interactif** : Matrice couleur montrant les durées moyennes par activité et catégorie
- **Détection automatique de goulots d'étranglement** :
  - Activités les plus lentes
  - Catégories problématiques
  - Transitions critiques
- **Statistiques par catégorie** :
  - Temps moyen par catégorie
  - Distribution des priorités
  - Impact sur le processus global

**Utilité** :
- Identifier rapidement les problèmes de performance
- Comparer les performances entre catégories
- Prioriser les actions d'optimisation

---

### 📈 **Onglet 3 : Distributions (Distributions Statistiques)**

**Utilité** : Analyser la distribution des données et les tendances

**Fonctionnalités** :
- **Distribution des durées** : Histogramme des temps de résolution
- **Comparaison par catégorie** : Graphiques en barres comparant les catégories
- **Fréquence des activités** : Nombre d'occurrences de chaque activité
- **Distribution des priorités** : Répartition des bugs par priorité et sévérité
- **Timeline** : Chronologie des événements par bug

**Utilité** :
- Comprendre les patterns de données
- Identifier les anomalies statistiques
- Analyser les tendances temporelles

---

### 🔄 **Onglet 4 : Variants & Loops (Variants et Boucles)**

**Utilité** : Découvrir les différentes variantes du processus et les boucles de retravail

**Fonctionnalités** :
- **Top Process Variants** : Les chemins les plus fréquents dans le processus
- **Statistiques par variant** :
  - Nombre d'occurrences
  - Durée moyenne
  - Taux de conformité SLA
- **Détection de boucles** : Identification des activités répétées (retravail)
- **Activités parallèles** : Détection des activités exécutées en parallèle

**Utilité** :
- Comprendre la variabilité du processus
- Identifier les cas de retravail (loops)
- Optimiser les variantes les plus fréquentes

---

### 📅 **Onglet 5 : Temporal Analysis (Analyse Temporelle)**

**Utilité** : Analyser l'évolution du processus dans le temps

**Fonctionnalités** :
- **Tendances temporelles** : Graphiques montrant l'évolution des métriques dans le temps
- **Analyse saisonnière** : Patterns par jour de la semaine, heure, etc.
- **Comparaisons périodiques** : Comparer différentes périodes
- **Prédictions temporelles** : Tendances futures basées sur les données historiques

**Utilité** :
- Identifier les périodes de charge
- Comprendre les patterns temporels
- Planifier les ressources

---

### 🤖 **Onglet 6 : AI Predictions (Prédictions IA)**

**Utilité** : Utiliser le Machine Learning pour prédire et optimiser

#### 📊 **Sous-onglet : Model Training & Evaluation**

**Fonctionnalités** :
- **Entraînement de modèles ML** :
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - Linear Regression
- **Évaluation des modèles** :
  - MAE (Mean Absolute Error)
  - RMSE (Root Mean Squared Error)
  - R² Score (Coefficient de détermination)
- **Comparaison des modèles** : Tableau comparatif des performances
- **Feature Importance** : Visualisation de l'importance des features

**Utilité** :
- Entraîner des modèles personnalisés sur vos données
- Comparer différents algorithmes
- Comprendre quels facteurs influencent le plus la durée

#### 🔮 **Sous-onglet : Predict New Bug Instance**

**Fonctionnalités** :
- **Formulaire de saisie** :
  - Catégorie du bug
  - Priorité
  - Sévérité
- **Prédictions** :
  - Temps de résolution estimé (en heures)
  - Indicateur de complexité/déviation du processus (score 0-100)
  - Niveau de risque
- **Recommandations** : Actions suggérées basées sur la prédiction

**Utilité** :
- Estimer à l'avance la durée de résolution d'un nouveau bug
- Prioriser les bugs selon leur complexité prédite
- Planifier les ressources

#### 📈 **Sous-onglet : Batch Predictions**

**Fonctionnalités** :
- **Prédictions en masse** : Analyser plusieurs bugs à la fois
- **Export des résultats** : Télécharger les prédictions en CSV/Excel
- **Classifications de risque** : Catégorisation automatique (faible/moyen/élevé)

**Utilité** :
- Analyser un lot de bugs en une seule fois
- Créer des rapports de prédiction
- Intégrer les prédictions dans d'autres systèmes

#### 🏆 **Sous-onglet : Category Prioritization**

**Fonctionnalités** :
- **Score de priorité par catégorie** (0-100) basé sur :
  - Risque de retard (40%)
  - Durée moyenne (30%)
  - Déviation du processus (20%)
  - Nombre d'instances (10%)
- **Prédictions** :
  - Temps de résolution estimé
  - Risque de retard prédit (%)
  - Score de déviation
- **Recommandations** :
  - "Handle First" : Priorité élevée
  - "Schedule Normally" : Priorité moyenne
  - "Can Defer" : Priorité faible
- **Tableau de classement** : Catégories triées par score de priorité
- **Visualisation** : Graphiques en barres du score de priorité

**Utilité** :
- Prioriser automatiquement les catégories de bugs
- Allouer les ressources efficacement
- Identifier les catégories à impact élevé

#### 📉 **Sous-onglet : Overall Process Performance**

**Fonctionnalités** :
- **KPIs globaux** :
  - Durée moyenne/mediane globale
  - Taux de violation SLA
  - Nombre moyen de réassignations
  - Taux de retravail (rework rate)
- **Impact par catégorie** :
  - Impact total en heures
  - Pourcentage d'impact
  - Taux de violation SLA par catégorie
- **Activités critiques** : Top 10 des activités les plus lentes

**Utilité** :
- Vue d'ensemble des performances globales
- Identifier les catégories ayant le plus d'impact
- Cibler les activités à optimiser en priorité

---

### 🎬 **Onglet 7 : Animation (Animation Token Replay)**

**Utilité** : Visualiser le flux de processus avec une animation de type "token replay"

**Fonctionnalités** :
- **Token Replay** : Animation des cas individuels le long du processus
- **Contrôles d'animation** :
  - Play/Pause
  - Vitesse de lecture
  - Sélection de cas spécifiques
- **Visualisation temporelle** : Comprendre la séquence d'événements

**Utilité** :
- Communiquer le processus de manière visuelle
- Déboguer les cas spécifiques
- Former les équipes sur le processus

---

## 📄 Description des Fichiers

### 🎯 **Fichier Principal**

#### `app.py`
**Rôle** : Application principale Streamlit qui orchestre toute l'interface utilisateur

**Responsabilités** :
- Configuration de la page Streamlit (titre, icône, layout)
- Gestion du sidebar (upload de fichiers, filtres)
- Affichage des KPIs principaux
- Gestion des onglets et de leur contenu
- Appels aux modules utilitaires pour les calculs et visualisations
- Gestion des états de session (filters, données, modèles ML)
- Intégration de tous les sous-systèmes (Process Mining, ML, Visualisations)

**Points clés** :
- Interface unique pour toutes les fonctionnalités
- Gestion des erreurs et affichage de messages utilisateur
- Synchronisation entre les différents modules

---

### 📦 **Modules Utilitaires (`utils/`)**

#### `utils/data_loader.py`
**Rôle** : Chargement, validation et transformation des données

**Fonctionnalités principales** :
- `load_and_validate_csv()` : Charge les fichiers CSV/Excel et valide les colonnes requises
- `standardize_column_names()` : Mappe les noms de colonnes alternatifs vers les noms standards
  - Ex: `case:concept:name` → `case_id`, `concept:name` → `activity`, etc.
- `apply_filters()` : Applique les filtres (catégorie, priorité, sévérité, dates)
- `get_filter_options()` : Extrait les options disponibles pour les filtres
- `convert_to_pm4py_log()` : Convertit le DataFrame pandas en format pm4py

**Utilité** :
- Interface unifiée pour le chargement de données
- Support de différents formats de colonnes
- Validation robuste des données d'entrée

---

#### `utils/process_mining.py`
**Rôle** : Calculs de Process Mining (DFG, variants, boucles)

**Fonctionnalités principales** :
- `compute_dfg_with_colors()` : Calcule le Directly-Follows Graph avec couleurs basées sur les durées
  - Retourne les arcs (transitions) avec fréquence et durée
  - Calcule les durées moyennes par nœud (activité)
  - Applique les couleurs (rouge/bleu) selon le seuil SLA
- `prepare_event_log()` : Prépare les données pour pm4py
- `get_process_variants()` : Identifie les variantes du processus (chemins différents)
- `analyze_loops()` : Détecte les boucles (retravail, réouverture)
- `detect_parallel_activities()` : Identifie les activités exécutées en parallèle

**Utilité** :
- Découverte de processus à partir des logs d'événements
- Analyse de la conformité et de la variabilité
- Support pour la visualisation du Process Map

---

#### `utils/metrics.py`
**Rôle** : Calcul des KPIs et statistiques diverses

**Fonctionnalités principales** :
- `calculate_kpis()` : Calcule les KPIs principaux
  - Temps de résolution moyen
  - Risque SLA (% et nombre)
  - Nombre de réouvertures
  - Taux de complétion
  - Bug le plus lent
- `calculate_case_durations()` : Calcule la durée de chaque cas (bug)
- `calculate_activity_durations()` : Calcule la durée de chaque activité
- `identify_bottlenecks()` : Identifie les goulots d'étranglement
- `calculate_heatmap_data()` : Prépare les données pour la heatmap
- `calculate_variant_analysis()` : Analyse statistique des variants
- `calculate_category_statistics()` : Statistiques par catégorie

**Utilité** :
- Métriques centralisées pour tout le dashboard
- Calculs optimisés et réutilisables
- Support pour les analyses statistiques

---

#### `utils/visualizations.py`
**Rôle** : Génération de tous les graphiques et visualisations

**Fonctionnalités principales** :
- `plot_process_map()` : Génère le Process Map avec Plotly
  - Nœuds rectangulaires avec durées
  - Arcs colorés avec épaisseur selon fréquence
  - Support pour l'animation avec tokens
- `plot_heatmap()` : Carte de chaleur activité × catégorie
- `plot_duration_distribution()` : Histogrammes de distribution
- `plot_timeline()` : Chronologie des événements
- `plot_category_comparison()` : Comparaisons par catégorie
- `plot_activity_frequency()` : Fréquence des activités
- `plot_temporal_analysis()` : Analyses temporelles
- `plot_priority_distribution()` : Distribution des priorités
- `plot_variant_analysis()` : Visualisation des variants

**Utilité** :
- Génération centralisée de toutes les visualisations
- Interface cohérente avec Plotly
- Support pour l'interactivité et les animations

---

#### `utils/feature_engineering.py`
**Rôle** : Extraction et préparation des features pour le Machine Learning

**Fonctionnalités principales** :
- `extract_features_from_log()` : Extrait les features historiques du log
  - Nombre de bugs similaires
  - Durée moyenne des corrections précédentes
  - Statistiques par catégorie/priorité/sévérité
- `prepare_features_for_prediction()` : Prépare les features pour une prédiction
  - Combine les inputs utilisateur avec les données historiques
  - Crée un DataFrame prêt pour le modèle ML
- `encode_categorical_features()` : Encode les variables catégorielles (LabelEncoder)
- `calculate_complexity_score()` : Calcule un score de complexité
- `calculate_process_deviation()` : Calcule la déviation du processus
  - Score de déviation (0-100)
  - Facteurs de déviation (nombre d'activités, durée, retravail)

**Utilité** :
- Préparation des données pour l'entraînement ML
- Feature engineering avancé
- Calcul de métriques de complexité

---

#### `utils/ml_models.py`
**Rôle** : Modèles Machine Learning pour la prédiction de durée

**Fonctionnalités principales** :
- `BugDurationPredictor` : Classe principale pour la prédiction
  - Support pour Random Forest, Gradient Boosting, Linear Regression
  - Entraînement avec validation croisée
  - Calcul de feature importance
  - Sauvegarde/chargement de modèles
- `train_model_cached()` : Entraîne un modèle avec cache
- `compare_models()` : Compare plusieurs modèles et retourne les métriques

**Utilité** :
- Prédiction de la durée de résolution des bugs
- Comparaison de différents algorithmes ML
- Réutilisation de modèles entraînés

---

#### `utils/category_prioritization.py`
**Rôle** : Priorisation intelligente des catégories de bugs

**Fonctionnalités principales** :
- `prioritize_categories()` : Calcule un score de priorité pour chaque catégorie
  - Utilise le modèle ML si disponible pour les prédictions
  - Combine plusieurs facteurs (risque, durée, déviation, instances)
  - Retourne un tableau de classement
- `analyze_overall_process_performance()` : Analyse globale des performances
  - KPIs globaux
  - Impact par catégorie
  - Activités critiques

**Utilité** :
- Priorisation automatique des catégories
- Analyse d'impact globale
- Recommandations d'action

---

#### `utils/animation.py`
**Rôle** : Animation token replay basique

**Fonctionnalités** :
- Animation des cas individuels le long du processus
- Contrôles de lecture

---

#### `utils/advanced_animation.py`
**Rôle** : Animation avancée style Fluxicon Disco

**Fonctionnalités** :
- Tokens animés sur les arcs du Process Map
- Vitesse variable selon les performances
- Flux continu avec plusieurs tokens par arc

---

### 🧪 **Tests (`tests/`)**

#### `tests/test_data_loader.py`
**Rôle** : Tests unitaires pour le module `data_loader`

**Utilité** :
- Validation du chargement de données
- Tests des fonctions de mapping de colonnes
- Assurance qualité du code

---

### 📚 **Documentation (`docs/`)**

Les fichiers `.md` dans `docs/` contiennent la documentation détaillée :
- **INSTALLATION.md** : Guide d'installation
- **USAGE_GUIDE.md** : Guide d'utilisation
- **AI_FEATURES.md** : Documentation des fonctionnalités IA
- **ANIMATION_GUIDE.md** : Guide des animations
- Et d'autres guides spécialisés...

---

### 🐳 **Docker**

#### `Dockerfile`
**Rôle** : Configuration pour créer une image Docker de l'application

#### `docker-compose.yml`
**Rôle** : Orchestration Docker pour déployer l'application avec toutes ses dépendances

#### `docker-run.bat`
**Rôle** : Script Windows pour lancer l'application via Docker

---

### ⚙️ **Configuration**

#### `requirements.txt`
**Rôle** : Liste de toutes les dépendances Python avec versions minimales

#### `setup.py`
**Rôle** : Configuration du package Python (si nécessaire)

#### `run.bat`
**Rôle** : Script Windows pour lancer l'application localement

---

## 🚀 Installation

### Prérequis
- Python 3.8 ou supérieur
- pip (gestionnaire de paquets Python)

### Installation des Dépendances

```bash
pip install -r requirements.txt
```

### Lancement de l'Application

**Windows :**
```bash
run.bat
```

**Linux/Mac :**
```bash
streamlit run app.py
```

### Installation via Docker

```bash
docker-compose up
```

Ou avec le script :
```bash
docker-run.bat
```

---

## 📖 Utilisation

### 1. **Charger les Données**

1. Cliquez sur "Upload Event Log (CSV or Excel)" dans la sidebar
2. Sélectionnez votre fichier CSV ou Excel
3. Le fichier doit contenir les colonnes suivantes :
   - `case_id` (ou `case:concept:name`) : Identifiant unique du bug
   - `activity` (ou `concept:name`) : Nom de l'activité
   - `timestamp` (ou `time:timestamp`) : Date et heure de l'événement
   - `category` (optionnel) : Catégorie du bug
   - `priority` (optionnel) : Priorité du bug
   - `severity` (optionnel) : Sévérité du bug

### 2. **Configurer les Filtres**

Dans la sidebar, vous pouvez filtrer par :
- **Catégorie** : Sélectionnez une ou plusieurs catégories
- **Priorité** : Filtrer par niveau de priorité
- **Sévérité** : Filtrer par niveau de sévérité
- **Plage de dates** : Sélectionner une période spécifique
- **Seuil SLA** : Définir le seuil en heures (par défaut 24h)

### 3. **Explorer les Visualisations**

Naviguez entre les onglets pour :
- **Process Map** : Voir le flux de processus avec animations
- **Heatmap** : Identifier les goulots d'étranglement
- **Distributions** : Analyser les statistiques
- **Variants & Loops** : Découvrir les variantes du processus
- **Temporal Analysis** : Analyser les tendances temporelles
- **AI Predictions** : Utiliser le ML pour prédire et prioriser
- **Animation** : Visualiser le flux avec token replay

### 4. **Utiliser les Prédictions IA**

1. Allez dans l'onglet **AI Predictions**
2. Dans **Model Training & Evaluation**, entraînez un modèle
3. Utilisez **Predict New Bug Instance** pour prédire un nouveau bug
4. Consultez **Category Prioritization** pour prioriser les catégories

---

## 🎯 Cas d'Usage

- **Équipes de Développement** : Identifier les goulots d'étranglement dans le processus de résolution de bugs
- **Chefs de Projet** : Prévoir les temps de résolution et prioriser les ressources
- **QA Managers** : Analyser les patterns de bugs et améliorer les processus
- **Data Analysts** : Explorer les données de processus avec des visualisations interactives

---

## 📝 Notes

- Les données sont traitées localement (pas d'envoi vers des serveurs externes)
- Les modèles ML peuvent être sauvegardés dans le dossier `models/`
- Les exports peuvent être sauvegardés dans le dossier `exports/`

---

## 🤝 Contribution

Ce projet est en développement continu. Pour contribuer :
1. Fork le projet
2. Créez une branche pour votre fonctionnalité
3. Commitez vos changements
4. Poussez vers la branche
5. Ouvrez une Pull Request

---

## 📄 Licence

Ce projet est disponible sous licence MIT (ou autre selon votre choix).

---

**Développé avec ❤️ pour l'optimisation des processus de développement logiciel**

