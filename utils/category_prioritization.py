"""
AI-Based Bug Category Prioritization Module
"""

import pandas as pd
import numpy as np
from utils.feature_engineering import extract_features_from_log, calculate_process_deviation
from utils.ml_models import BugDurationPredictor


def get_category_recommendations(category_name, metrics):
    """
    Get detailed recommendations for a specific bug category.
    
    Args:
        category_name: Name of the bug category
        metrics: Dictionary containing category metrics
        
    Returns:
        dict: Recommendations with insights and actions
    """
    # Normalize category name for matching
    cat_lower = category_name.lower()
    
    # Define category-specific recommendations
    recommendations = {
        'insights': [],
        'actions': [],
        'kpis': [],
        'priority_level': 'Medium'
    }
    
    # Functional Bugs
    if any(keyword in cat_lower for keyword in ['functional', 'function', 'fonctionnel', 'feature']):
        recommendations['insights'] = [
            "Très fréquents et souvent critiques pour la fonctionnalité principale du logiciel",
            "Souvent associés à de longs délais entre 'Assign' → 'Fix'",
            f"Taux de dépassement SLA: {metrics.get('predicted_delay_risk', 0):.1f}%"
        ]
        recommendations['actions'] = [
            "🧠 Mettre en place une détection automatique de régressions (tests unitaires automatisés)",
            "🚀 Prioriser les cas similaires déjà résolus (similarity-based retrieval)",
            "🕒 Définir un SLA strict (ex. < 48h) pour éviter accumulation",
            "📊 Surveiller les transitions 'Fix → QA' pour éviter retards de validation"
        ]
        recommendations['kpis'] = [
            "Temps moyen Assign → Fix",
            "Taux de régression",
            "Couverture de tests unitaires"
        ]
        recommendations['priority_level'] = 'High' if metrics.get('predicted_delay_risk', 0) > 50 else 'Medium'
    
    # Performance Bugs
    elif any(keyword in cat_lower for keyword in ['performance', 'perf', 'slow', 'speed', 'latency', 'memory']):
        recommendations['insights'] = [
            "Moins fréquents mais temps de résolution très élevé",
            "Corrélation forte avec retards 'Assign → Fix' et 'Fix → QA'",
            f"Durée moyenne: {metrics.get('avg_duration', 0):.1f}h (vs moyenne globale)"
        ]
        recommendations['actions'] = [
            "🔍 Surveiller les métriques de performance (CPU, mémoire, latence)",
            "🧩 Appliquer du profiling automatique dès détection du bug",
            "⚡ Prioriser selon impact (critical path ou composant clé)",
            "🧠 Utiliser un modèle prédictif pour estimer le délai probable",
            "📈 Mettre en place des benchmarks automatiques"
        ]
        recommendations['kpis'] = [
            "Temps de résolution moyen",
            "Impact sur performance système",
            "Nombre de sessions affectées"
        ]
        recommendations['priority_level'] = 'Critical' if metrics.get('avg_duration', 0) > 48 else 'High'
    
    # Testing/QA Bugs
    elif any(keyword in cat_lower for keyword in ['test', 'qa', 'quality', 'validation']):
        recommendations['insights'] = [
            "Fréquence moyenne, mais réouvertures fréquentes ('Reopen rate' élevé)",
            "Souvent dus à des cas de test incomplets ou ambigus",
            f"Score de déviation: {metrics.get('deviation_score', 0):.1f}/100"
        ]
        recommendations['actions'] = [
            "🧱 Renforcer la couverture des tests automatisés",
            "🔁 Mettre en place un contrôle qualité sur la rédaction des cas de test",
            "🕵️‍♀️ Vérifier les incohérences entre versions testées et corrigées",
            "📋 Documenter les scénarios de test manquants",
            "🤖 Automatiser les tests de régression"
        ]
        recommendations['kpis'] = [
            "Taux de réouverture",
            "Couverture de tests",
            "Temps moyen en QA"
        ]
        recommendations['priority_level'] = 'Medium'
    
    # GUI/UI Bugs
    elif any(keyword in cat_lower for keyword in ['gui', 'ui', 'interface', 'visual', 'frontend', 'display']):
        recommendations['insights'] = [
            "Impact modéré mais haute fréquence en phase de test",
            "Souvent rejetés plusieurs fois pour retouches mineures ('QA repetition count' élevé)",
            f"Nombre d'instances: {metrics.get('instance_count', 0)}"
        ]
        recommendations['actions'] = [
            "🎯 Grouper les bugs visuels par composant (bouton, formulaire, menu)",
            "👀 Utiliser des outils d'automated UI testing (ex. Selenium, Cypress)",
            "📅 Planifier des sprints UI courts dédiés",
            "💬 Encourager les validations croisées entre devs et testeurs",
            "📸 Automatiser les tests visuels (screenshot comparison)"
        ]
        recommendations['kpis'] = [
            "Nombre de rejets QA",
            "Temps moyen de correction",
            "Taux de satisfaction utilisateur"
        ]
        recommendations['priority_level'] = 'Medium'
    
    # Backend Bugs
    elif any(keyword in cat_lower for keyword in ['backend', 'api', 'server', 'database', 'db']):
        recommendations['insights'] = [
            "Impact critique sur la stabilité du système",
            "Nécessite souvent une expertise technique avancée",
            f"Temps de résolution prévu: {metrics.get('predicted_resolution_time', 0):.1f}h"
        ]
        recommendations['actions'] = [
            "🔧 Assigner à des développeurs backend expérimentés",
            "📊 Mettre en place un monitoring proactif des APIs",
            "🗄️ Vérifier l'intégrité et performance de la base de données",
            "🔍 Utiliser des outils de debugging avancés (profilers, logs structurés)",
            "⚡ Mettre en cache les requêtes fréquentes si applicable"
        ]
        recommendations['kpis'] = [
            "Temps de réponse API",
            "Taux d'erreurs serveur",
            "Disponibilité du service"
        ]
        recommendations['priority_level'] = 'High'
    
    # Security Bugs
    elif any(keyword in cat_lower for keyword in ['security', 'sécurité', 'vulnerability', 'auth', 'authentication']):
        recommendations['insights'] = [
            "⚠️ PRIORITÉ MAXIMALE - Impact sur la sécurité du système",
            "Nécessite une résolution immédiate et des tests approfondis",
            "Peut nécessiter un patch urgent en production"
        ]
        recommendations['actions'] = [
            "🚨 Traiter immédiatement - bloquer tout autre travail si critique",
            "🔒 Audit de sécurité complet du composant affecté",
            "🧪 Tests de pénétration avant déploiement",
            "📢 Communication transparente avec les stakeholders",
            "🔐 Révision du code par un expert sécurité",
            "📝 Documentation complète de la faille et de la correction"
        ]
        recommendations['kpis'] = [
            "Temps de réponse à l'incident",
            "Score de sévérité CVE",
            "Impact utilisateurs"
        ]
        recommendations['priority_level'] = 'Critical'
    
    # Integration Bugs
    elif any(keyword in cat_lower for keyword in ['integration', 'intégration', 'connectivity', 'connection']):
        recommendations['insights'] = [
            "Affecte la communication entre systèmes/composants",
            "Peut causer des effets en cascade sur d'autres services",
            f"Déviation du processus standard: {metrics.get('deviation_score', 0):.1f}/100"
        ]
        recommendations['actions'] = [
            "🔗 Vérifier tous les endpoints et contrats d'API",
            "🧪 Tester les scénarios de failover et retry",
            "📊 Mettre en place un monitoring des intégrations",
            "🔄 Documenter les dépendances inter-systèmes",
            "⚡ Implémenter des circuit breakers si applicable"
        ]
        recommendations['kpis'] = [
            "Taux de succès des intégrations",
            "Temps de détection des pannes",
            "MTTR (Mean Time To Recovery)"
        ]
        recommendations['priority_level'] = 'High'
    
    # Default recommendations for unclassified categories
    else:
        recommendations['insights'] = [
            f"Catégorie: {category_name}",
            f"Durée moyenne de résolution: {metrics.get('avg_duration', 0):.1f}h",
            f"Risque de retard: {metrics.get('predicted_delay_risk', 0):.1f}%",
            f"Nombre d'instances: {metrics.get('instance_count', 0)}"
        ]
        recommendations['actions'] = [
            "📊 Analyser les patterns de résolution historiques",
            "🎯 Identifier les goulots d'étranglement spécifiques",
            "👥 Assigner selon l'expertise requise",
            "📈 Suivre l'évolution des métriques clés",
            "🔄 Mettre en place un processus de revue régulier"
        ]
        recommendations['kpis'] = [
            "Temps de résolution",
            "Taux de complétion",
            "Satisfaction client"
        ]
        
        # Determine priority based on metrics
        if metrics.get('priority_score', 0) >= 70:
            recommendations['priority_level'] = 'Critical'
        elif metrics.get('priority_score', 0) >= 40:
            recommendations['priority_level'] = 'High'
        else:
            recommendations['priority_level'] = 'Medium'
    
    return recommendations


def prioritize_categories(df, sla_threshold=24.0, predictor=None):
    """
    Prioritize bug categories based on AI analysis.
    
    Args:
        df: Event log dataframe
        sla_threshold: SLA threshold in hours
        predictor: Trained BugDurationPredictor (optional)
        
    Returns:
        pd.DataFrame: Category prioritization ranking with scores
    """
    if df.empty or 'category' not in df.columns:
        return pd.DataFrame()
    
    # Get unique categories
    categories = df['category'].dropna().unique()
    
    if len(categories) == 0:
        return pd.DataFrame()
    
    # Calculate statistics per category
    case_durations = df.groupby('case_id').agg({
        'timestamp': ['min', 'max'],
        'category': 'first'
    }).reset_index()
    case_durations.columns = ['case_id', 'start_time', 'end_time', 'category']
    case_durations['duration_hours'] = (
        case_durations['end_time'] - case_durations['start_time']
    ).dt.total_seconds() / 3600
    
    results = []
    
    for category in categories:
        cat_data = df[df['category'] == category]
        cat_cases = case_durations[case_durations['category'] == category]
        
        if cat_cases.empty:
            continue
        
        # Basic statistics
        instance_count = cat_cases['case_id'].nunique()
        avg_duration = cat_cases['duration_hours'].mean()
        median_duration = cat_cases['duration_hours'].median()
        std_duration = cat_cases['duration_hours'].std()
        
        # Priority and severity distribution
        priority_dist = {}
        severity_dist = {}
        if 'priority' in df.columns:
            priority_dist = cat_data.groupby('case_id')['priority'].first().value_counts().to_dict()
        if 'severity' in df.columns:
            severity_dist = cat_data.groupby('case_id')['severity'].first().value_counts().to_dict()
        
        # Calculate predicted resolution time if predictor available
        predicted_resolution_time = avg_duration
        if predictor is not None and predictor.is_trained:
            try:
                from utils.feature_engineering import prepare_features_for_prediction, extract_features_from_log
                hist_data = extract_features_from_log(df)
                
                # Use most common priority/severity for this category
                most_common_priority = cat_data.groupby('case_id')['priority'].first().mode()[0] if 'priority' in df.columns else None
                most_common_severity = cat_data.groupby('case_id')['severity'].first().mode()[0] if 'severity' in df.columns else None
                
                if most_common_priority and most_common_severity:
                    features = prepare_features_for_prediction(
                        category=category,
                        priority=most_common_priority,
                        severity=most_common_severity,
                        historical_data=hist_data
                    )
                    predicted_resolution_time = predictor.predict(features)[0]
            except:
                predicted_resolution_time = avg_duration
        
        # Calculate delay risk (probability of exceeding SLA)
        delay_risk = 0
        if sla_threshold > 0:
            cases_exceeding_sla = len(cat_cases[cat_cases['duration_hours'] > sla_threshold])
            delay_risk = (cases_exceeding_sla / len(cat_cases)) * 100 if len(cat_cases) > 0 else 0
        
        # Calculate process deviation
        deviation = calculate_process_deviation(df, category=category)
        deviation_score = deviation.get('deviation_score', 0)
        
        # Calculate priority score (0-100)
        # Factors: delay risk (40%), duration (30%), deviation (20%), instance count (10%)
        priority_score = (
            min(40, delay_risk) * 0.4 +
            min(30, (avg_duration / sla_threshold * 30) if sla_threshold > 0 else 0) +
            min(20, deviation_score * 0.2) +
            min(10, (instance_count / max(df['case_id'].nunique(), 1) * 100) * 0.1)
        )
        
        # Determine suggested action
        if priority_score >= 70:
            suggested_action = "Handle First"
        elif priority_score >= 40:
            suggested_action = "Schedule Normally"
        else:
            suggested_action = "Can Defer"
        
        results.append({
            'category': category,
            'priority_score': priority_score,
            'predicted_resolution_time': predicted_resolution_time,
            'predicted_delay_risk': delay_risk,
            'avg_duration': avg_duration,
            'instance_count': instance_count,
            'deviation_score': deviation_score,
            'suggested_action': suggested_action,
            'median_duration': median_duration,
            'std_duration': std_duration
        })
    
    # Create DataFrame and sort by priority score
    result_df = pd.DataFrame(results)
    if not result_df.empty:
        result_df = result_df.sort_values('priority_score', ascending=False)
        result_df = result_df.reset_index(drop=True)
    
    return result_df


def analyze_overall_process_performance(df, sla_threshold=24.0):
    """
    Analyze overall process performance by category.
    
    Args:
        df: Event log dataframe
        sla_threshold: SLA threshold in hours
        
    Returns:
        dict: Overall performance metrics and category impacts
    """
    if df.empty:
        return {}
    
    # Calculate overall KPIs
    case_durations = df.groupby('case_id').agg({
        'timestamp': ['min', 'max']
    }).reset_index()
    case_durations.columns = ['case_id', 'start_time', 'end_time']
    case_durations['duration_hours'] = (
        case_durations['end_time'] - case_durations['start_time']
    ).dt.total_seconds() / 3600
    
    overall_avg_duration = case_durations['duration_hours'].mean()
    overall_median_duration = case_durations['duration_hours'].median()
    cases_exceeding_sla = len(case_durations[case_durations['duration_hours'] > sla_threshold])
    sla_breach_rate = (cases_exceeding_sla / len(case_durations)) * 100 if len(case_durations) > 0 else 0
    
    # Count reassignments (activity changes that suggest reassignment)
    reassignments = df.groupby('case_id')['activity'].nunique()
    avg_reassignments = reassignments.mean()
    max_reassignments = reassignments.max()
    
    # Check for loops (rework)
    reopened_cases = df[df['activity'].str.lower().str.contains('reopen', na=False)]['case_id'].nunique()
    rework_rate = (reopened_cases / df['case_id'].nunique()) * 100 if df['case_id'].nunique() > 0 else 0
    
    # Category impact analysis
    category_impacts = []
    if 'category' in df.columns:
        categories = df['category'].dropna().unique()
        case_durations_with_cat = case_durations.merge(
            df.groupby('case_id')['category'].first().reset_index(),
            on='case_id',
            how='left'
        )
        
        for category in categories:
            cat_cases = case_durations_with_cat[case_durations_with_cat['category'] == category]
            if cat_cases.empty:
                continue
            
            cat_avg = cat_cases['duration_hours'].mean()
            cat_count = len(cat_cases)
            cat_impact = cat_count * cat_avg  # Total hours consumed
            
            category_impacts.append({
                'category': category,
                'instance_count': cat_count,
                'avg_duration': cat_avg,
                'total_impact_hours': cat_impact,
                'impact_percentage': (cat_impact / (overall_avg_duration * len(case_durations))) * 100 if len(case_durations) > 0 else 0,
                'sla_breach_count': len(cat_cases[cat_cases['duration_hours'] > sla_threshold]),
                'sla_breach_rate': (len(cat_cases[cat_cases['duration_hours'] > sla_threshold]) / len(cat_cases)) * 100 if len(cat_cases) > 0 else 0
            })
        
        category_impacts_df = pd.DataFrame(category_impacts)
        if not category_impacts_df.empty:
            category_impacts_df = category_impacts_df.sort_values('total_impact_hours', ascending=False)
    
    # Most critical activities (slowest activities)
    activity_durations = df.groupby(['case_id', 'activity']).agg({
        'timestamp': ['min', 'max']
    }).reset_index()
    activity_durations.columns = ['case_id', 'activity', 'start_time', 'end_time']
    activity_durations['duration_hours'] = (
        activity_durations['end_time'] - activity_durations['start_time']
    ).dt.total_seconds() / 3600
    
    critical_activities = activity_durations.groupby('activity').agg({
        'duration_hours': ['mean', 'count']
    }).reset_index()
    critical_activities.columns = ['activity', 'avg_duration', 'frequency']
    critical_activities = critical_activities.sort_values('avg_duration', ascending=False).head(10)
    
    return {
        'overall_avg_duration': overall_avg_duration,
        'overall_median_duration': overall_median_duration,
        'sla_breach_rate': sla_breach_rate,
        'sla_breach_count': cases_exceeding_sla,
        'avg_reassignments': avg_reassignments,
        'max_reassignments': max_reassignments,
        'rework_rate': rework_rate,
        'total_cases': len(case_durations),
        'category_impacts': category_impacts_df if 'category_impacts_df' in locals() else pd.DataFrame(),
        'critical_activities': critical_activities
    }

