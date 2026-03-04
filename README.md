# 🔬 ForecastingLLM — AI Forecasting Laboratory

> Évaluation probabiliste multi-domaine d'un LLM (GPT-5.2) : **Météo · Marchés Financiers · Performance Sportive**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://forecastingllm.streamlit.app)

---

## 🎯 Objectif

Ce laboratoire compare les capacités de **prédiction probabiliste** d'un LLM avancé (GPT-5.2 avec reasoning) dans 3 domaines distincts, selon deux conditions expérimentales :

- **Web OFF** : prédiction basée uniquement sur les données fournies dans le prompt (pas d'accès internet)
- **Web ON** : mêmes données + accès aux connaissances externes (navigation web)

L'évaluation s'appuie sur des métriques probabilistes rigoureuses (**Brier Score**, **MAE**, **Coverage P10-P90**) comparées à des baselines spécialisées par domaine.

---

## 📊 Études Réalisées

| Domaine | Prédictions | Baseline | Métrique principale |
|---|---|---|---|
| 🌦️ Météo | 35 prédictions (5 villes × 7 jours) | Open-Meteo API | MAE (°C) + Brier Score |
| 📈 Marchés | 10 actifs européens (horizon J+5) | Stratégie aléatoire (p=0.5) | Brier Score |
| 🏃 Running | 7 méthodes, 1 marathon (Run In Lyon 2025) | Riegel, Coros, HR→Pace | MAE (minutes) |

---

## 🏆 Résultats Clés (V1 — Décembre 2025)

- ✅ **Météo** : L'IA améliore la précision de ~15-20% vs API professionnelle (MAE ~2°C)
- ✅ **Marchés** : Brier Score ~0.22 (vs 0.25 pour le hasard pur) — compétence prédictive réelle
- ✅ **Running** : Performance comparable aux formules physiologiques établies (Riegel, Coros)
- ⚠️ **Biais identifié** : Web ON tend à rendre les prédictions plus optimistes (+10-20% en finance)

---

## 🗂️ Structure du Projet

```
ForecastingLLM/
├── forecasting_llm_lab.py     # Streamlit app (entrypoint)
├── requirements.txt           # Dépendances Python
├── protocol.md                # Protocole expérimental détaillé
│
├── WEATHER/
│   └── weather_experiment_FINAL.xlsx   # Données météo (35 lignes)
│
├── MARKETS/
│   └── markets_experiments_FINAL.xlsx  # Données marchés (10 actifs)
│
└── RUNNING/
    ├── marathon_predictions_oneshot.csv # Prédictions toutes méthodes
    └── CSVs_transitions/
        ├── marathon_snapshot_16w.csv    # Résumé 16 semaines
        ├── marathon_weekly_16w.csv      # Agrégats hebdomadaires
        └── marathon_runs_16w.csv        # Détail des séances
```

---

## 🚀 Lancer en local

**Prérequis** : Python 3.10+

```bash
# 1. Cloner le repo
git clone https://github.com/BadreddineEK/ForecastingLLM.git
cd ForecastingLLM

# 2. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows

# 3. Installer les dépendances
pip install -r requirements.txt

# 4. Lancer l'app
streamlit run forecasting_llm_lab.py
```

> ⚠️ Lancer **depuis la racine du repo** pour que les chemins relatifs vers les dossiers `WEATHER/`, `MARKETS/`, `RUNNING/` soient corrects.

---

## ☁️ Déployer sur Streamlit Community Cloud

1. Aller sur [share.streamlit.io](https://share.streamlit.io)
2. Sélectionner le repo : `BadreddineEK/ForecastingLLM`
3. Branche : `main`
4. Main file path : `forecasting_llm_lab.py`
5. Cliquer **Deploy**

Aucun secret ou variable d'environnement requis pour la V1 — les données sont bundlées dans le repo.

---

## 📐 Protocole Expérimental

Le protocole complet (design expérimental, templates de prompts, règles anti-leak, métriques) est disponible dans [`protocol.md`](./protocol.md).

**Règles clés :**
- J0 (prédictions générées) : **17 décembre 2025**
- J+7 (résolution des outcomes) : **24 décembre 2025**
- Anti-leak strict : aucune donnée post-J0 utilisée pour les prédictions
- 4 threads expérimentaux hermétiques (Weather OFF/ON, Markets OFF/ON)

---

## 🛠️ Stack Technique

- **App** : [Streamlit](https://streamlit.io)
- **Data** : Pandas, NumPy, OpenPyXL
- **Viz** : Plotly Express / Graph Objects
- **Stats** : SciPy (t-tests, bootstrap CI)
- **LLM évalué** : GPT-5.2 (reasoning ON) via Perplexity Pro
- **Baselines météo** : Open-Meteo API
- **Baselines running** : Formule de Riegel, Coros, HR→Pace regression

---

## 👤 Auteur

**EL KHAMLICHI Badreddine** — Data Scientist  
Ingénieur Mathématiques Appliquées · Polytech Lyon (2024)  
[GitHub](https://github.com/BadreddineEK)

---

*V1 — Mars 2026*
