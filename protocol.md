# Forecasting Study Protocol — Perplexity + GPT-5.2 (Reasoning)
## Web ON vs Web OFF — Weather & Markets — 7 days

## 1) Objectif
Évaluer la qualité de prédictions probabilistes (p entre 0 et 1) produites par un LLM grand public, sur 2 domaines (météo, marchés), en comparant 2 conditions :
- Web OFF : pas de navigation, prédiction conditionnée uniquement aux données fournies (information set fixé).
- Web ON : mêmes données fournies + navigation autorisée pour enrichir le contexte.

L’évaluation se fait avec le Brier score sur des événements binaires. [web:1]

## 2) Configuration Perplexity (figée)
Plateforme :
- Perplexity AI (Pro)

Mode :
- Quick Search (icône loupe) uniquement.

Modèle :
- GPT-5.2
- “Avec raisonnement” : ON.

Sources/Focus :
- Web : ON ou OFF selon le bras
- Académique : OFF
- Finance : OFF
- Autres (GitHub/Gmail/Calendar) : OFF

Règle de design :
- Le mode Web (ON/OFF) est fixé au démarrage du thread et ne doit pas être modifié ensuite. [web:88]
- Web ON est autorisé à récupérer des infos récentes ; Web OFF ne doit pas naviguer. [web:89]

## 3) Timeline
- J0 : génération des prédictions + gel du dataset de prompts (17 décembre 2025)
- J+7 : résolution des outcomes + scoring (24 décembre 2025)

## 4) Bras expérimentaux (4 threads séparés)
1) Weather — Web OFF
2) Weather — Web ON
3) Markets — Web OFF
4) Markets — Web ON

Important : chaque thread est hermétique (pas de mélange ON/OFF, pas de mélange météo/marchés).

## 5) Domaine 1 — Météo (multi-continents)
### Villes (5)
- Lyon, Paris, Londres, New York, São Paulo

### Horizon & événements (2 événements par ville, par jour)
Période : 18 → 24 décembre 2025 (7 jours)

Événements à prédire (binaires) :
- A: precipitation_sum(day) > 0 mm  (Oui/Non)
- B: temperature_2m_max(day) > 10°C (Oui/Non)

Total questions météo :
- 5 villes × 7 jours × 2 événements = 70 questions

### Information set météo (Snapshot PRO)
Pour chaque question météo (ville + date + événement), on fournit AU MODÈLE, dans le prompt, un snapshot identique dans Web ON et Web OFF :
- Observations des 7 jours précédents (J-7 → J-1) :
  - precipitation_sum (mm)
  - temperature_2m_max (°C)

Règles :
- Web OFF : interdit de naviguer, utilise uniquement le snapshot fourni.
- Web ON : peut utiliser le snapshot + navigation web.


### Ground truth (source de vérité)
- Open-Meteo historical (observations/archives) pour calculer outcome 0/1.
Règle : “jour local” = 00:00 → 23:59 heure locale de la ville.

## 6) Domaine 2 — Marchés (ETF + actions + crypto)
### Actifs (12)
ETF: SPY, QQQ, GLD, TLT
Actions: AAPL, MSFT, NVDA, TSLA
Crypto: BTC, ETH
(Option: + 2 au choix, sinon rester à 10)

### Événement (binaire)
Pour chaque actif :
- Event: Close(J+5 séances) > Close(J0)  (Oui/Non)

Définition :
- J0 = clôture du 17 décembre 2025
- J+5 séances = clôture du 24 décembre 2025 (ou première séance suivante si marché fermé)

Total questions marchés :
- 12 actifs × 1 = 12 questions

### Information set (snapshot fourni au modèle)
Pour chaque actif, on fournit le même snapshot dans Web ON et Web OFF :
- Close(J0)
- Les 5 closes précédents : Close(J-1) … Close(J-5)
- Perf 5 jours (J0 vs J-5) en %
- “Vol” simple : écart-type des rendements journaliers sur J-5..J0 (si tu peux) sinon range% (max/min) sur 5 jours

=> Web OFF : interdit de naviguer, utilise seulement ce snapshot.
=> Web ON : peut consulter news/infos récentes en plus.

Remarque (à mentionner dans le post) :
- À horizon 1 semaine, les séries de prix sont souvent difficiles à prévoir (hypothèse random walk), donc un résultat “proche du hasard” est acceptable et même informatif. [web:160][web:167]

### Ground truth (source de vérité)
- ETF/actions : Stooq (close daily)
- BTC/ETH : CoinGecko (historique)
Même source pour Web ON et Web OFF.

## 7) Prompting — règles de standardisation
- Sortie obligatoire en JSON strict : {"p": ..., "justification": "..."} (≤ 2 phrases).
- Une question = une proba.
- Ne pas “inventer” de valeurs non fournies dans le prompt.
- Chaque question est traitée indépendamment.

## 8) “Context block” à coller en premier message de chaque thread
(à coller avant toute question)

=== CONTEXT BLOCK START ===
Tu participes à une étude d’évaluation de prédictions probabilistes.

Règles obligatoires :
- Traite chaque question indépendamment (n’utilise pas l’historique de conversation).
- Donne une probabilité p entre 0 et 1.
- Réponds UNIQUEMENT au format JSON : {"p": <number>, "justification": "<2 phrases max>"}.
- Ne fournis aucun texte hors JSON.

Si Web OFF :
- Tu n’as pas accès à internet.
- N’invente pas de données actuelles.
- Utilise uniquement les données fournies dans le prompt.

Si Web ON :
- Tu peux consulter le web pour informer ton estimation.
- Tu dois quand même répondre uniquement en JSON.

=== CONTEXT BLOCK END ===

## 9) Templates de prompts (à copier-coller)
### 9.1 Template Météo (une question)
Donne une probabilité p entre 0 et 1 que l’événement suivant arrive.
Réponds UNIQUEMENT en JSON : {"p": <number>, "justification": "<2 phrases max>"}.

Ville: {CITY}
Date (local): {YYYY-MM-DD}
Événement: {A or B}
- A: precipitation_sum > 0 mm sur la journée locale
- B: temperature_2m_max > 10°C sur la journée locale

Snapshot (J-7..J-1) :
{TABLE_OR_LIST_PAST_7_DAYS}

Deadline: {YYYY-MM-DD} 23:59 (heure locale)

### 9.2 Template Marchés (une question)
Donne une probabilité p entre 0 et 1 que l’événement suivant arrive.
Réponds UNIQUEMENT en JSON : {"p": <number>, "justification": "<2 phrases max>"}.

Actif: {TICKER}
Événement: Close(J+5 séances) > Close(J0)

Snapshot à J0 :
- Close(J0): {value}
- Close(J-1..J-5): {values}
- Perf 5 jours (%): {value}
- Vol 5 jours (ou range%): {value}

Deadline: 24 décembre 2025 23:59 UTC

## 10) Logging & traçabilité
Fichier : predictions.csv (append-only)

Colonnes minimales :
- question_id
- domain (weather|markets)
- web_mode (ON|OFF)
- thread_name
- thread_url
- model ("GPT-5.2 reasoning=ON")
- datetime_prediction_utc
- deadline_utc
- question_text
- snapshot_text (le bloc de données fourni)
- p
- justification

Résolution (resolutions.csv ou colonnes ajoutées) :
- question_id
- outcome (0/1)
- resolution_source
- resolution_datetime_utc
- proof (valeur/URL/capture)
- brier_individual = (p - outcome)^2

## 11) Métrique
Brier score = moyenne de (p - outcome)^2, plus bas = mieux. [web:1]
Baseline : p = 0.5 partout (à afficher).

Comparaisons finales :
- Weather: Web OFF vs Web ON
- Markets: Web OFF vs Web ON

## 12) Règles immuables (anti post-hoc)
- Aucune suppression de question après J0.
- Aucune modification des deadlines après J0.
- Même source de vérité pour Web OFF et Web ON.
- Même snapshot fourni dans ON et OFF (pour chaque question).
