---
name: kaggle-feature-ideation
description: Generate predictive features based on the Formula 1 (F1) and Pit Stops domain. Use when engineering new variables, brainstorming columns, or expanding domain-specific indicators for F1 timing datasets.
---

# F1 Pit Stops Feature Ideation

Use this skill to brainstorm and engineer highly predictive, domain-specific features for Formula 1 and pit stop duration or strategy datasets.

## Quick start

Calculate a rolling average of a team's pit stop performance to capture crew fatigue or efficiency trends:
```python
import pandas as pd

# Dynamic crew performance feature (avoiding global data leakage)
df['team_pit_rolling_avg'] = (
    df.groupby('constructorId')['pit_duration']
    .transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
)
```

## F1 & Pit Stops Domain Concepts

### 1. Tires & Degradation (O Desgaste dos Pneus)
- **Compound Age (Idade do Composto):** The number of laps completed on the current set of tires. High lap counts mean higher tire degradation, which reduces grip and increases pit stop likelihood.
- **Compound Type (Tipo de Composto):** Hard, Medium, Soft, Intermediate, Wet. Harder tires degrade slower but are slower per lap; softer tires are fast but degrade rapidly.
- **Estimated Degradation Rate:** A calculated proxy variable, e.g., $\text{Laps on Set} \times \text{Track Temperature}$.

### 2. Crew and Pit Lane Performance (Equipe e Desempenho no Boxe)
- **Constructor Pit Stop History:** A rolling window average of a team's past pit stop times in the current season (captures team speed and pit stop errors).
- **Pit Lane Speed Limit & Loss Delta:** The average delta time lost just by driving through the pit lane at the speed limit (varies heavily by track, e.g., Monza is short, Spa-Francorchamps is very long).

### 3. Race Context & Dynamics (Contexto da Corrida)
- **Safety Car / VSC (Virtual Safety Car) Flags:** Pit stops under Safety Car are "cheap" because the field is driving slowly, reducing the time lost relative to cars on track.
  - *Engineered Feature:* `is_under_safety_car` or `delta_under_safety_car`.
- **Weather & Track State:** Track temperature and rain indicator. If it starts raining, teams will pit immediately for Intermediates or Wets.
  - *Engineered Feature:* `weather_change_indicator` (detects if track state changed from dry to wet).

### 4. Driver Performance (Desempenho do Piloto)
- **Driver Age/Experience:** Total F1 races participated.
- **In-lap and Out-lap Deltas:** The speed of the driver entering and exiting the pits. Veteran drivers tend to hit the speed limit line perfectly and exit cleanly.
