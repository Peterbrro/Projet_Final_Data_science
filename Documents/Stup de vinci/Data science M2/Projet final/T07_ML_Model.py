import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. Chargement des données (issues de la T05/T06)
df = pd.read_parquet("data/GBPUSD_M15_FEATURES.parquet")

# 2. Création de la cible (y) : 1 si la bougie suivante monte, 0 sinon
# On utilise shift(-1) pour regarder le futur
df['target'] = (df['close'].shift(-1) > df['close']).astype(int)

# 3. Split temporel strict 
train_df = df[df.index.year == 2022].copy()
val_df = df[df.index.year == 2023].copy()
test_df = df[df.index.year == 2024].copy()

# Suppression de la dernière ligne (qui a un NaN en target à cause du shift)
train_df = train_df.dropna()
val_df = val_df.dropna()

# 4. Sélection des features (on retire les colonnes prix et la cible)
features = [col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'target', 'returns']]

X_train, y_train = train_df[features], train_df['target']
X_val, y_val = val_df[features], val_df['target']

print(f"--- T07 : Entraînement ML ---")
print(f"Train (2022): {len(X_train)} samples")
print(f"Val (2023): {len(X_val)} samples")

# 5. Modèle Baseline ML
model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# 6. Évaluation simple
y_pred = model.predict(X_val)
acc = accuracy_score(y_val, y_pred)

print(f"Accuracy sur Validation (2023) : {acc:.2%}")
print("\nClassification Report :")
print(classification_report(y_val, y_pred))

# Note : On ne touche pas encore au test_df (2024) pour éviter l'overfitting ! [cite: 169]