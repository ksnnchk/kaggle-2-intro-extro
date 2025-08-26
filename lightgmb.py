from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from lightgbm import LGBMClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LGBMClassifier(random_state=42))
])

param_grid = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.01, 0.1],
    'model__num_leaves': [31, 63],
    'model__max_depth': [3, 5],
    'model__min_child_samples': [20, 50],
    'model__subsample': [0.8, 1.0], 
    'model__colsample_bytree': [0.8, 1.0], 
    'model__reg_alpha': [0, 0.1],
    'model__reg_lambda': [0, 0.1]
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    n_jobs=-1,
    verbose=1
)

grid_search.fit(tr_X, tr_y)
best_model = grid_search.best_estimator_

y_pred = best_model.predict(val_X)

accuracy_score(y_pred, val_y)
