from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from catboost import CatBoostClassifier

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', CatBoostClassifier(random_state=42, silent=True))
])  

param_grid = {
    'model__iterations': [100, 200], 
    'model__learning_rate': [0.01, 0.1],
    'model__depth': [4, 6],  
    'model__l2_leaf_reg': [0, 3], 
    'model__border_count': [32, 64], 
    'model__random_strength': [0, 1] 
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
