from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('model', RandomForestClassifier(random_state=42)) 
])


param_grid = {
    'model__n_estimators': [100, 200, 300], 
    'model__max_depth': [None, 5, 10, 20], 
    'model__min_samples_split': [2, 5, 10],  
    'model__min_samples_leaf': [1, 2, 4],  
    'model__max_features': ['sqrt', 'log2']  
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
