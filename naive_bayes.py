from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('model', GaussianNB())  
])


param_grid = {
    'model__var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6]  
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
