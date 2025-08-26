from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

pipeline = Pipeline([
    ('scaler', StandardScaler()),  
    ('model', KNeighborsClassifier()) 
])

param_grid = {
    'model__n_neighbors': [3, 5, 7, 10, 15], 
    'model__weights': ['uniform', 'distance'],  
    'model__p': [1, 2], 
    'model__metric': ['minkowski', 'euclidean', 'manhattan']
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
