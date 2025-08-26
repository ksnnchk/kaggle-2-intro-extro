from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42))
])

param_grid = {
    'model__penalty': ['l1', 'l2'],
    'model__C': np.logspace(-4, 4, 20),
    'model__solver': ['liblinear', 'saga'],
    'model__class_weight': [None, 'balanced'],
    'model__max_iter': [100, 500, 1000]
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

accuracy_score(val_y, y_pred)
