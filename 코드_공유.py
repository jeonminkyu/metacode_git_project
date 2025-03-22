# 라이브러리 및 데이터 불러오기

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.datasets import load_wine

from sklearn.model_selection import train_test_split, GridSearchCV

import matplotlib.pyplot as plt

wine = load_wine()

# feature로 사용할 데이터에서는 'target' 컬럼을 drop합니다.
# target은 'target' 컬럼만을 대상으로 합니다.
# X, y 데이터를 test size는 0.2, random_state 값은 42로 하여 train 데이터와 test 데이터로 분할합니다.

''' 코드 작성 바랍니다 '''
wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target  # target 추가

# feature와 target 분리
X = wine_df.drop('target', axis=1)
y = wine_df['target']
# train/test 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

####### A 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
# 하이퍼파라미터 후보
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [2,5],
    'min_samples_split': [2,10],
    'min_samples_leaf': [1, 2, 4]
}
#모델 정의 및 GridSearchCV 설정
DT =  DecisionTreeClassifier(random_state=42)
grid_search = GridSearchCV(DT, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

#최적 모델로 예측 및 평가
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Best Hyper-parameter:", grid_search.best_params_)
print("Best Score:", accuracy)

#Feature Importance 시각화
importances = pd.Series(best_model.feature_importances_, index=X.columns)


plt.figure(figsize=(16, 10))
sns.barplot(x=importances.index, y=importances)
plt.title("Feature Importance ")
plt.xlabel("Features")
plt.ylabel("importances")
plt.xticks(rotation =45)
plt.tight_layout()
plt.show()

####### B 작업자 작업 수행 #######

''' 코드 작성 바랍니다 '''

