import os
import joblib
from sklearn.preprocessing import LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6
HOUSING_PATH = "datasets/housing"
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
class label_(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        encoder=LabelBinarizer()
        df_word_encode=encoder.fit_transform(X[self.attribute_names])
        return df_word_encode

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

def display_scores(tree_reg, housing_prepared, housing_labels):
    scores=cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)

    scores = np.sqrt(-scores)
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

'''                 加载数据             '''
df=load_housing_data()
df["income_cat"] = np.ceil(df["median_income"] / 1.5)
df["income_cat"].where(df["income_cat"] < 5, 5.0, inplace=True)  #将高收入群体划为5
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(df, df["income_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

#删除自定义数据
strat_test_set.drop(['income_cat'],axis=1,inplace=True)
strat_train_set.drop(['income_cat'],axis=1,inplace=True)


'''                            特征预处理                   '''
df_num = strat_train_set.drop(["ocean_proximity","median_house_value"], axis=1)
num_attribs = list(df_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),  #将df转成np数组
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])
cat_pipeline = Pipeline([
        ('selector', label_(cat_attribs)),
        # ('selector', DataFrameSelector(cat_attribs)),
        ('label_binarizer', OneHotEncoder()),
    ])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])


# 初步模型选择
'''                                                模型选择             '''
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def mode_select(housing_prepared, housing_labels):
    # model 1
    forest_reg = RandomForestRegressor()
    forest_reg.fit(housing_prepared, housing_labels)
    print("RandomForestRegressor is \n")
    display_scores(forest_reg, housing_prepared, housing_labels)
    print("\n")

    # model 2
    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(housing_prepared, housing_labels)
    print("RandomForestRegressor is \n")
    display_scores(tree_reg, housing_prepared, housing_labels)
    print("\n")

    # model 3
    lin_reg = LinearRegression()
    lin_reg.fit(housing_prepared, housing_labels)
    print("RandomForestRegressor is \n")
    # 交叉验证
    display_scores(lin_reg, housing_prepared, housing_labels)
    print("\n")
def  main():

    housing_prepared = strat_train_set.drop("median_house_value", axis=1)
    housing_labels = strat_train_set["median_house_value"].copy()
    #特征处理
    housing_prepared = full_pipeline.fit_transform(housing_prepared)
    # 模型选择
    mode_select(housing_prepared, housing_labels)

    my_model_loaded = RandomForestRegressor()
    my_model_loaded.fit(housing_prepared, housing_labels)


    #模型微调2
    from sklearn.model_selection import RandomizedSearchCV
    from scipy.stats import randint
    param_dist = {
        'n_estimators': randint(10, 100),  # 10到100之间的整数
        'max_features': ['auto', 'sqrt', 'log2', None],  # 从列表中随机选择
        'max_depth': [None, 10, 20, 30, 40, 50],  # 从列表中随机选择
        'min_samples_split': randint(2, 20),  # 2到20之间的整数
        'min_samples_leaf': randint(1, 20)  # 1到20之间的整数
    }
    random_search = RandomizedSearchCV(
        my_model_loaded,
        param_distributions=param_dist,
        n_iter=10,  # 进行10次随机采样
        scoring='neg_mean_squared_error',  # 使用均方误差作为评分指标
        cv=5,  # 5折交叉验证
        verbose=1,  # 显示详细信息
        n_jobs=-1  # 使用所有可用的处理器
    )

    # 4. 执行搜索
    random_search.fit(housing_prepared, housing_labels)

    # 5. 获取最佳参数
    best_params = random_search.best_params_
    print("Best Hyperparameters:", best_params)
    my_model_loaded=random_search.best_estimator_


    #测试集评估
    y_test = strat_test_set["median_house_value"].copy()
    X_test = strat_test_set.drop("median_house_value", axis=1)
    X_test_prepared = full_pipeline.transform(X_test)
    final_predictions = my_model_loaded.predict(X_test_prepared)

    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)   # => evaluates to 48,209.6
    print(final_rmse)

if __name__ == '__main__':
    main()
