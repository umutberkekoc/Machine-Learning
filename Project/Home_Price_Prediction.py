import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, RobustScaler, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
import warnings

pd.set_option("display.width", 700)
pd.set_option("display.max_columns", None)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

"""
SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.
MSSubClass: The building class
MSZoning: The general zoning classification
LotFrontage: Linear feet of street connected to property
LotArea: Lot size in square feet
Street: Type of road access
Alley: Type of alley access
LotShape: General shape of property
LandContour: Flatness of the property
Utilities: Type of utilities available
LotConfig: Lot configuration
LandSlope: Slope of property
Neighborhood: Physical locations within Ames city limits
Condition1: Proximity to main road or railroad
Condition2: Proximity to main road or railroad (if a second is present)
BldgType: Type of dwelling
HouseStyle: Style of dwelling
OverallQual: Overall material and finish quality
OverallCond: Overall condition rating
YearBuilt: Original construction date
YearRemodAdd: Remodel date
RoofStyle: Type of roof
RoofMatl: Roof material
Exterior1st: Exterior covering on house
Exterior2nd: Exterior covering on house (if more than one material)
MasVnrType: Masonry veneer type
MasVnrArea: Masonry veneer area in square feet
ExterQual: Exterior material quality
ExterCond: Present condition of the material on the exterior
Foundation: Type of foundation
BsmtQual: Height of the basement
BsmtCond: General condition of the basement
BsmtExposure: Walkout or garden level basement walls
BsmtFinType1: Quality of basement finished area
BsmtFinSF1: Type 1 finished square feet
BsmtFinType2: Quality of second finished area (if present)
BsmtFinSF2: Type 2 finished square feet
BsmtUnfSF: Unfinished square feet of basement area
TotalBsmtSF: Total square feet of basement area
Heating: Type of heating
HeatingQC: Heating quality and condition
CentralAir: Central air conditioning
Electrical: Electrical system
1stFlrSF: First Floor square feet
2ndFlrSF: Second floor square feet
LowQualFinSF: Low quality finished square feet (all floors)
GrLivArea: Above grade (ground) living area square feet
BsmtFullBath: Basement full bathrooms
BsmtHalfBath: Basement half bathrooms
FullBath: Full bathrooms above grade
HalfBath: Half baths above grade
Bedroom: Number of bedrooms above basement level
Kitchen: Number of kitchens
KitchenQual: Kitchen quality
TotRmsAbvGrd: Total rooms above grade (does not include bathrooms)
Functional: Home functionality rating
Fireplaces: Number of fireplaces
FireplaceQu: Fireplace quality
GarageType: Garage location
GarageYrBlt: Year garage was built
GarageFinish: Interior finish of the garage
GarageCars: Size of garage in car capacity
GarageArea: Size of garage in square feet
GarageQual: Garage quality
GarageCond: Garage condition
PavedDrive: Paved driveway
WoodDeckSF: Wood deck area in square feet
OpenPorchSF: Open porch area in square feet
EnclosedPorch: Enclosed porch area in square feet
3SsnPorch: Three season porch area in square feet
ScreenPorch: Screen porch area in square feet
PoolArea: Pool area in square feet
PoolQC: Pool quality
Fence: Fence quality
MiscFeature: Miscellaneous feature not covered in other categories
MiscVal: $Value of miscellaneous feature
MoSold: Month Sold
YrSold: Year Sold
SaleType: Type of sale
SaleCondition: Condition of sale"""
# Görev 1: Keşifçi Veri Analizi

# Adım 1: Train ve test veri setlerini okutup birleştiriniz

train_df = pd.read_csv("ML Datasets/train.csv")
test_df = pd.read_csv("ML Datasets/test.csv")

train_df.head()
test_df.head()

dataframe = pd.concat([train_df, test_df], ignore_index=True)
train_df.shape
test_df.shape
dataframe.shape
dataframe.head()
dataframe.tail()
df = dataframe.copy()

# Adım 2: Numerik ve Kategorik değişkenleri yakalayınız

def grab_variables(dataframe, num_th=10, car_th=20):
    # Categoric Variables #
    cat_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["category", "object", "bool"]]
    num_but_cat = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"]
                   and dataframe[i].nunique() < num_th]
    cat_but_car = [i for i in dataframe.columns if dataframe[i].dtypes in ["category", "object"]
                   and dataframe[i].nunique() > car_th]
    cat_var = cat_var + num_but_cat
    cat_var = [i for i in cat_var if i not in cat_but_car]

    # Numeric Variables #
    num_var = [i for i in dataframe.columns if dataframe[i].dtypes in ["int64", "float64"]
               and dataframe[i].nunique() > num_th]

    print("Observations: {}".format(dataframe.shape[0]))
    print("Variables:", dataframe.shape[1])
    print("Categoric Variables:", len(cat_var))
    # print("Numeric But Categoric Variables:", len(num_but_cat))
    print("Categoric But Cardinal Variables:", len(cat_but_car))
    print("Numeric Variables:", len(num_var))
    return cat_var, cat_but_car, num_var

grab_variables(df)
cat_var, cat_but_car, num_var = grab_variables(df)

# Adım 3: Gerekli düzenlemeleri yapınız. (Tip hatası olan değişkenler gibi)
df.info()


# Adım 4: Numerik ve Kategoriik değişkenlerin veri içindeki dağılımını gözlemleyiniz

# Analysis of Categoric Variables:
def cat_analyze(dataframe, cat_var, plot=False):
    print(pd.DataFrame({cat_var: dataframe[cat_var].value_counts(),
                        "Ratio": dataframe[cat_var].value_counts() * 100 / len(dataframe)}), end="\n\n")
    if plot:
        sns.countplot(data=dataframe, x=cat_var)
        plt.grid()
        plt.show(block=True)

for i in cat_var:
    print(i, cat_analyze(df, i))

# Analysis of Numeric Variables:
def num_analyze(dataframe, num_var, plot=False):
    quantiles = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    print(dataframe[num_var].describe(quantiles).T)
    print("*************************")
    if plot:
        sns.histplot(data=dataframe, x=num_var, bins=20, color="red")
        plt.grid()
        plt.show(block=True)

for i in num_var:
    print(i, num_analyze(df, i))

# Adım 5: Kategorik değişkenler ile hedef değişken incelemesini yapınız
# Analysis of Target with Categoric Variables:
def target_analyze(dataframe, variable, target):
    print(dataframe.groupby(variable).agg({target: ["mean", "max", "min"]}), end="\n\n\n")

for i in cat_var:
    print(i, target_analyze(df, i, "SalePrice"))

# Adım 6: Aykırı Gözlem var mı inceleyiniz

def outlier_thresholds(dataframe, variable, q1=0.01, q3=0.99):
    q1 = dataframe[variable].quantile(q1)
    q3 = dataframe[variable].quantile(q3)
    iqr = q3 - q1
    lower_limit = q1 - 1.5 * iqr
    upper_limit = q3 + 1.5 * iqr
    return lower_limit, upper_limit

def check_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
        return True
    else:
        return False

def suppress_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < lower_limit), variable] = lower_limit
    dataframe.loc[(dataframe[variable] > upper_limit), variable] = upper_limit
    print(variable + " Has Suppressed")

def show_outliers(dataframe, variable, head=3):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    if dataframe[(dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit)].shape[0] > 0:
        print(variable, dataframe[dataframe[variable] < lower_limit | dataframe[variable] > upper_limit].head(head))
    else:
        return "There is no outliers for " + variable

def remove_outliers(dataframe, variable):
    lower_limit, upper_limit = outlier_thresholds(dataframe, variable)
    dataframe_without_outliers = dataframe[~((dataframe[variable] < lower_limit) | (dataframe[variable] > upper_limit))]
    return  dataframe_without_outliers


# Adım 7: Eksik gözlem var mı inceleyiniz
def missing_values_table(dataframe, na_name=False):
    na_columns = [i for i in dataframe.columns if dataframe[i].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / len(dataframe) * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns

missing_values_table(df)


# Görev 2: Feature Engineering

# Adım 1: Eksik ve aykırı gözlemler için gerekli işlemleri yapınız
for i in num_var:
    print(i, outlier_thresholds(df, i))

for i in num_var:
    print(i, check_outliers(df, i))

for i in num_var:
    if i != "SalePrice":
        if check_outliers(df, i) == True:
            print(i, suppress_outliers(df, i))

missing_values_table(df)
# Bazı değişkenlerdeki boş değerler evin o özelliğe sahip olmadığını ifade etmektedir
no_cols = ["Alley", "BsmtQual", "BsmtCond", "BsmtExposure",
           "BsmtFinType1", "BsmtFinType2", "FireplaceQu",
           "GarageType", "GarageFinish", "GarageQual",
           "GarageCond", "PoolQC", "Fence", "MiscFeature"]

for i in no_cols:
    print(i, df[i].isnull().sum())

for i in no_cols:
    df[i].fillna("No", inplace=True)
    print(i, df[i].isnull().sum())

missing_values_table(df)


def deal_with_missings(dataframe):
    nan_var = [i for i in dataframe.columns if dataframe[i].isnull().sum() > 0
               and i != "SalePrice"]
    for i in nan_var:
        print(" INFORMATION OF NAN VARIABLES\n", dataframe[i].describe().T)
        print(" **** Options ****\n1. Fill NA\n2. Drop NA")
        option = int(input("Enter you choice, 1/2"))
        if option == 1:
            print(" **** Options ****\n1. mean\n2. median\n3. mode\n4. variance\n5. Specific Number")
            option = int(input("Enter you choice, 1/2/3/4"))
            if option == 1:
                dataframe[i].fillna(dataframe[i].mean(), inplace=True)
            elif option == 2:
                dataframe[i].fillna(dataframe[i].median(), inplace=True)
            elif option == 3:
                dataframe[i].fillna(dataframe[i].mode()[0], inplace=True)
            elif option == 4:
                dataframe[i].fillna(dataframe[i].var(), inplace=True)
            elif option == 5:
                dataframe[i].fillna(int(input("Enter a number to filling nan values")), inplace=True)
            else:
                print("Enter a correct option number!!!")
                break
        elif option == 2:
            dataframe.dropna(inplace=True)
        else:
            print("Enter a correct option number!!!")

deal_with_missings(df)
missing_values_table(df)

cat_var, cat_but_car, num_var = grab_variables(df)

# Adım 2: Rare Encoder Uygulayınız
def rare_analyser(dataframe, target, cat_var):
    for i in cat_var:
        print(i, ":", len(dataframe[i].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[i].value_counts(),
                            "RATIO": dataframe[i].value_counts() * 100 / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(i)[target].mean()}), end="\n\n\n")

rare_analyser(df, "SalePrice", cat_var)

def rare_encoder(dataframe, rare_perc=0.1):
    temp_df = dataframe.copy()

    rare_columns = [i for i in temp_df.columns if temp_df[i].dtypes == "Object"
                    and (temp_df[i].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df

rare_encoder(df, 0.01)
df.head()

# Adım 3: Yeni Değişkenler Oluşturunuz ve başına NEW ekleyiniz
# Total Floor
df["NEW_TotalFlrSF"] = df["1stFlrSF"] + df["2ndFlrSF"]  # 32

# Total Finished Basement Area
df["NEW_TotalBsmtFin"] = df.BsmtFinSF1 + df.BsmtFinSF2  # 56

# Porch Area
df["NEW_PorchArea"] = df.OpenPorchSF + df.EnclosedPorch + df.ScreenPorch + df["3SsnPorch"] + df.WoodDeckSF  # 93

# Total House Area
df["NEW_TotalHouseArea"] = df.NEW_TotalFlrSF + df.TotalBsmtSF  # 156

df["NEW_TotalSqFeet"] = df.GrLivArea + df.TotalBsmtSF # 35


# Lot Ratio
df["NEW_LotRatio"] = df.GrLivArea / df.LotArea  # 64

df["NEW_RatioArea"] = df.NEW_TotalHouseArea / df.LotArea  # 57

df["NEW_GarageLotRatio"] = df.GarageArea / df.LotArea  # 69

# MasVnrArea
df["NEW_MasVnrRatio"] = df.MasVnrArea / df.NEW_TotalHouseArea  # 36

# Dif Area
df["NEW_DifArea"] = (df.LotArea - df["1stFlrSF"] - df.GarageArea - df.NEW_PorchArea - df.WoodDeckSF)  # 73

df["NEW_OverallGrade"] = df["OverallQual"] * df["OverallCond"]  # 61


df["NEW_Restoration"] = df.YearRemodAdd - df.YearBuilt  # 31

df["NEW_HouseAge"] = df.YrSold - df.YearBuilt  # 73

df["NEW_RestorationAge"] = df.YrSold - df.YearRemodAdd  # 40

df["NEW_GarageAge"] = df.GarageYrBlt - df.YearBuilt  # 17

df["NEW_GarageRestorationAge"] = np.abs(df.GarageYrBlt - df.YearRemodAdd)  # 30

df["NEW_GarageSold"] = df.YrSold - df.GarageYrBlt  # 48

drop_list = ["Street", "Alley", "LandContour", "Utilities", "LandSlope",
             "Heating", "PoolQC", "MiscFeature", "Neighborhood"]
df.drop(drop_list, axis=1, inplace=True)
df.head()
cat_var, cat_but_car, num_var = grab_variables(df)

# Adım 4: Encoding İşlemlerini Gerçekleştiriniz

def label_encoder(dataframe, variable):
    label_encoder = LabelEncoder()
    dataframe[variable] = label_encoder.fit_transform(dataframe[variable])
    print("0, 1-->", label_encoder.inverse_transform([0, 1]))
    return dataframe

bin_var = [i for i in df.columns if df[i].nunique() == 2]

for i in bin_var:
    label_encoder(df, i)


def one_hot_encoder(dataframe, variable):
    dataframe = pd.get_dummies(data=dataframe, columns=variable, dtype="int64", drop_first=True, dummy_na=False)
    return dataframe

df = one_hot_encoder(df, cat_var)
df.head()

# Görev 3: Model Kurma

# Adım 1: Train ve Test verisini ayırınız (SalePrice değişkeni boş olan değerler test verisidir)
test = df[df["SalePrice"].isnull() == True]
train = df[df["SalePrice"].isnull() == False]

y = train["SalePrice"]
X = train.drop(["SalePrice", "Id"], axis=1)

# Adım 2: Train verisi ile model kurup, model başarısını değerlendiriniz
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

models = [('LR', LinearRegression()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor())]

for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, X, y, cv=5, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")