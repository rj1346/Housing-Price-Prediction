import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

training_data= pd.read_csv('train.csv')
testing_data= pd.read_csv('test.csv')
#Concatinating the training_data and testing_data
train=pd.concat([training_data,testing_data], axis=0)
print(train.head())
print(train.describe())

#Checking the heatmap for null values
sns.heatmap(train.isnull(),yticklabels=False)
plt.show()

#finding all the null values column wise
sum_null=train.isnull().sum(axis=0).sort_values(ascending=False)
print(sum_null)


#Removing columns with maximum missing data
train.drop(['Alley', 'PoolQC', 'MiscFeature', 'Fence','FireplaceQu'], axis=1, inplace=True)
#checking the heatmap
sns.heatmap(train.isnull(), yticklabels=False)
plt.show()

#Filling missing data with mean/mode of their respective column
train['LotFrontage']= train['LotFrontage'].fillna(train['LotFrontage'].mean())
train['GarageQual']= train['GarageQual'].fillna(train['GarageQual'].mode()[0])
train['GarageYrBlt']= train['GarageYrBlt'].fillna(train['GarageYrBlt'].mode()[0])
train.GarageCond= train.GarageCond.fillna(train.GarageCond.mode()[0])
train.GarageFinish= train.GarageFinish.fillna(train.GarageFinish.mode()[0])
train.GarageType= train.GarageType.fillna(train.GarageType.mode()[0])
train.BsmtCond= train.BsmtCond.fillna(train.BsmtCond.mode()[0])
train.BsmtExposure= train.BsmtExposure.fillna(train.BsmtExposure.mode()[0])
train.BsmtQual= train.BsmtQual.fillna(train.BsmtQual.mode()[0])
train.BsmtFinType2= train.BsmtFinType2.fillna(train.BsmtFinType2.mode()[0])
train.BsmtFinType1= train.BsmtFinType1.fillna(train.BsmtFinType1.mode()[0])
train.MasVnrType= train.MasVnrType.fillna(train.MasVnrType.mode()[0])
train.MasVnrArea= train.MasVnrArea.fillna(train.MasVnrArea.mean())
train.Utilities= train.Utilities.fillna(train.Utilities.mode()[0])
train.ExterQual= train.ExterQual.fillna(train.ExterQual.mode()[0])
train.BsmtFullBath= train.BsmtFullBath.fillna(train.BsmtFullBath.mode()[0])
train.BsmtHalfBath= train.BsmtHalfBath.fillna(train.BsmtHalfBath.mode()[0])
train.GarageCars= train.GarageCars.fillna(train.GarageCars.mode()[0])
train.TotalBsmtSF= train.TotalBsmtSF.fillna(train.TotalBsmtSF.mode()[0])
train.BsmtFinSF1= train.BsmtFinSF1.fillna(train.BsmtFinSF1.mode()[0])
train.BsmtFinSF2= train.BsmtFinSF2.fillna(train.BsmtFinSF2.mode()[0])
train.BsmtUnfSF= train.BsmtUnfSF.fillna(train.BsmtUnfSF.mode()[0])
train.GarageArea= train.GarageArea.fillna(train.GarageArea.mode()[0])

#Checking the heatmap after filling the missing data
sns.heatmap(train.isnull(), yticklabels=False)
plt.show()

#priniting the columns containing null values
print(train.isnull().sum(axis=0).sort_values(ascending=False))

#Creating dummies
d1=pd.get_dummies(train.MSZoning, drop_first=True)
d2=pd.get_dummies(train.Street, drop_first=True)
d3=pd.get_dummies(train.LotShape, drop_first=True)
d4=pd.get_dummies(train.LandContour, drop_first=True)
d5=pd.get_dummies(train.Utilities, drop_first=True)
d6=pd.get_dummies(train.LotConfig, drop_first=True)
d7=pd.get_dummies(train.LandSlope, drop_first=True)
d8=pd.get_dummies(train.Neighborhood, drop_first=True)
d9=pd.get_dummies(train.Condition1, drop_first=True)
d10=pd.get_dummies(train.BldgType, drop_first=True)
d11=pd.get_dummies(train.Condition2, drop_first=True)
d12=pd.get_dummies(train.HouseStyle, drop_first=True)
d13=pd.get_dummies(train.SaleType, drop_first=True)
d14=pd.get_dummies(train.SaleCondition, drop_first=True)
d15=pd.get_dummies(train.ExterCond, drop_first=True)
d16=pd.get_dummies(train.ExterQual, drop_first=True)
d17=pd.get_dummies(train.Foundation, drop_first=True)
d18=pd.get_dummies(train.BsmtQual, drop_first=True)
d19=pd.get_dummies(train.BsmtCond, drop_first=True)
d20=pd.get_dummies(train.BsmtExposure, drop_first=True)
d21=pd.get_dummies(train.BsmtFinType1, drop_first=True)
d22=pd.get_dummies(train.BsmtFinType2, drop_first=True)
d23=pd.get_dummies(train.RoofStyle, drop_first=True)
d24=pd.get_dummies(train.RoofMatl, drop_first=True)
d25=pd.get_dummies(train.Exterior1st, drop_first=True)
d26=pd.get_dummies(train.Exterior2nd, drop_first=True)
d27=pd.get_dummies(train.MasVnrType, drop_first=True)
d28=pd.get_dummies(train.Heating, drop_first=True)
d29=pd.get_dummies(train.HeatingQC, drop_first=True)
d30=pd.get_dummies(train.CentralAir, drop_first=True)
d31=pd.get_dummies(train.Electrical, drop_first=True)
d32=pd.get_dummies(train.KitchenQual, drop_first=True)
d33=pd.get_dummies(train.Functional, drop_first=True)
d34=pd.get_dummies(train.GarageType, drop_first=True)
d35=pd.get_dummies(train.GarageFinish, drop_first=True)
d36=pd.get_dummies(train.GarageQual, drop_first=True)
d37=pd.get_dummies(train.GarageCond, drop_first=True)
d38=pd.get_dummies(train.PavedDrive, drop_first=True)


#Contatinating the new columns
train=pd.concat([train,d1,d2,d3,d4,d5,d6,d7,d8,d9,d10,d11,d12,d13,d14,d15,d16,d17,d18,d19,d20,
                   d21,d22,d23,d24,d25,d26,d27,d28,d29,d30,d31,d32,d33,d34,d35,d36,d37,d38],axis=1)

#dropping unnecssary columns
train.drop(['MSZoning', 'Street', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood',
            'Condition2', 'BldgType', 'Condition1', 'HouseStyle', 'SaleType', 'SaleCondition', 'ExterCond', 'ExterQual',
            'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'RoofStyle', 'RoofMatl', 'Exterior1st',
            'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 'Functional',
            'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive','Foundation'], axis=1, inplace=True)


#checking the heatmap for missing data
sns.heatmap(train.isnull(), yticklabels=False)
plt.show()
#printing the column with null values
print(train.isnull().sum(axis=1).sort_values(ascending=False))


#Dividing the train dataset into training and testing dataset
df_train=train.iloc[:1460,:]
df_test=train.iloc[1460:,:]


#Spliting dataset into X_train, y_train, X_test
X_train= df_train.drop('SalePrice', axis=1)
y_train= df_train.SalePrice
X_test=df_test.drop('SalePrice', axis=1)

#checking if there is any missing data in X_train and X_test
sns.heatmap(X_train.isnull(), yticklabels=False)
plt.show()
sns.heatmap(X_test.isnull(), yticklabels=False)
plt.show()

#Applying Random Forest
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor()
regressor.fit(X_train,y_train)
y_pred=regressor.predict(X_test)
y_pred=pd.DataFrame(data=y_pred)

#importing sample submission
sample_submission= pd.read_csv("sample_submission.csv")

#Creating a DataFrame to store the values of y_pred along with Id
submission= pd.concat([sample_submission.Id, y_pred], axis=1)
submission.columns=['Id','SalePrice']
print(submission)

#Saving 'submission' in csv
submission.to_csv('submission.csv', index=False)
