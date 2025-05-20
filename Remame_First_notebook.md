***Python Version:*** *3.11.8*  <BR>
# **House Prices - Advanced Regression Techniques**
## **Competition Description:**  
Ask a home buyer to describe their dream house, and they probably won't begin with the height of the basement ceiling or the proximity to an east-west railroad. But this playground competition's dataset proves that much more influences price negotiations than the number of bedrooms or a white-picket fence.

With 79 explanatory variables describing (almost) every aspect of residential homes in Ames, Iowa, this competition challenges you to predict the final price of each home.    

## **Evaluation**
### **Goal**
It is your job to predict the sales price for each house. For each Id in the test set, you must predict the value of the SalePrice variable. 

### **Metric**
Submissions are evaluated on Root-Mean-Squared-Error (RMSE) between the logarithm of the predicted value and the logarithm of the observed sales price. (Taking logs means that errors in predicting expensive houses and cheap houses will affect the result equally.)

### **Submission File Format**
The file should contain a header and have the following format:


Id,SalePrice  
1461,169000.1  
1462,187724.1233  
1463,175221  
etc.  

<BR><BR>
# **Rough Plan:**

Although I am aware of stacking and it may provide a better model result if I stack multiple models together, for this project I am mainly going to using the information and models available to learn on Kaggle in the [Kaggle Learn](https://www.kaggle.com/learn) section.  <BR><BR>
The model we are going to use for this project is **XGBoost** or Extreme Gradient Boosting. As Kaggle Learn does not teach all the models available, this seems to be the obvious choice as to what would perform best. We could try out other models and compare, but for this notebook we will focus on correct feature selection, creative feature engineering and hyperparameter tuning.

### **Rough workflow for now:**
1. Domain/Feature understanding + Plan  
2. Initial EDA (nulls, data types)  - 2 to 4 could be as one
3. Handle missing values  
4. Deeper EDA (plots, feature interactions)  
5. Basic encoding  
6. Build base XGBoost model (log SalePrice)  
7. Feature engineering + iterative testing (what feature engineering?, target encoding, PCA values, PCA as feature discovery  
&nbsp;&nbsp; rememeber to also add any encodings/changes to the test set (mapping encodings, etc)
8. Explainability (SHAP, feature importance)  
9. Refine + resubmit  

<details> <!-- remove from .ipynb file, this is for md convert -->
<summary>Markdown (.md) set up code</Summary>


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.io.formats.style_render import StylerRenderer

StylerRenderer.render = lambda self, **kwargs: ""

#stestse

pd.set_option('max_colwidth',200)
pd.set_option('display.width',200)
pd.set_option('display.max_columns',500) 
pd.set_option('display.max_rows',1000)
```

</details>

<Details>
<Summary>Data Set-up code</Summary>


```python
import pandas as pd
df = pd.read_csv(r"https://raw.githubusercontent.com/ddrant/HousePrices_KaggleComp/refs/heads/main/data/train.csv")
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>MSSubClass</th>
      <th>MSZoning</th>
      <th>LotFrontage</th>
      <th>LotArea</th>
      <th>Street</th>
      <th>Alley</th>
      <th>LotShape</th>
      <th>LandContour</th>
      <th>Utilities</th>
      <th>LotConfig</th>
      <th>LandSlope</th>
      <th>Neighborhood</th>
      <th>Condition1</th>
      <th>Condition2</th>
      <th>BldgType</th>
      <th>HouseStyle</th>
      <th>OverallQual</th>
      <th>OverallCond</th>
      <th>YearBuilt</th>
      <th>YearRemodAdd</th>
      <th>RoofStyle</th>
      <th>RoofMatl</th>
      <th>Exterior1st</th>
      <th>Exterior2nd</th>
      <th>MasVnrType</th>
      <th>MasVnrArea</th>
      <th>ExterQual</th>
      <th>ExterCond</th>
      <th>Foundation</th>
      <th>BsmtQual</th>
      <th>BsmtCond</th>
      <th>BsmtExposure</th>
      <th>BsmtFinType1</th>
      <th>BsmtFinSF1</th>
      <th>BsmtFinType2</th>
      <th>BsmtFinSF2</th>
      <th>BsmtUnfSF</th>
      <th>TotalBsmtSF</th>
      <th>Heating</th>
      <th>HeatingQC</th>
      <th>CentralAir</th>
      <th>Electrical</th>
      <th>1stFlrSF</th>
      <th>2ndFlrSF</th>
      <th>LowQualFinSF</th>
      <th>GrLivArea</th>
      <th>BsmtFullBath</th>
      <th>BsmtHalfBath</th>
      <th>FullBath</th>
      <th>HalfBath</th>
      <th>BedroomAbvGr</th>
      <th>KitchenAbvGr</th>
      <th>KitchenQual</th>
      <th>TotRmsAbvGrd</th>
      <th>Functional</th>
      <th>Fireplaces</th>
      <th>FireplaceQu</th>
      <th>GarageType</th>
      <th>GarageYrBlt</th>
      <th>GarageFinish</th>
      <th>GarageCars</th>
      <th>GarageArea</th>
      <th>GarageQual</th>
      <th>GarageCond</th>
      <th>PavedDrive</th>
      <th>WoodDeckSF</th>
      <th>OpenPorchSF</th>
      <th>EnclosedPorch</th>
      <th>3SsnPorch</th>
      <th>ScreenPorch</th>
      <th>PoolArea</th>
      <th>PoolQC</th>
      <th>Fence</th>
      <th>MiscFeature</th>
      <th>MiscVal</th>
      <th>MoSold</th>
      <th>YrSold</th>
      <th>SaleType</th>
      <th>SaleCondition</th>
      <th>SalePrice</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>60</td>
      <td>RL</td>
      <td>65.0</td>
      <td>8450</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2003</td>
      <td>2003</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>196.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>No</td>
      <td>GLQ</td>
      <td>706</td>
      <td>Unf</td>
      <td>0</td>
      <td>150</td>
      <td>856</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>856</td>
      <td>854</td>
      <td>0</td>
      <td>1710</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>8</td>
      <td>Typ</td>
      <td>0</td>
      <td>NaN</td>
      <td>Attchd</td>
      <td>2003.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>548</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>61</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>208500</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>20</td>
      <td>RL</td>
      <td>80.0</td>
      <td>9600</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>Reg</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>Veenker</td>
      <td>Feedr</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>1Story</td>
      <td>6</td>
      <td>8</td>
      <td>1976</td>
      <td>1976</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>MetalSd</td>
      <td>MetalSd</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>CBlock</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Gd</td>
      <td>ALQ</td>
      <td>978</td>
      <td>Unf</td>
      <td>0</td>
      <td>284</td>
      <td>1262</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1262</td>
      <td>0</td>
      <td>0</td>
      <td>1262</td>
      <td>0</td>
      <td>1</td>
      <td>2</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>TA</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>1976.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>460</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>298</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>5</td>
      <td>2007</td>
      <td>WD</td>
      <td>Normal</td>
      <td>181500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>60</td>
      <td>RL</td>
      <td>68.0</td>
      <td>11250</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Inside</td>
      <td>Gtl</td>
      <td>CollgCr</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>2001</td>
      <td>2002</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>162.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Mn</td>
      <td>GLQ</td>
      <td>486</td>
      <td>Unf</td>
      <td>0</td>
      <td>434</td>
      <td>920</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>920</td>
      <td>866</td>
      <td>0</td>
      <td>1786</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>6</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2001.0</td>
      <td>RFn</td>
      <td>2</td>
      <td>608</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>42</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>9</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>223500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>70</td>
      <td>RL</td>
      <td>60.0</td>
      <td>9550</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>Corner</td>
      <td>Gtl</td>
      <td>Crawfor</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>7</td>
      <td>5</td>
      <td>1915</td>
      <td>1970</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>Wd Sdng</td>
      <td>Wd Shng</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>TA</td>
      <td>TA</td>
      <td>BrkTil</td>
      <td>TA</td>
      <td>Gd</td>
      <td>No</td>
      <td>ALQ</td>
      <td>216</td>
      <td>Unf</td>
      <td>0</td>
      <td>540</td>
      <td>756</td>
      <td>GasA</td>
      <td>Gd</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>961</td>
      <td>756</td>
      <td>0</td>
      <td>1717</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>3</td>
      <td>1</td>
      <td>Gd</td>
      <td>7</td>
      <td>Typ</td>
      <td>1</td>
      <td>Gd</td>
      <td>Detchd</td>
      <td>1998.0</td>
      <td>Unf</td>
      <td>3</td>
      <td>642</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>0</td>
      <td>35</td>
      <td>272</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>2</td>
      <td>2006</td>
      <td>WD</td>
      <td>Abnorml</td>
      <td>140000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>60</td>
      <td>RL</td>
      <td>84.0</td>
      <td>14260</td>
      <td>Pave</td>
      <td>NaN</td>
      <td>IR1</td>
      <td>Lvl</td>
      <td>AllPub</td>
      <td>FR2</td>
      <td>Gtl</td>
      <td>NoRidge</td>
      <td>Norm</td>
      <td>Norm</td>
      <td>1Fam</td>
      <td>2Story</td>
      <td>8</td>
      <td>5</td>
      <td>2000</td>
      <td>2000</td>
      <td>Gable</td>
      <td>CompShg</td>
      <td>VinylSd</td>
      <td>VinylSd</td>
      <td>BrkFace</td>
      <td>350.0</td>
      <td>Gd</td>
      <td>TA</td>
      <td>PConc</td>
      <td>Gd</td>
      <td>TA</td>
      <td>Av</td>
      <td>GLQ</td>
      <td>655</td>
      <td>Unf</td>
      <td>0</td>
      <td>490</td>
      <td>1145</td>
      <td>GasA</td>
      <td>Ex</td>
      <td>Y</td>
      <td>SBrkr</td>
      <td>1145</td>
      <td>1053</td>
      <td>0</td>
      <td>2198</td>
      <td>1</td>
      <td>0</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>1</td>
      <td>Gd</td>
      <td>9</td>
      <td>Typ</td>
      <td>1</td>
      <td>TA</td>
      <td>Attchd</td>
      <td>2000.0</td>
      <td>RFn</td>
      <td>3</td>
      <td>836</td>
      <td>TA</td>
      <td>TA</td>
      <td>Y</td>
      <td>192</td>
      <td>84</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0</td>
      <td>12</td>
      <td>2008</td>
      <td>WD</td>
      <td>Normal</td>
      <td>250000</td>
    </tr>
  </tbody>
</table>
</div>



</details>

<BR><BR>
# **1. Domain Knowledge, Feature review and transformation plan**

First we will start by looking through the [data_description.txt](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data?select=data_description.txt) file from the competition page feature by feature.<BR>
  
Some of the things that initially come to mind to explore are:
1. **Feature description** — What does it represent?
2. **Possible values** — Especially for categoricals; what do the values mean?
3. **Data type and missing values** — Type, % missing, any obvious issues
4. **Encoding** — Does it have a natural or ordinal order?
5. **Grouping** — Can it be grouped with similar features (e.g. size, time, location)?
6. **Interactions** — Does this feature affect or depend on other features?
7. **Expected importance** — High/medium/low signal? Any gut feel?
8. **Missing value strategy** — If there are few nulls, what could we fill with?
9. **Area knowledge** — Any local context (e.g. Ames zoning, neighborhoods)?
10. **Transformation ideas** — Binning, ratios, log scale, engineered combos?

SalePrice - the property's sale price in dollars. This is the target variable that you're trying to predict.<BR>

## **Feature List**
[**MSSubClass:**](#mssubclass) The building class<BR>
[**MSZoning:**](#mszoning) The general zoning classification<BR>
[**LotFrontage:**](#lotfrontage) Linear feet of street connected to property<BR>
[**LotArea:**](#lotarea) Lot size in square feet<BR>
[**Street:**](#street) Type of road access<BR>
[**Alley:**](#alley) Type of alley access<BR>
[**LotShape:**](#lotshape) General shape of property<BR>
[**LandContour:**](#landcontour) Flatness of the property<BR>
[**Utilities:**](#utilities) Type of utilities available<BR>
[**LotConfig:**](#lotconfig) Lot configuration<BR>
[**LandSlope:**](#landslope) Slope of property<BR>
[**Neighborhood:**](#neighborhood) Physical locations within Ames city limits<BR>
[**Condition1:**](#condition1) Proximity to main road or railroad<BR>
[**Condition2:**](#condition2) Proximity to main road or railroad (if a second is present)<BR>
[**BldgType:**](#bldgtype) Type of dwelling<BR>
[**HouseStyle:**](#housestyle) Style of dwelling<BR>
[**OverallQual:**](#overallqual) Overall material and finish quality<BR>
[**OverallCond:**](#overallcond) Overall condition rating<BR>
[**YearBuilt:**](#yearbuilt) Original construction date<BR>
[**YearRemodAdd:**](#yearremodadd) Remodel date<BR>
[**RoofStyle:**](#roofstyle) Type of roof<BR>
[**RoofMatl:**](#roofmatl) Roof material<BR>
[**Exterior1st:**](#exterior1st) Exterior covering on house<BR>
[**Exterior2nd:**](#exterior2nd) Exterior covering on house (if more than one material)<BR>
[**MasVnrType:**](#masvnrtype) Masonry veneer type<BR>
[**MasVnrArea:**](#masvnrarea) Masonry veneer area in square feet<BR>
[**ExterQual:**](#exterqual) Exterior material quality<BR>
[**ExterCond:**](#extercond) Present condition of the material on the exterior<BR>
[**Foundation:**](#foundation) Type of foundation<BR>
[**BsmtQual:**](#bsmtqual) Height of the basement<BR>
[**BsmtCond:**](#bsmtcond) General condition of the basement<BR>
[**BsmtExposure:**](#bsmtexposure) Walkout or garden level basement walls<BR>
[**BsmtFinType1:**](#bsmtfintype1) Quality of basement finished area<BR>
[**BsmtFinSF1:**](#bsmtfinsf1) Type 1 finished square feet<BR>
[**BsmtFinType2:**](#bsmtfintype2) Quality of second finished area (if present)<BR>
[**BsmtFinSF2:**](#bsmtfinsf2) Type 2 finished square feet<BR>
[**BsmtUnfSF:**](#bsmtunfsf) Unfinished square feet of basement area<BR>
[**TotalBsmtSF:**](#totalbsmtsf) Total square feet of basement area<BR>
[**Heating:**](#heating) Type of heating<BR>
[**HeatingQC:**](#heatingqc) Heating quality and condition<BR>
[**CentralAir:**](#centralair) Central air conditioning<BR>
[**Electrical:**](#electrical) Electrical system<BR>
[**1stFlrSF:**](#1stflrsf) First Floor square feet<BR>
[**2ndFlrSF:**](#2ndflrsf) Second floor square feet<BR>
[**LowQualFinSF:**](#lowqualfinsf) Low quality finished square feet (all floors)<BR>
[**GrLivArea:**](#grlivarea) Above grade (ground) living area square feet<BR>
[**BsmtFullBath:**](#bsmtfullbath) Basement full bathrooms<BR>
[**BsmtHalfBath:**](#bsmthalfbath) Basement half bathrooms<BR>
[**FullBath:**](#fullbath) Full bathrooms above grade<BR>
[**HalfBath:**](#halfbath) Half baths above grade<BR>
[**Bedroom:**](#bedroom) Number of bedrooms above basement level<BR>
[**Kitchen:**](#kitchen) Number of kitchens<BR>
[**KitchenQual:**](#kitchenqual) Kitchen quality<BR>
[**TotRmsAbvGrd:**](#totrmsabvgrd) Total rooms above grade (does not include bathrooms)<BR>
[**Functional:**](#functional) Home functionality rating<BR>
[**Fireplaces:**](#fireplaces) Number of fireplaces<BR>
[**FireplaceQu:**](#fireplacequ) Fireplace quality<BR>
[**GarageType:**](#garagetype) Garage location<BR>
[**GarageYrBlt:**](#garageyrblt) Year garage was built<BR>
[**GarageFinish:**](#garagefinish) Interior finish of the garage<BR>
[**GarageCars:**](#garagecars) Size of garage in car capacity<BR>
[**GarageArea:**](#garagearea) Size of garage in square feet<BR>
[**GarageQual:**](#garagequal) Garage quality<BR>
[**GarageCond:**](#garagecond) Garage condition<BR>
[**PavedDrive:**](#paveddrive) Paved driveway<BR>
[**WoodDeckSF:**](#wooddecksf) Wood deck area in square feet<BR>
[**OpenPorchSF:**](#openporchsf) Open porch area in square feet<BR>
[**EnclosedPorch:**](#enclosedporch) Enclosed porch area in square feet<BR>
[**3SsnPorch:**](#3ssnporch) Three season porch area in square feet<BR>
[**ScreenPorch:**](#screenporch) Screen porch area in square feet<BR>
[**PoolArea:**](#poolarea) Pool area in square feet<BR>
[**PoolQC:**](#poolqc) Pool quality<BR>
[**Fence:**](#fence) Fence quality<BR>
[**MiscFeature:**](#miscfeature) Miscellaneous feature not covered in other categories<BR>
[**MiscVal:**](#miscval) $Value of miscellaneous feature<BR>
[**MoSold:**](#mosold) Month Sold<BR>
[**YrSold:**](#yrsold) Year Sold<BR>
[**SaleType:**](#saletype) Type of sale<BR>
[**SaleCondition:**](#salecondition) Condition of sale<BR>

## **MSSubClass**
***description:***

Identifies the type of dwelling involved in the sale.

***Categories:***

| Value | Description                                   | Notes / Thoughts |
|-------|-----------------------------------------------|------------------|
| 20    | 1-STORY 1946 & NEWER ALL STYLES               | Category split by year built because of WWII housing boom, change of architecture and infrastructure. |
| 30    | 1-STORY 1945 & OLDER                          | — |
| 40    | 1-STORY W/FINISHED ATTIC ALL AGES             | — |
| 45    | 1-1/2 STORY - UNFINISHED ALL AGES             | 1/2 story means partial or smaller upstairs level, usually with slanted ceiling, unfinished may mean the 2nd story is unusable |
| 50    | 1-1/2 STORY FINISHED ALL AGES                 | — |
| 60    | 2-STORY 1946 & NEWER                          | — |
| 70    | 2-STORY 1945 & OLDER                          | — |
| 75    | 2-1/2 STORY ALL AGES                          | Same here — check vs attic or 3rd flr? |
| 80    | SPLIT OR MULTI-LEVEL                           | suburban type houses with split living spaces (ie living area bottom floor and bedrooms above) Typically 3 or 4 floors (incl basement?) — [Split level source](<https://therealestateguylv.com/blog/split-level-homes/#:~:text=A%20split-level%20home%20is%20a%20single-family%20dwelling%20with,above%2C%20and%20the%20recreational%20spaces%20or%20garages%20below.>) <BR> Note: split houses on a slope may tend to have smaller 2nd floors due to building contraints. We could investigate this interaction with the [Landslope](#landslope) feature|
| 85    | SPLIT FOYER                                   | Reverse split — main living upstairs, bedrooms below |
| 90    | DUPLEX - ALL STYLES AND AGES                  | Unit of housing with 2 seperate homes, above and below, or side-by-side. This most likely means the sale was for both units together (the full duplex)|
| 120   | 1-STORY PUD - (planned unit development) 1946 & NEWER | These PUDs are like communities of houses which all pay their share towards private ammenities in the neightborhood, like parks, security, yard maintenance. Tend to be more cheaper than similar non-PUD houses |
| 150   | 1-1/2 STORY PUD - ALL AGES                    | — |
| 160   | 2-STORY PUD - 1946 & NEWER                    | — |
| 180   | PUD - MULTILEVEL - INCL SPLIT LEV/FOYER       | — |
| 190   | 2 FAMILY CONVERSION - ALL STYLES AND AGES     | — |

*Cardianlity* - 16 <BR>
*Nulls* - 0 <BR>
*Feature interactions* - BldgType, HouseStyle, YearBuilt<BR>
*Feature Engineering* - Binary feautres (BuildBefore_1946, isPUD), Semi-high cardinality, maybe target encoding would be useful?

<BR><BR>


[*back to feature list*](#feature-list)


```python
df['MSSubClass'].isna().sum() # ZERO NULLS
```




    0



This feature is alsmost an amalgamation or kluster of other potential features.<BR>
Would this be more useful split up into different features? (still leave original feature)<BR>
1. Maybe we have a feature for number of stories (ie 1, 1.5, 2, 2.5 ,3) - (***Note: we already have feature [HouseStyle](#housestyle) which captures the number of stories and finished or unfinished***)
2. a boolean feature for whether it is a PUD or not.
3. Some of the features contain rough dates, do we keep these categories seperate or merge into one? We already have the feature yrBuilt telling us when the house was built maybe again we can create a binary feature telling us whether the building was built > 1945 (or been remodded since 1945?) - Then we can try reducing cardinality my merging the categories seperating classification by dates
4. If we look at [BldgType](#bldgtype) it also contains the category duplex, [HouseStyle](#housestyle) for the number of stories and finished info, then if we create a binary feature for each isPUD and BuiltBefore_1946, we may be able to remove this feature and have the model perform better with the new.


```python
#print(f'Total Number of rows in train is: {df['MSSubClass'].count()}')
```

***TODO:*** Make the below into a function, decide what parameters are needed, to start we need labels, x, y, cmap color, Lognorm cmap or Norm cmap or PowerNorm cmap option, etc (if lognorm/PowerNorm -> how to label the ticks for the colorbar, if PowerNorm -> Gamma value?)


```python
from matplotlib import cm, colors
# Box plot of the feature 

# compute the counts per category 
cat = 'MSSubClass'
counts = df[cat].value_counts()
order = sorted(df[cat].dropna().unique())


# mapping the category code to the description for our xticks (yticks if horizontal)
mssubclass_map = {
     20: "1-STORY 1946 & NEWER ALL STYLES",
     30: "1-STORY 1945 & OLDER",
     40: "1-STORY W/FINISHED ATTIC ALL AGES",
     45: "1-1/2 STORY – UNFINISHED ALL AGES",
     50: "1-1/2 STORY FINISHED ALL AGES",
     60: "2-STORY 1946 & NEWER",
     70: "2-STORY 1945 & OLDER",
     75: "2-1/2 STORY ALL AGES",
     80: "SPLIT OR MULTI-LEVEL",
     85: "SPLIT FOYER",
     90: "DUPLEX – ALL STYLES AND AGES",
    120: "1-STORY PUD (Planned Unit Dev) – 1946 & NEWER",
    150: "1-1/2 STORY PUD – ALL AGES",
    160: "2-STORY PUD – 1946 & NEWER",
    180: "PUD – MULTILEVEL (INCL SPLIT LEV/FOYER)",
    190: "2-FAMILY CONVERSION – ALL STYLES & AGES"
}

# Labels for the different categories
labels = [f"{category} - {mssubclass_map[category]}" for category in order]

# Create color map based on the value counts
norm = colors.Normalize(vmin=counts.min(), vmax=counts.max())

#or try lognorm for differnt cmap scaling
norm = colors.LogNorm(vmin=counts.min(), vmax=counts.max())

# last norm technique 'PowerNorm' like a gamma normalisation
norm = colors.PowerNorm(gamma=0.4, vmin=counts.min(), vmax=counts.max())


cmap = cm.get_cmap('plasma')
#cmap = cm.get_cmap('viridis') # also good for powernorm
#cmap = cm.get_cmap('inferno') # not bad for PowerNorm (gamma)
#cmap = cm.get_cmap('cool')
#cmap = cm.get_cmap('autumn')
palette = [cmap(norm(counts[category])) for category in order]

# Set the plot
plot = sns.catplot(data=df, x='SalePrice', y='MSSubClass', kind='boxen',  height=7, aspect=1.8, palette=palette, orient='h')

# Set the y-ticks
plot.set_yticklabels(labels, rotation=90)
plt.yticks(rotation=0)

# Add the colour bar key for the value counts

# Create a scalar mappable
sm = cm.ScalarMappable(cmap=cmap, norm=norm) # norm is our normalisation of the value counts into [0,1] to map to the colors on the cmap

cbar = plot.figure.colorbar(sm, ax=plot.ax, aspect=50, location='right')
cbar.set_label('Number of Instances')

# colorbar ticks for lognorm cmap
custom_ticks = [5, 10, 25, 50, 100, 200, 500]  # or whatever makes sense
cbar.set_ticks(custom_ticks)
cbar.set_ticklabels([str(t) for t in custom_ticks])


plt.title("SalesPrice by HSSubclass")
plt.show()
```

    C:\Users\Scotts\AppData\Local\Temp\ipykernel_45816\3741891665.py:43: MatplotlibDeprecationWarning: The get_cmap function was deprecated in Matplotlib 3.7 and will be removed in 3.11. Use ``matplotlib.colormaps[name]`` or ``matplotlib.colormaps.get_cmap()`` or ``pyplot.get_cmap()`` instead.
      cmap = cm.get_cmap('plasma')
    C:\Users\Scotts\AppData\Local\Temp\ipykernel_45816\3741891665.py:51: FutureWarning: 
    
    Passing `palette` without assigning `hue` is deprecated and will be removed in v0.14.0. Assign the `y` variable to `hue` and set `legend=False` for the same effect.
    
      plot = sns.catplot(data=df, x='SalePrice', y='MSSubClass', kind='boxen',  height=7, aspect=1.8, palette=palette, orient='h')
    


    
![png](Remame_First_notebook_files/Remame_First_notebook_18_1.png)
    



```python
counts
```




    MSSubClass
    20     536
    60     299
    50     144
    120     87
    30      69
    160     63
    70      60
    80      58
    90      52
    190     30
    85      20
    75      16
    45      12
    180     10
    40       4
    Name: count, dtype: int64



From the above it doesnt look like many of the categories hold a lot of information about the price, we see a slight shift higher for the means as we move from 1-1 1/2 stories to 2+ (20-50) and (60-75). But not extrememly significant.<BR>
The houses with the classification "Newer than 1946" (20 and 60) have significant variance (and generally higher means) in the prices compared to all the other categories.

*TODO:* 
<BR>We can try seperating by floor and producing another boxplot based off that.<BR>
Explore the dates made after 1946<BR>
See how if feature [HouseStyle](#housestyle) matching the number of floors in mssubclass's description, if it covers number of floors maybe we can then simplify this feature!


## **MSZoning:** 
***Description:*** Identifies the general zoning classification of the sale.
		
***Categories***:

       A	Agriculture
       C	Commercial
       FV	Floating Village Residential
       I	Industrial
       RH	Residential High Density
       RL	Residential Low Density
       RP	Residential Low Density Park 
       RM	Residential Medium Density
	

From the description and categories, we can understand roughly what this feature represents.  
`MSZoning` refers to the **zoning classification of the land**, not necessarily the building itself.

While categories like 'Commercial' or 'Industrial' exist, all properties in this dataset are residential (see `BldgType` for confirmation). So this likely means the house is located in a commercially-zoned **area**, not that it's a commercial **building**.

- Still worth checking how zoning type correlates with price  
- If some categories (like 'I' or 'FV') are very rare, we may group them into 'Other'  
- Can we compare this to `BldgType`? Same for Industrial, Agriculture, etc.  
- If some categories have very low counts, do we drop them or group into an 'Other'? Will they skew the data?  



**Nulls:** 0  
**Cardinality:** 8  
**Expected importance:** Medium  
**Feature Grouping:** Categorical, Geographical?, AreaType (or similar)?  
**Encoding:** Not 100% obvious — a candidate for Target-guided ordinal encoding  
**EDA ideas:**
- See how MSZoning varies with `Neighborhood`, `BldgType`

**Feature Engineering (optional, depending on EDA):**
- Group rare categories (e.g. C, I, A) into ‘NonResidential’ or ‘Other’
- Consider creating a new binary feature like `IsResidential`



```python
df['MSZoning'].isna().sum()
```




    0



[*back to feature list*](#feature-list)

## **LotFrontage:**
***Description:*** Linear feet of street connected to property
<BR>
How much of the front of the property (lot) connects to the street.<BR>
Higher value should correlate with higher sale price in general.<BR>
Some PUDs may enforce a certain LotFrontage size to keep property values high.
<BR><BR>
**Nulls:** 17.7%<BR>
**Null replacements:** null values could likely mean no lot frontage.

[*back to feature list*](#feature-list)



```python
df['LotFrontage'].isna().sum()/1460
```




    0.1773972602739726




## **LotArea:**
***Description:*** Lot size in square feet

[*back to feature list*](#feature-list)

## **Street:**
***Description:*** Type of road access to property

       Grvl	Gravel	
       Pave	Paved

[*back to feature list*](#feature-list)
       	
## **Alley:**
***Description:*** Type of alley access to property

       Grvl	Gravel
       Pave	Paved
       NA 	No alley access
		
[*back to feature list*](#feature-list)

## **LotShape:** 
***Description:*** General shape of property

       Reg	Regular	
       IR1	Slightly irregular
       IR2	Moderately Irregular
       IR3	Irregular

[*back to feature list*](#feature-list)
       
## **LandContour:** 
***Description:*** Flatness of the property

       Lvl	Near Flat/Level	
       Bnk	Banked - Quick and significant rise from street grade to building
       HLS	Hillside - Significant slope from side to side
       Low	Depression

[*back to feature list*](#feature-list)
		
## **Utilities:** 
***Description:*** Type of utilities available
		 
       AllPub	All public Utilities (E,G,W,& S)	
       NoSewr	Electricity, Gas, and Water (Septic Tank)
       NoSeWa	Electricity and Gas Only
       ELO	Electricity only	
[*back to feature list*](#feature-list)
	
## **LotConfig:** 
***Description:*** Lot configuration

       Inside	Inside lot
       Corner	Corner lot
       CulDSac	Cul-de-sac
       FR2	Frontage on 2 sides of property
       FR3	Frontage on 3 sides of property
[*back to feature list*](#feature-list)
	
## **LandSlope:** 
***Description:*** Slope of property
		
       Gtl	Gentle slope
       Mod	Moderate Slope	
       Sev	Severe Slope
[*back to feature list*](#feature-list)
	
## **Neighborhood:** 
***Description:*** Physical locations within Ames city limits

       Blmngtn	Bloomington Heights
       Blueste	Bluestem
       BrDale	Briardale
       BrkSide	Brookside
       ClearCr	Clear Creek
       CollgCr	College Creek
       Crawfor	Crawford
       Edwards	Edwards
       Gilbert	Gilbert
       IDOTRR	Iowa DOT and Rail Road
       MeadowV	Meadow Village
       Mitchel	Mitchell
       Names	North Ames
       NoRidge	Northridge
       NPkVill	Northpark Villa
       NridgHt	Northridge Heights
       NWAmes	Northwest Ames
       OldTown	Old Town
       SWISU	South & West of Iowa State University
       Sawyer	Sawyer
       SawyerW	Sawyer West
       Somerst	Somerset
       StoneBr	Stone Brook
       Timber	Timberland
       Veenker	Veenker
[*back to feature list*](#feature-list)
			
## **Condition1:** 
***Description:*** Proximity to various conditions
	
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
[*back to feature list*](#feature-list)
	
## **Condition2:** 
***Description:*** Proximity to various conditions (if more than one is present)
		
       Artery	Adjacent to arterial street
       Feedr	Adjacent to feeder street	
       Norm	Normal	
       RRNn	Within 200' of North-South Railroad
       RRAn	Adjacent to North-South Railroad
       PosN	Near positive off-site feature--park, greenbelt, etc.
       PosA	Adjacent to postive off-site feature
       RRNe	Within 200' of East-West Railroad
       RRAe	Adjacent to East-West Railroad
[*back to feature list*](#feature-list)
	
## **BldgType:** 
***Description:*** Type of dwelling
		
       1Fam	Single-family Detached	
       2FmCon	Two-family Conversion; originally built as one-family dwelling
       Duplx	Duplex
       TwnhsE	Townhouse End Unit
       TwnhsI	Townhouse Inside Unit
[*back to feature list*](#feature-list)
	
## **HouseStyle:** 
***Description:*** Style of dwelling
	
       1Story	One story
       1.5Fin	One and one-half story: 2nd level finished
       1.5Unf	One and one-half story: 2nd level unfinished
       2Story	Two story
       2.5Fin	Two and one-half story: 2nd level finished
       2.5Unf	Two and one-half story: 2nd level unfinished
       SFoyer	Split Foyer
       SLvl	Split Level
[*back to feature list*](#feature-list)
	
## **OverallQual:** 
***Description:*** Rates the overall material and finish of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average
       5	Average
       4	Below Average
       3	Fair
       2	Poor
       1	Very Poor
[*back to feature list*](#feature-list)
	
## **OverallCond:** 
***Description:*** Rates the overall condition of the house

       10	Very Excellent
       9	Excellent
       8	Very Good
       7	Good
       6	Above Average	
       5	Average
       4	Below Average	
       3	Fair
       2	Poor
       1	Very Poor
[*back to feature list*](#feature-list)
		
## **YearBuilt:** 
***Description:*** Original construction date
[*back to feature list*](#feature-list)

## **YearRemodAdd:** 
***Description:*** Remodel date (same as construction date if no remodeling or additions)
[*back to feature list*](#feature-list)

## **RoofStyle:** 
***Description:*** Type of roof

       Flat	Flat
       Gable	Gable
       Gambrel	Gabrel (Barn)
       Hip	Hip
       Mansard	Mansard
       Shed	Shed
[*back to feature list*](#feature-list)
		
## **RoofMatl:** 
***Description:*** Roof material

       ClyTile	Clay or Tile
       CompShg	Standard (Composite) Shingle
       Membran	Membrane
       Metal	Metal
       Roll	Roll
       Tar&Grv	Gravel & Tar
       WdShake	Wood Shakes
       WdShngl	Wood Shingles
[*back to feature list*](#feature-list)
		
## **Exterior1st:** 
***Description:*** Exterior covering on house

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast	
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
[*back to feature list*](#feature-list)
	
## **Exterior2nd:** 
***Description:*** Exterior covering on house (if more than one material)

       AsbShng	Asbestos Shingles
       AsphShn	Asphalt Shingles
       BrkComm	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       CemntBd	Cement Board
       HdBoard	Hard Board
       ImStucc	Imitation Stucco
       MetalSd	Metal Siding
       Other	Other
       Plywood	Plywood
       PreCast	PreCast
       Stone	Stone
       Stucco	Stucco
       VinylSd	Vinyl Siding
       Wd Sdng	Wood Siding
       WdShing	Wood Shingles
[*back to feature list*](#feature-list)
	
## **MasVnrType:** 
***Description:*** Masonry veneer type

       BrkCmn	Brick Common
       BrkFace	Brick Face
       CBlock	Cinder Block
       None	None
       Stone	Stone
[*back to feature list*](#feature-list)
	
## **MasVnrArea:** 
***Description:*** Masonry veneer area in square feet
[*back to feature list*](#feature-list)

## **ExterQual:** 
***Description:*** Evaluates the quality of the material on the exterior 
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
[*back to feature list*](#feature-list)
		
## **ExterCond:** 
***Description:*** Evaluates the present condition of the material on the exterior
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
[*back to feature list*](#feature-list)
		
## **Foundation:** 
***Description:*** Type of foundation
		
       BrkTil	Brick & Tile
       CBlock	Cinder Block
       PConc	Poured Contrete	
       Slab	Slab
       Stone	Stone
       Wood	Wood
[*back to feature list*](#feature-list)
		
## **BsmtQual:** 
***Description:*** Evaluates the height of the basement

       Ex	Excellent (100+ inches)	
       Gd	Good (90-99 inches)
       TA	Typical (80-89 inches)
       Fa	Fair (70-79 inches)
       Po	Poor (<70 inches
       NA	No Basement
[*back to feature list*](#feature-list)
		
## **BsmtCond:** 
***Description:*** Evaluates the general condition of the basement

       Ex	Excellent
       Gd	Good
       TA	Typical - slight dampness allowed
       Fa	Fair - dampness or some cracking or settling
       Po	Poor - Severe cracking, settling, or wetness
       NA	No Basement
[*back to feature list*](#feature-list)
	
## **BsmtExposure:** 
***Description:*** Refers to walkout or garden level walls

       Gd	Good Exposure
       Av	Average Exposure (split levels or foyers typically score average or above)	
       Mn	Mimimum Exposure
       No	No Exposure
       NA	No Basement
[*back to feature list*](#feature-list)
	
## **BsmtFinType1:** 
***Description:*** Rating of basement finished area

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
[*back to feature list*](#feature-list)
		
## **BsmtFinSF1:** 
***Description:*** Type 1 finished square feet
[*back to feature list*](#feature-list)

## **BsmtFinType2:** 
***Description:*** Rating of basement finished area (if multiple types)

       GLQ	Good Living Quarters
       ALQ	Average Living Quarters
       BLQ	Below Average Living Quarters	
       Rec	Average Rec Room
       LwQ	Low Quality
       Unf	Unfinshed
       NA	No Basement
[*back to feature list*](#feature-list)

## **BsmtFinSF2:** 
***Description:*** Type 2 finished square feet
[*back to feature list*](#feature-list)

## **BsmtUnfSF:** 
***Description:*** Unfinished square feet of basement area
[*back to feature list*](#feature-list)

## **TotalBsmtSF:** 
***Description:*** Total square feet of basement area
[*back to feature list*](#feature-list)

## **Heating:** 
***Description:*** Type of heating
		
       Floor	Floor Furnace
       GasA	Gas forced warm air furnace
       GasW	Gas hot water or steam heat
       Grav	Gravity furnace	
       OthW	Hot water or steam heat other than gas
       Wall	Wall furnace
[*back to feature list*](#feature-list)
		
## **HeatingQC:** 
***Description:*** Heating quality and condition

       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       Po	Poor
[*back to feature list*](#feature-list)
		
## **CentralAir:** 
***Description:*** Central air conditioning

       N	No
       Y	Yes
[*back to feature list*](#feature-list)
		
## **Electrical:** 
***Description:*** Electrical system

       SBrkr	Standard Circuit Breakers & Romex
       FuseA	Fuse Box over 60 AMP and all Romex wiring (Average)	
       FuseF	60 AMP Fuse Box and mostly Romex wiring (Fair)
       FuseP	60 AMP Fuse Box and mostly knob & tube wiring (poor)
       Mix	Mixed
[*back to feature list*](#feature-list)
		
## **1stFlrSF:** 
***Description:*** First Floor square feet
[*back to feature list*](#feature-list)
 
## **2ndFlrSF:** 
***Description:*** Second floor square feet
[*back to feature list*](#feature-list)

## **LowQualFinSF:** 
***Description:*** Low quality finished square feet (all floors)
[*back to feature list*](#feature-list)

## **GrLivArea:** 
***Description:*** Above grade (ground) living area square feet
[*back to feature list*](#feature-list)

## **BsmtFullBath:** 
***Description:*** Basement full bathrooms
[*back to feature list*](#feature-list)

## **BsmtHalfBath:** 
***Description:*** Basement half bathrooms
[*back to feature list*](#feature-list)

## **FullBath:** 
***Description:*** Full bathrooms above grade
[*back to feature list*](#feature-list)

## **HalfBath:** 
***Description:*** Half baths above grade
[*back to feature list*](#feature-list)

## **Bedroom:** 
***Description:*** Bedrooms above grade (does NOT include basement bedrooms)
[*back to feature list*](#feature-list)

## **Kitchen:** 
***Description:*** Kitchens above grade
[*back to feature list*](#feature-list)

## **KitchenQual:** 
***Description:*** Kitchen quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
[*back to feature list*](#feature-list)
       	
## **TotRmsAbvGrd:** 
***Description:*** Total rooms above grade (does not include bathrooms)
[*back to feature list*](#feature-list)

## **Functional:** 
***Description:*** Home functionality (Assume typical unless deductions are warranted)

       Typ	Typical Functionality
       Min1	Minor Deductions 1
       Min2	Minor Deductions 2
       Mod	Moderate Deductions
       Maj1	Major Deductions 1
       Maj2	Major Deductions 2
       Sev	Severely Damaged
       Sal	Salvage only
[*back to feature list*](#feature-list)
		
## **Fireplaces:** 
***Description:*** Number of fireplaces
[*back to feature list*](#feature-list)

## **FireplaceQu:** 
***Description:*** Fireplace quality

       Ex	Excellent - Exceptional Masonry Fireplace
       Gd	Good - Masonry Fireplace in main level
       TA	Average - Prefabricated Fireplace in main living area or Masonry Fireplace in basement
       Fa	Fair - Prefabricated Fireplace in basement
       Po	Poor - Ben Franklin Stove
       NA	No Fireplace
[*back to feature list*](#feature-list)
		
## **GarageType:** 
***Description:*** Garage location
		
       2Types	More than one type of garage
       Attchd	Attached to home
       Basment	Basement Garage
       BuiltIn	Built-In (Garage part of house - typically has room above garage)
       CarPort	Car Port
       Detchd	Detached from home
       NA	No Garage
[*back to feature list*](#feature-list)
		
## **GarageYrBlt:** 
***Description:*** Year garage was built
[*back to feature list*](#feature-list)
		
## **GarageFinish:** 
***Description:*** Interior finish of the garage

       Fin	Finished
       RFn	Rough Finished	
       Unf	Unfinished
       NA	No Garage
[*back to feature list*](#feature-list)
		
## **GarageCars:** 
***Description:*** Size of garage in car capacity
[*back to feature list*](#feature-list)

## **GarageArea:** 
***Description:*** Size of garage in square feet
[*back to feature list*](#feature-list)

## **GarageQual:** 
***Description:*** Garage quality

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
[*back to feature list*](#feature-list)
		
## **GarageCond:** 
***Description:*** Garage condition

       Ex	Excellent
       Gd	Good
       TA	Typical/Average
       Fa	Fair
       Po	Poor
       NA	No Garage
[*back to feature list*](#feature-list)
		
## **PavedDrive:** 
***Description:*** Paved driveway

       Y	Paved 
       P	Partial Pavement
       N	Dirt/Gravel
[*back to feature list*](#feature-list)
		
## **WoodDeckSF:** 
***Description:*** Wood deck area in square feet
[*back to feature list*](#feature-list)

## **OpenPorchSF:** 
***Description:*** Open porch area in square feet
[*back to feature list*](#feature-list)

## **EnclosedPorch:** 
***Description:*** Enclosed porch area in square feet
[*back to feature list*](#feature-list)

## **3SsnPorch:** 
***Description:*** Three season porch area in square feet
[*back to feature list*](#feature-list)

## **ScreenPorch:** 
***Description:*** Screen porch area in square feet
[*back to feature list*](#feature-list)

## **PoolArea:** 
***Description:*** Pool area in square feet
[*back to feature list*](#feature-list)

## **PoolQC:** 
***Description:*** Pool quality
		
       Ex	Excellent
       Gd	Good
       TA	Average/Typical
       Fa	Fair
       NA	No Pool
[*back to feature list*](#feature-list)
		
## **Fence:** 
***Description:*** Fence quality
		
       GdPrv	Good Privacy
       MnPrv	Minimum Privacy
       GdWo	Good Wood
       MnWw	Minimum Wood/Wire
       NA	No Fence

[*back to feature list*](#feature-list)
	
## **MiscFeature:** 
***Description:*** Miscellaneous feature not covered in other categories
		
       Elev	Elevator
       Gar2	2nd Garage (if not described in garage section)
       Othr	Other
       Shed	Shed (over 100 SF)
       TenC	Tennis Court
       NA	None
[*back to feature list*](#feature-list)
		
## **MiscVal:** 
***Description:*** $Value of miscellaneous feature
[*back to feature list*](#feature-list)

## **MoSold:** 
***Description:*** Month Sold (MM)
[*back to feature list*](#feature-list)

## **YrSold:** 
***Description:*** Year Sold (YYYY)
[*back to feature list*](#feature-list)

## **SaleType:** 
***Description:*** Type of sale
		
       WD 	Warranty Deed - Conventional
       CWD	Warranty Deed - Cash
       VWD	Warranty Deed - VA Loan
       New	Home just constructed and sold
       COD	Court Officer Deed/Estate
       Con	Contract 15% Down payment regular terms
       ConLw	Contract Low Down payment and low interest
       ConLI	Contract Low Interest
       ConLD	Contract Low Down
       Oth	Other
[*back to feature list*](#feature-list)
		
## **SaleCondition:** 
***Description:*** Condition of sale

       Normal	Normal Sale
       Abnorml	Abnormal Sale -  trade, foreclosure, short sale
       AdjLand	Adjoining Land Purchase
       Alloca	Allocation - two linked properties with separate deeds, typically condo with a garage unit	
       Family	Sale between family members
       Partial	Home was not completed when last assessed (associated with New Homes)



#



