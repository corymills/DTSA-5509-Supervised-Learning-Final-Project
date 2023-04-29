# DTSA 5509 Supervised Learning Final Project
## Decision Tree Binary Classification Tasks Applied to Food Access Research Atlas
### Project Topic
Food Deserts are areas where there is low access to supermarkets and other sources of healthy foods. They are often reported as occuring in communities that are already underserved, low income, and in Black communities. Access to food that is affordable and healthy is critical to support the health and wellbeing of a community, and areas that are struggling with poverty and inequality often also struggle with basic access to healthy food.

Understanding the circumstances that create food deserts is critical in being able to identify and address the root causes of the problem. If we can model this, we can make connections in the data we may not have seen otherwise.  We can also use this to predict outcomes as communities change.

Using socio-economic data from the Department of Agriculture's Food Access Research Atlas, I explored three decision tree binary classification models to attempt to classify whether a community is likely to have low access to healthy food sources.

**For more information**

Annie E. Casey Foundation: Food Deserts in the United States: https://www.aecf.org/blog/exploring-americas-food-deserts

Wikipedia: Food Desert: https://en.wikipedia.org/wiki/Food_desert

### Imports


```python
import pandas as pd
import altair as alt
import dtale
from sklearn.svm import SVC
from xgboost import XGBClassifier
from xgboost import DMatrix
from xgboost import plot_importance
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from imblearn.under_sampling import RandomUnderSampler
from scipy.stats import ttest_ind
alt.data_transformers.disable_max_rows()
```




    DataTransformerRegistry.enable('default')



### Data
The data used in this report is the Food Access Research Atlas, published by the Economic Research Service, Department of Agriculture, via data.gov, a public source of data made available by the United States government.

This dataset has 72,531 records, one for each U.S. Census tract.  The dataset includes 147 features for each tract, describing the tract, the population of the tract, demographic breakdowns of each tract (population Black, Seniors, Children, etc), and numerous measures on the food access for each tract.

For my task, I wanted to focus on how the socio-economic factors contribututed to whether a tract was likely to be classified as low access to food.  I limited the features used to the following:

| Field              | Description                                                                                                 |
|:-------------------|:------------------------------------------------------------------------------------------------------------|
| LILATracts_1And10  | **`Label`** Flag for low-income and low access when considering low accessibilty at 1 and 10 miles                      |
| State              | **`Feature`** State name                                                                                                  |
| Urban              | **`Feature`** Flag for urban tract                                                                                        |
| OHU2010            | **`Feature`** Occupied housing unit count from 2010 census                                                                |
| PovertyRate        | **`Feature`** Share of the tract population living with income at or below the Federal poverty thresholds for family size |
| MedianFamilyIncome | **`Feature`** Tract median family income                                                                                  |
| TractLOWI          | **`Feature`** Total count of low-income population in tract                                                               |
| TractKids          | **`Feature`** Total count of children age 0-17 in tract                                                                   |
| TractSeniors       | **`Feature`** Total count of seniors age 65+ in tract                                                                     |
| TractWhite         | **`Feature`** Total count of White population in tract                                                                    |
| TractBlack         | **`Feature`** Total count of Black or African American population in tract                                                |
| TractAsian         | **`Feature`** Total count of Asian population in tract                                                                    |
| TractNHOPI         | **`Feature`** Total count of Native Hawaiian and Other Pacific Islander population in tract                               |
| TractAIAN          | **`Feature`** Total count of American Indian and Alaska Native population in tract                                        |
| TractOMultir       | **`Feature`** Total count of Other/Multiple race population in tract                                                      |
| TractHispanic      | **`Feature`** Total count of Hispanic or Latino population in tract                                                       |
| TractHUNV          | **`Feature`** Total count of housing units without a vehicle in tract                                                     |
| TractSNAP          | **`Feature`** Total count of housing units receiving SNAP benefits in tract                                               |

Field `LILATracts_1And10` is the label I'm attempting to predict.  The remaining fields are features.


Data: https://catalog.data.gov/dataset/food-access-research-atlas

About the Atlas: https://www.ers.usda.gov/data-products/food-access-research-atlas/about-the-atlas.aspx

In addition, I'm using the U.S. Census Divisions listed at the following URL to perform further feature engineering to supplement my dataset.

https://www.ncei.noaa.gov/access/monitoring/reference-maps/us-census-divisions

**Data Citations**

Economic Research Service, Department of Agriculture (2021, February 24). Data.Gov. Food Access Research Atlas. Retrieved April 23, 2023, from https://catalog.data.gov/dataset/food-access-research-atlas


```python
label = 'LILATracts_1And10'
```


```python
# Create variable table and copy to clipboard for above
used_columns = [label,'Pop2010','State','OHU2010','Urban','PovertyRate','MedianFamilyIncome','TractBlack','TractLOWI','TractKids','TractSeniors','TractWhite','TractAsian','TractNHOPI','TractAIAN','TractOMultir','TractHispanic','TractHUNV','TractSNAP']
var_df = pd.read_excel('../data/FoodAccessResearchAtlasData2019.xlsx', sheet_name='Variable Lookup')
var_df = var_df[var_df['Field'].isin(used_columns)][['Field','Description']]
#pd.io.clipboards.to_clipboard(var_df.to_markdown(index=False), excel=False)
```

### Data Cleaning

As outlined above, I've limited the features used to the most relevant features for the classification task I wanted to perform.  Of the 147 features, 18 are relevant for the task of classifying the low food access status based on the demographic makeup of an area.  Many of the other features in the original file describe in detail the low food access for specific populations, so have been left out since they would be auto-correlative with the low access label I'm attempting to predict.

In addition, while reviewing the number of NA values present in the data, I found that `MedianFamilyIncome` had a high count of NAs. I have dropped this feature from the dataset for this reason.

`OHU2010` has values of 0 for 106 rows. This appears to be null data, coded as 0.  As this will cause a division by 0 error when calculating percentages below, I'm also dropping these rows.

The remaining features have a low number of NA values.  For these, I have dropped the rows that contained NA values.

In the original dataset, the demographic data is reported as total counts of residents in that demographic. Because I want to compare tracts, and each tract may have a different total population, it makes most sense to convert these to percentages.  Below, I calculate the percentage of residents in each demographic using the demographic count and the total population count `Pop2010`.  After I've done these calculations, I drop the original count columns as they are no longer needed.

I also investigated whether the labels in the dataset were balanced.  I've found that they are not, and I rebalance these later while building the model.

#### Read Data
Load the data into a dataframe


```python
# Read Data
df = pd.read_excel('../data/FoodAccessResearchAtlasData2019.xlsx', sheet_name='Food Access Research Atlas')
df = df[used_columns]
```

#### Data Types Check
Reviewing the dtypes listed below, these are as I would expect them to be.  I do not believe any data type munging is necessary for this dataset.


```python
df.dtypes
```




    LILATracts_1And10       int64
    Pop2010                 int64
    State                  object
    OHU2010                 int64
    Urban                   int64
    PovertyRate           float64
    MedianFamilyIncome    float64
    TractBlack            float64
    TractLOWI             float64
    TractKids             float64
    TractSeniors          float64
    TractWhite            float64
    TractAsian            float64
    TractNHOPI            float64
    TractAIAN             float64
    TractOMultir          float64
    TractHispanic         float64
    TractHUNV             float64
    TractSNAP             float64
    dtype: object



#### Check for NAs and Vizualize
Check for explicit nas in the dataset


```python
# Check for NAs
na_counts = pd.DataFrame(df.isna().sum()).reset_index()
na_counts.columns = ['Field', 'NA Count']
na_counts = na_counts.sort_values('NA Count', ascending=False)

# Visualize NAs
base = alt.Chart(na_counts).encode(
    x=alt.X('Field:N', sort='-y'),
    y='NA Count',
    text = 'NA Count'
).properties(width=800)
base.mark_bar() + base.mark_text(align='center', dy=-10)
```





<div id="altair-viz-004ae5c787e4463bb47a213097fdce99"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-004ae5c787e4463bb47a213097fdce99") {
      outputDiv = document.getElementById("altair-viz-004ae5c787e4463bb47a213097fdce99");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": "bar", "encoding": {"text": {"field": "NA Count", "type": "quantitative"}, "x": {"field": "Field", "sort": "-y", "type": "nominal"}, "y": {"field": "NA Count", "type": "quantitative"}}, "width": 800}, {"mark": {"type": "text", "align": "center", "dy": -10}, "encoding": {"text": {"field": "NA Count", "type": "quantitative"}, "x": {"field": "Field", "sort": "-y", "type": "nominal"}, "y": {"field": "NA Count", "type": "quantitative"}}, "width": 800}], "data": {"name": "data-ade0ae590ca65299e43816310959829a"}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-ade0ae590ca65299e43816310959829a": [{"Field": "MedianFamilyIncome", "NA Count": 748}, {"Field": "TractKids", "NA Count": 4}, {"Field": "TractSeniors", "NA Count": 4}, {"Field": "TractHUNV", "NA Count": 4}, {"Field": "TractHispanic", "NA Count": 4}, {"Field": "TractOMultir", "NA Count": 4}, {"Field": "TractAIAN", "NA Count": 4}, {"Field": "TractNHOPI", "NA Count": 4}, {"Field": "TractAsian", "NA Count": 4}, {"Field": "TractWhite", "NA Count": 4}, {"Field": "TractSNAP", "NA Count": 4}, {"Field": "TractLOWI", "NA Count": 4}, {"Field": "TractBlack", "NA Count": 4}, {"Field": "PovertyRate", "NA Count": 3}, {"Field": "Pop2010", "NA Count": 0}, {"Field": "Urban", "NA Count": 0}, {"Field": "OHU2010", "NA Count": 0}, {"Field": "State", "NA Count": 0}, {"Field": "LILATracts_1And10", "NA Count": 0}]}}, {"mode": "vega-lite"});
</script>



#### Drop NAs
Dropping explicit NAs from the dataset


```python
len_df_bef = len(df)

# Drop high NA column Median Family Income
df = df.drop('MedianFamilyIncome', axis=1)

# Drop records with NAs in values
df = df.dropna()
print(str(len_df_bef - len(df)) + ' rows dropped')

# Check for no remaining NAs
print('Remaining NAs:')
display(df.isna().sum())
```

    4 rows dropped
    Remaining NAs:



    LILATracts_1And10    0
    Pop2010              0
    State                0
    OHU2010              0
    Urban                0
    PovertyRate          0
    TractBlack           0
    TractLOWI            0
    TractKids            0
    TractSeniors         0
    TractWhite           0
    TractAsian           0
    TractNHOPI           0
    TractAIAN            0
    TractOMultir         0
    TractHispanic        0
    TractHUNV            0
    TractSNAP            0
    dtype: int64


#### Drop 0-coded NAs
Dropping missing data that is coded as 0 in the dataset


```python
len_df_bef = len(df)

# Check count of 0 values where we expect > 0
print('Count of 0 values for Pop2010: {}'.format(sum(df['Pop2010'] == 0)))
print('Count of 0 values for OHU2010: {}'.format(sum(df['OHU2010'] == 0)))

# Keep only rows where values > 0
df = df[df['Pop2010'] > 0]
df = df[df['OHU2010'] > 0]
print(str(len_df_bef - len(df)) + ' rows dropped')

# Check our work
print('Count of 0 values for Pop2010: {}'.format(sum(df['Pop2010'] == 0)))
print('Count of 0 values for OHU2010: {}'.format(sum(df['OHU2010'] == 0)))
```

    Count of 0 values for Pop2010: 0
    Count of 0 values for OHU2010: 106
    106 rows dropped
    Count of 0 values for Pop2010: 0
    Count of 0 values for OHU2010: 0


#### Check for Label Imbalance
Checking for label imbalance here. There is a clear imbalance in the data.  We'll account for the label imbalance later, when building our model.


```python
# Check for imbalaced labels
val_counts = pd.DataFrame(df[label].value_counts()).reset_index()
base = alt.Chart(val_counts).encode(
    x=alt.X(label + ':N', sort='-y'),
    y='count',
    text = 'count'
).properties(width=100)
base.mark_bar(size=30) + base.mark_text(align='center', dy=-10)
```





<div id="altair-viz-311071f151e148c58a5a8c74650436c7"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-311071f151e148c58a5a8c74650436c7") {
      outputDiv = document.getElementById("altair-viz-311071f151e148c58a5a8c74650436c7");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": {"type": "bar", "size": 30}, "encoding": {"text": {"field": "count", "type": "quantitative"}, "x": {"field": "LILATracts_1And10", "sort": "-y", "type": "nominal"}, "y": {"field": "count", "type": "quantitative"}}, "width": 100}, {"mark": {"type": "text", "align": "center", "dy": -10}, "encoding": {"text": {"field": "count", "type": "quantitative"}, "x": {"field": "LILATracts_1And10", "sort": "-y", "type": "nominal"}, "y": {"field": "count", "type": "quantitative"}}, "width": 100}], "data": {"name": "data-7f6d7f452a6740ddaa5f4d663ad7acfc"}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-7f6d7f452a6740ddaa5f4d663ad7acfc": [{"LILATracts_1And10": 0, "count": 63132}, {"LILATracts_1And10": 1, "count": 9289}]}}, {"mode": "vega-lite"});
</script>



### Feature Engineering

#### Calculate Percentage Share for Each Demographic
In order to compare tract to tract, where total populations may differ, I calculated the percentage makeup of each population and stored it back into my dataset.  These engineered features will make up a majority of the features used in the model.


```python
# Calculate percentage share for each demographic population
df['BlackPopShare'] = df['TractBlack'] / df['Pop2010']
df['LOWIPopShare'] = df['TractLOWI'] / df['Pop2010']
df['KidsPopShare'] = df['TractKids'] / df['Pop2010']
df['SeniorsPopShare'] = df['TractSeniors'] / df['Pop2010']
df['WhitePopShare'] = df['TractWhite'] / df['Pop2010']
df['AsianPopShare'] = df['TractAsian'] / df['Pop2010']
df['NHOPIPopShare'] = df['TractNHOPI'] / df['Pop2010']
df['AIANPopShare'] = df['TractAIAN'] / df['Pop2010']
df['OMultirPopShare'] = df['TractOMultir'] / df['Pop2010']
df['HispanicPopShare'] = df['TractHispanic'] / df['Pop2010']
df['HUNVPopShare'] = df['TractHUNV'] / df['OHU2010']
df['SNAPPopShare'] = df['TractSNAP'] / df['OHU2010']

# Drop the original features, since we don't need them any more
df = df.drop(['Pop2010','OHU2010','TractBlack','TractLOWI','TractKids','TractSeniors','TractWhite','TractAsian','TractNHOPI','TractAIAN','TractOMultir','TractHispanic','TractHUNV','TractSNAP'], axis=1)
```

#### Add U.S Census Divisions to Data
In Exploratory Data Analysis below, I found evidence that certain regions of the U.S. may have more low income, low access trackts than others.  Here, I map state to regions, based on the assigned region names from the U.S. Census Regions mapping described in the Data section above.


```python
div_map = {
    'Illinois':'East North Central',
    'Indiana':'East North Central',
    'Michigan':'East North Central',
    'Ohio':'East North Central',
    'Wisconsin':'East North Central',
    'Alabama':'East South Central',
    'Kentucky':'East South Central',
    'Mississippi':'East South Central',
    'Tennessee':'East South Central',
    'New Jersey':'Middle Atlantic',
    'New York':'Middle Atlantic',
    'Pennsylvania':'Middle Atlantic',
    'Arizona':'Mountain',
    'Colorado':'Mountain',
    'Idaho':'Mountain',
    'Montana':'Mountain',
    'New Mexico':'Mountain',
    'Nevada':'Mountain',
    'Utah':'Mountain',
    'Wyoming':'Mountain',
    'Connecticut':'New England',
    'Maine':'New England',
    'Massachusetts':'New England',
    'New Hampshire':'New England',
    'Rhode Island':'New England',
    'Vermont':'New England',
    'California':'Pacific',
    'Oregon':'Pacific',
    'Washington':'Pacific',
    'Delaware':'South Atlantic',
    'Florida':'South Atlantic',
    'Georgia':'South Atlantic',
    'Maryland':'South Atlantic',
    'North Carolina':'South Atlantic',
    'South Carolina':'South Atlantic',
    'Virginia':'South Atlantic',
    'West Virginia':'South Atlantic',
    'Iowa':'West North Central',
    'Kansas':'West North Central',
    'Minnesota':'West North Central',
    'Missouri':'West North Central',
    'Nebraska':'West North Central',
    'North Dakota':'West North Central',
    'South Dakota':'West North Central',
    'Arkansas':'West South Central',
    'Louisiana':'West South Central',
    'Oklahoma':'West South Central',
    'Texas':'West South Central'
}
df['Census Division'] = df['State'].map(div_map)
```

### Exploratory Data Analysis

#### Correlation
Calculate and visualize a correlation matrix.

From the correlation matrix below, we can see a few interesting correlations:
- BlackPopShare is significantly negatively correlated with WhitePopShare. This specific strong correlation may point to these population numbers being good features for a binary classification model, with some more analysis on the correlation to to our label to follow.
- In addition, there is a negative correlation between WhitePopShare, LOWIPopShare (Low Income), PovertyRate, and SNAPPopShare.  There is a positive correlation between BlackPopShare and these same features. Since we suspect that low access, low income tracts may correlate to communities with higher poverty rate, etc, these correlations seem to suggest that they may be good indicators to identify feed desert communities.
- It's interesting that features like KidsPopShare appear to be positively correlated with PovertyRate and SNAPPopShare.  I would not have guessed this going into this analysis, but this may also be a good indicator for my models. 


```python
# Calculate Correlations
eda_df = df.copy()
eda_df = eda_df.drop(['State', 'Census Division', 'LILATracts_1And10', 'Urban'], axis=1)
cor_data = (eda_df.corr().stack().reset_index() 
              .rename(columns={0: 'correlation', 'level_0': 'variable', 'level_1': 'variable2'}))
cor_data['correlation_label'] = cor_data['correlation'].map('{:.2f}'.format)
```


```python
# Vizualize correlations
base = alt.Chart(cor_data).encode(
    x='variable2:O',
    y='variable:O'    
)

text = base.mark_text().encode(
    text='correlation_label',
    color=alt.condition(
        alt.datum.correlation > 0.5, 
        alt.value('white'),
        alt.value('black')
    )
)

cor_plot = base.mark_rect().encode(
    color=alt.Color('correlation:Q', scale=alt.Scale(scheme='redblue'))
).properties(width=600, height=600)

cor_plot + text
```





<div id="altair-viz-ecd38ad93bbe418087ed3f4527d5f9b4"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-ecd38ad93bbe418087ed3f4527d5f9b4") {
      outputDiv = document.getElementById("altair-viz-ecd38ad93bbe418087ed3f4527d5f9b4");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}}, "layer": [{"mark": "rect", "encoding": {"color": {"field": "correlation", "scale": {"scheme": "redblue"}, "type": "quantitative"}, "x": {"field": "variable2", "type": "ordinal"}, "y": {"field": "variable", "type": "ordinal"}}, "height": 600, "width": 600}, {"mark": "text", "encoding": {"color": {"condition": {"value": "white", "test": "(datum.correlation > 0.5)"}, "value": "black"}, "text": {"field": "correlation_label", "type": "nominal"}, "x": {"field": "variable2", "type": "ordinal"}, "y": {"field": "variable", "type": "ordinal"}}}], "data": {"name": "data-7263edee01085b18be1d5221125d5c00"}, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-7263edee01085b18be1d5221125d5c00": [{"variable": "PovertyRate", "variable2": "PovertyRate", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "PovertyRate", "variable2": "BlackPopShare", "correlation": 0.4425977125573052, "correlation_label": "0.44"}, {"variable": "PovertyRate", "variable2": "LOWIPopShare", "correlation": 0.6090738295981086, "correlation_label": "0.61"}, {"variable": "PovertyRate", "variable2": "KidsPopShare", "correlation": 0.11872078425905126, "correlation_label": "0.12"}, {"variable": "PovertyRate", "variable2": "SeniorsPopShare", "correlation": -0.1935208978945761, "correlation_label": "-0.19"}, {"variable": "PovertyRate", "variable2": "WhitePopShare", "correlation": -0.473037642317788, "correlation_label": "-0.47"}, {"variable": "PovertyRate", "variable2": "AsianPopShare", "correlation": -0.10633494054874681, "correlation_label": "-0.11"}, {"variable": "PovertyRate", "variable2": "NHOPIPopShare", "correlation": -0.002953677798931915, "correlation_label": "-0.00"}, {"variable": "PovertyRate", "variable2": "AIANPopShare", "correlation": 0.11312832669257476, "correlation_label": "0.11"}, {"variable": "PovertyRate", "variable2": "OMultirPopShare", "correlation": 0.26503630022370694, "correlation_label": "0.27"}, {"variable": "PovertyRate", "variable2": "HispanicPopShare", "correlation": 0.2447115298191319, "correlation_label": "0.24"}, {"variable": "PovertyRate", "variable2": "HUNVPopShare", "correlation": 0.2829084983393503, "correlation_label": "0.28"}, {"variable": "PovertyRate", "variable2": "SNAPPopShare", "correlation": 0.7060501024305214, "correlation_label": "0.71"}, {"variable": "BlackPopShare", "variable2": "PovertyRate", "correlation": 0.4425977125573052, "correlation_label": "0.44"}, {"variable": "BlackPopShare", "variable2": "BlackPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "BlackPopShare", "variable2": "LOWIPopShare", "correlation": 0.2910165294600782, "correlation_label": "0.29"}, {"variable": "BlackPopShare", "variable2": "KidsPopShare", "correlation": 0.14885321115335012, "correlation_label": "0.15"}, {"variable": "BlackPopShare", "variable2": "SeniorsPopShare", "correlation": -0.1627552647443349, "correlation_label": "-0.16"}, {"variable": "BlackPopShare", "variable2": "WhitePopShare", "correlation": -0.8100886338141112, "correlation_label": "-0.81"}, {"variable": "BlackPopShare", "variable2": "AsianPopShare", "correlation": -0.10894940540121263, "correlation_label": "-0.11"}, {"variable": "BlackPopShare", "variable2": "NHOPIPopShare", "correlation": -0.03807134231852622, "correlation_label": "-0.04"}, {"variable": "BlackPopShare", "variable2": "AIANPopShare", "correlation": -0.05773753425334272, "correlation_label": "-0.06"}, {"variable": "BlackPopShare", "variable2": "OMultirPopShare", "correlation": -0.024183082909660578, "correlation_label": "-0.02"}, {"variable": "BlackPopShare", "variable2": "HispanicPopShare", "correlation": -0.07347609624427635, "correlation_label": "-0.07"}, {"variable": "BlackPopShare", "variable2": "HUNVPopShare", "correlation": 0.23872163013090897, "correlation_label": "0.24"}, {"variable": "BlackPopShare", "variable2": "SNAPPopShare", "correlation": 0.4704181429214918, "correlation_label": "0.47"}, {"variable": "LOWIPopShare", "variable2": "PovertyRate", "correlation": 0.6090738295981086, "correlation_label": "0.61"}, {"variable": "LOWIPopShare", "variable2": "BlackPopShare", "correlation": 0.2910165294600782, "correlation_label": "0.29"}, {"variable": "LOWIPopShare", "variable2": "LOWIPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "LOWIPopShare", "variable2": "KidsPopShare", "correlation": 0.1786324309365767, "correlation_label": "0.18"}, {"variable": "LOWIPopShare", "variable2": "SeniorsPopShare", "correlation": -0.12798279905141066, "correlation_label": "-0.13"}, {"variable": "LOWIPopShare", "variable2": "WhitePopShare", "correlation": -0.3455185352389485, "correlation_label": "-0.35"}, {"variable": "LOWIPopShare", "variable2": "AsianPopShare", "correlation": -0.1049347616018606, "correlation_label": "-0.10"}, {"variable": "LOWIPopShare", "variable2": "NHOPIPopShare", "correlation": 0.034591646146519446, "correlation_label": "0.03"}, {"variable": "LOWIPopShare", "variable2": "AIANPopShare", "correlation": 0.08869664714423041, "correlation_label": "0.09"}, {"variable": "LOWIPopShare", "variable2": "OMultirPopShare", "correlation": 0.2815892149431609, "correlation_label": "0.28"}, {"variable": "LOWIPopShare", "variable2": "HispanicPopShare", "correlation": 0.2782108970331435, "correlation_label": "0.28"}, {"variable": "LOWIPopShare", "variable2": "HUNVPopShare", "correlation": 0.2561147097146154, "correlation_label": "0.26"}, {"variable": "LOWIPopShare", "variable2": "SNAPPopShare", "correlation": 0.6046305687159963, "correlation_label": "0.60"}, {"variable": "KidsPopShare", "variable2": "PovertyRate", "correlation": 0.11872078425905126, "correlation_label": "0.12"}, {"variable": "KidsPopShare", "variable2": "BlackPopShare", "correlation": 0.14885321115335012, "correlation_label": "0.15"}, {"variable": "KidsPopShare", "variable2": "LOWIPopShare", "correlation": 0.1786324309365767, "correlation_label": "0.18"}, {"variable": "KidsPopShare", "variable2": "KidsPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "KidsPopShare", "variable2": "SeniorsPopShare", "correlation": -0.4697129829052088, "correlation_label": "-0.47"}, {"variable": "KidsPopShare", "variable2": "WhitePopShare", "correlation": -0.25198637554890047, "correlation_label": "-0.25"}, {"variable": "KidsPopShare", "variable2": "AsianPopShare", "correlation": -0.08725519060714214, "correlation_label": "-0.09"}, {"variable": "KidsPopShare", "variable2": "NHOPIPopShare", "correlation": 0.032064618157964984, "correlation_label": "0.03"}, {"variable": "KidsPopShare", "variable2": "AIANPopShare", "correlation": 0.09355020223263774, "correlation_label": "0.09"}, {"variable": "KidsPopShare", "variable2": "OMultirPopShare", "correlation": 0.3409743847501985, "correlation_label": "0.34"}, {"variable": "KidsPopShare", "variable2": "HispanicPopShare", "correlation": 0.3296765739817506, "correlation_label": "0.33"}, {"variable": "KidsPopShare", "variable2": "HUNVPopShare", "correlation": -0.05273259186214481, "correlation_label": "-0.05"}, {"variable": "KidsPopShare", "variable2": "SNAPPopShare", "correlation": 0.2809451289140096, "correlation_label": "0.28"}, {"variable": "SeniorsPopShare", "variable2": "PovertyRate", "correlation": -0.1935208978945761, "correlation_label": "-0.19"}, {"variable": "SeniorsPopShare", "variable2": "BlackPopShare", "correlation": -0.1627552647443349, "correlation_label": "-0.16"}, {"variable": "SeniorsPopShare", "variable2": "LOWIPopShare", "correlation": -0.12798279905141066, "correlation_label": "-0.13"}, {"variable": "SeniorsPopShare", "variable2": "KidsPopShare", "correlation": -0.4697129829052088, "correlation_label": "-0.47"}, {"variable": "SeniorsPopShare", "variable2": "SeniorsPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "SeniorsPopShare", "variable2": "WhitePopShare", "correlation": 0.3208260157613004, "correlation_label": "0.32"}, {"variable": "SeniorsPopShare", "variable2": "AsianPopShare", "correlation": -0.11596188136092349, "correlation_label": "-0.12"}, {"variable": "SeniorsPopShare", "variable2": "NHOPIPopShare", "correlation": -0.04562005429605059, "correlation_label": "-0.05"}, {"variable": "SeniorsPopShare", "variable2": "AIANPopShare", "correlation": -0.048014584743454804, "correlation_label": "-0.05"}, {"variable": "SeniorsPopShare", "variable2": "OMultirPopShare", "correlation": -0.3361408222052703, "correlation_label": "-0.34"}, {"variable": "SeniorsPopShare", "variable2": "HispanicPopShare", "correlation": -0.2864472738585831, "correlation_label": "-0.29"}, {"variable": "SeniorsPopShare", "variable2": "HUNVPopShare", "correlation": -0.05514190726297414, "correlation_label": "-0.06"}, {"variable": "SeniorsPopShare", "variable2": "SNAPPopShare", "correlation": -0.1545011100894765, "correlation_label": "-0.15"}, {"variable": "WhitePopShare", "variable2": "PovertyRate", "correlation": -0.473037642317788, "correlation_label": "-0.47"}, {"variable": "WhitePopShare", "variable2": "BlackPopShare", "correlation": -0.8100886338141112, "correlation_label": "-0.81"}, {"variable": "WhitePopShare", "variable2": "LOWIPopShare", "correlation": -0.3455185352389485, "correlation_label": "-0.35"}, {"variable": "WhitePopShare", "variable2": "KidsPopShare", "correlation": -0.25198637554890047, "correlation_label": "-0.25"}, {"variable": "WhitePopShare", "variable2": "SeniorsPopShare", "correlation": 0.3208260157613004, "correlation_label": "0.32"}, {"variable": "WhitePopShare", "variable2": "WhitePopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "WhitePopShare", "variable2": "AsianPopShare", "correlation": -0.2962010445071246, "correlation_label": "-0.30"}, {"variable": "WhitePopShare", "variable2": "NHOPIPopShare", "correlation": -0.13208338174022527, "correlation_label": "-0.13"}, {"variable": "WhitePopShare", "variable2": "AIANPopShare", "correlation": -0.13563654702432712, "correlation_label": "-0.14"}, {"variable": "WhitePopShare", "variable2": "OMultirPopShare", "correlation": -0.4373200625070667, "correlation_label": "-0.44"}, {"variable": "WhitePopShare", "variable2": "HispanicPopShare", "correlation": -0.3033886391343884, "correlation_label": "-0.30"}, {"variable": "WhitePopShare", "variable2": "HUNVPopShare", "correlation": -0.27505280573810764, "correlation_label": "-0.28"}, {"variable": "WhitePopShare", "variable2": "SNAPPopShare", "correlation": -0.4804250695653086, "correlation_label": "-0.48"}, {"variable": "AsianPopShare", "variable2": "PovertyRate", "correlation": -0.10633494054874681, "correlation_label": "-0.11"}, {"variable": "AsianPopShare", "variable2": "BlackPopShare", "correlation": -0.10894940540121263, "correlation_label": "-0.11"}, {"variable": "AsianPopShare", "variable2": "LOWIPopShare", "correlation": -0.1049347616018606, "correlation_label": "-0.10"}, {"variable": "AsianPopShare", "variable2": "KidsPopShare", "correlation": -0.08725519060714214, "correlation_label": "-0.09"}, {"variable": "AsianPopShare", "variable2": "SeniorsPopShare", "correlation": -0.11596188136092349, "correlation_label": "-0.12"}, {"variable": "AsianPopShare", "variable2": "WhitePopShare", "correlation": -0.2962010445071246, "correlation_label": "-0.30"}, {"variable": "AsianPopShare", "variable2": "AsianPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "AsianPopShare", "variable2": "NHOPIPopShare", "correlation": 0.19990148617905207, "correlation_label": "0.20"}, {"variable": "AsianPopShare", "variable2": "AIANPopShare", "correlation": -0.04448442329308261, "correlation_label": "-0.04"}, {"variable": "AsianPopShare", "variable2": "OMultirPopShare", "correlation": 0.16387753106144903, "correlation_label": "0.16"}, {"variable": "AsianPopShare", "variable2": "HispanicPopShare", "correlation": 0.07439188042340097, "correlation_label": "0.07"}, {"variable": "AsianPopShare", "variable2": "HUNVPopShare", "correlation": 0.05402301408683258, "correlation_label": "0.05"}, {"variable": "AsianPopShare", "variable2": "SNAPPopShare", "correlation": -0.14834678386726421, "correlation_label": "-0.15"}, {"variable": "NHOPIPopShare", "variable2": "PovertyRate", "correlation": -0.002953677798931915, "correlation_label": "-0.00"}, {"variable": "NHOPIPopShare", "variable2": "BlackPopShare", "correlation": -0.03807134231852622, "correlation_label": "-0.04"}, {"variable": "NHOPIPopShare", "variable2": "LOWIPopShare", "correlation": 0.034591646146519446, "correlation_label": "0.03"}, {"variable": "NHOPIPopShare", "variable2": "KidsPopShare", "correlation": 0.032064618157964984, "correlation_label": "0.03"}, {"variable": "NHOPIPopShare", "variable2": "SeniorsPopShare", "correlation": -0.04562005429605059, "correlation_label": "-0.05"}, {"variable": "NHOPIPopShare", "variable2": "WhitePopShare", "correlation": -0.13208338174022527, "correlation_label": "-0.13"}, {"variable": "NHOPIPopShare", "variable2": "AsianPopShare", "correlation": 0.19990148617905207, "correlation_label": "0.20"}, {"variable": "NHOPIPopShare", "variable2": "NHOPIPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "NHOPIPopShare", "variable2": "AIANPopShare", "correlation": 0.0003911724550259673, "correlation_label": "0.00"}, {"variable": "NHOPIPopShare", "variable2": "OMultirPopShare", "correlation": 0.15497754497224042, "correlation_label": "0.15"}, {"variable": "NHOPIPopShare", "variable2": "HispanicPopShare", "correlation": 0.03916764532731688, "correlation_label": "0.04"}, {"variable": "NHOPIPopShare", "variable2": "HUNVPopShare", "correlation": 0.009043128632616073, "correlation_label": "0.01"}, {"variable": "NHOPIPopShare", "variable2": "SNAPPopShare", "correlation": 0.022474545009895625, "correlation_label": "0.02"}, {"variable": "AIANPopShare", "variable2": "PovertyRate", "correlation": 0.11312832669257476, "correlation_label": "0.11"}, {"variable": "AIANPopShare", "variable2": "BlackPopShare", "correlation": -0.05773753425334272, "correlation_label": "-0.06"}, {"variable": "AIANPopShare", "variable2": "LOWIPopShare", "correlation": 0.08869664714423041, "correlation_label": "0.09"}, {"variable": "AIANPopShare", "variable2": "KidsPopShare", "correlation": 0.09355020223263774, "correlation_label": "0.09"}, {"variable": "AIANPopShare", "variable2": "SeniorsPopShare", "correlation": -0.048014584743454804, "correlation_label": "-0.05"}, {"variable": "AIANPopShare", "variable2": "WhitePopShare", "correlation": -0.13563654702432712, "correlation_label": "-0.14"}, {"variable": "AIANPopShare", "variable2": "AsianPopShare", "correlation": -0.04448442329308261, "correlation_label": "-0.04"}, {"variable": "AIANPopShare", "variable2": "NHOPIPopShare", "correlation": 0.0003911724550259673, "correlation_label": "0.00"}, {"variable": "AIANPopShare", "variable2": "AIANPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "AIANPopShare", "variable2": "OMultirPopShare", "correlation": 0.04532528385373499, "correlation_label": "0.05"}, {"variable": "AIANPopShare", "variable2": "HispanicPopShare", "correlation": 0.017850492152250548, "correlation_label": "0.02"}, {"variable": "AIANPopShare", "variable2": "HUNVPopShare", "correlation": 0.012167489456452065, "correlation_label": "0.01"}, {"variable": "AIANPopShare", "variable2": "SNAPPopShare", "correlation": 0.08514295414213996, "correlation_label": "0.09"}, {"variable": "OMultirPopShare", "variable2": "PovertyRate", "correlation": 0.26503630022370694, "correlation_label": "0.27"}, {"variable": "OMultirPopShare", "variable2": "BlackPopShare", "correlation": -0.024183082909660578, "correlation_label": "-0.02"}, {"variable": "OMultirPopShare", "variable2": "LOWIPopShare", "correlation": 0.2815892149431609, "correlation_label": "0.28"}, {"variable": "OMultirPopShare", "variable2": "KidsPopShare", "correlation": 0.3409743847501985, "correlation_label": "0.34"}, {"variable": "OMultirPopShare", "variable2": "SeniorsPopShare", "correlation": -0.3361408222052703, "correlation_label": "-0.34"}, {"variable": "OMultirPopShare", "variable2": "WhitePopShare", "correlation": -0.4373200625070667, "correlation_label": "-0.44"}, {"variable": "OMultirPopShare", "variable2": "AsianPopShare", "correlation": 0.16387753106144903, "correlation_label": "0.16"}, {"variable": "OMultirPopShare", "variable2": "NHOPIPopShare", "correlation": 0.15497754497224042, "correlation_label": "0.15"}, {"variable": "OMultirPopShare", "variable2": "AIANPopShare", "correlation": 0.04532528385373499, "correlation_label": "0.05"}, {"variable": "OMultirPopShare", "variable2": "OMultirPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "OMultirPopShare", "variable2": "HispanicPopShare", "correlation": 0.8658350672812237, "correlation_label": "0.87"}, {"variable": "OMultirPopShare", "variable2": "HUNVPopShare", "correlation": 0.12244067209636002, "correlation_label": "0.12"}, {"variable": "OMultirPopShare", "variable2": "SNAPPopShare", "correlation": 0.26788647420949874, "correlation_label": "0.27"}, {"variable": "HispanicPopShare", "variable2": "PovertyRate", "correlation": 0.2447115298191319, "correlation_label": "0.24"}, {"variable": "HispanicPopShare", "variable2": "BlackPopShare", "correlation": -0.07347609624427635, "correlation_label": "-0.07"}, {"variable": "HispanicPopShare", "variable2": "LOWIPopShare", "correlation": 0.2782108970331435, "correlation_label": "0.28"}, {"variable": "HispanicPopShare", "variable2": "KidsPopShare", "correlation": 0.3296765739817506, "correlation_label": "0.33"}, {"variable": "HispanicPopShare", "variable2": "SeniorsPopShare", "correlation": -0.2864472738585831, "correlation_label": "-0.29"}, {"variable": "HispanicPopShare", "variable2": "WhitePopShare", "correlation": -0.3033886391343884, "correlation_label": "-0.30"}, {"variable": "HispanicPopShare", "variable2": "AsianPopShare", "correlation": 0.07439188042340097, "correlation_label": "0.07"}, {"variable": "HispanicPopShare", "variable2": "NHOPIPopShare", "correlation": 0.03916764532731688, "correlation_label": "0.04"}, {"variable": "HispanicPopShare", "variable2": "AIANPopShare", "correlation": 0.017850492152250548, "correlation_label": "0.02"}, {"variable": "HispanicPopShare", "variable2": "OMultirPopShare", "correlation": 0.8658350672812237, "correlation_label": "0.87"}, {"variable": "HispanicPopShare", "variable2": "HispanicPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "HispanicPopShare", "variable2": "HUNVPopShare", "correlation": 0.1032862966810038, "correlation_label": "0.10"}, {"variable": "HispanicPopShare", "variable2": "SNAPPopShare", "correlation": 0.26425647729412693, "correlation_label": "0.26"}, {"variable": "HUNVPopShare", "variable2": "PovertyRate", "correlation": 0.2829084983393503, "correlation_label": "0.28"}, {"variable": "HUNVPopShare", "variable2": "BlackPopShare", "correlation": 0.23872163013090897, "correlation_label": "0.24"}, {"variable": "HUNVPopShare", "variable2": "LOWIPopShare", "correlation": 0.2561147097146154, "correlation_label": "0.26"}, {"variable": "HUNVPopShare", "variable2": "KidsPopShare", "correlation": -0.05273259186214481, "correlation_label": "-0.05"}, {"variable": "HUNVPopShare", "variable2": "SeniorsPopShare", "correlation": -0.05514190726297414, "correlation_label": "-0.06"}, {"variable": "HUNVPopShare", "variable2": "WhitePopShare", "correlation": -0.27505280573810764, "correlation_label": "-0.28"}, {"variable": "HUNVPopShare", "variable2": "AsianPopShare", "correlation": 0.05402301408683258, "correlation_label": "0.05"}, {"variable": "HUNVPopShare", "variable2": "NHOPIPopShare", "correlation": 0.009043128632616073, "correlation_label": "0.01"}, {"variable": "HUNVPopShare", "variable2": "AIANPopShare", "correlation": 0.012167489456452065, "correlation_label": "0.01"}, {"variable": "HUNVPopShare", "variable2": "OMultirPopShare", "correlation": 0.12244067209636002, "correlation_label": "0.12"}, {"variable": "HUNVPopShare", "variable2": "HispanicPopShare", "correlation": 0.1032862966810038, "correlation_label": "0.10"}, {"variable": "HUNVPopShare", "variable2": "HUNVPopShare", "correlation": 1.0, "correlation_label": "1.00"}, {"variable": "HUNVPopShare", "variable2": "SNAPPopShare", "correlation": 0.4607312721409357, "correlation_label": "0.46"}, {"variable": "SNAPPopShare", "variable2": "PovertyRate", "correlation": 0.7060501024305214, "correlation_label": "0.71"}, {"variable": "SNAPPopShare", "variable2": "BlackPopShare", "correlation": 0.4704181429214918, "correlation_label": "0.47"}, {"variable": "SNAPPopShare", "variable2": "LOWIPopShare", "correlation": 0.6046305687159963, "correlation_label": "0.60"}, {"variable": "SNAPPopShare", "variable2": "KidsPopShare", "correlation": 0.2809451289140096, "correlation_label": "0.28"}, {"variable": "SNAPPopShare", "variable2": "SeniorsPopShare", "correlation": -0.1545011100894765, "correlation_label": "-0.15"}, {"variable": "SNAPPopShare", "variable2": "WhitePopShare", "correlation": -0.4804250695653086, "correlation_label": "-0.48"}, {"variable": "SNAPPopShare", "variable2": "AsianPopShare", "correlation": -0.14834678386726421, "correlation_label": "-0.15"}, {"variable": "SNAPPopShare", "variable2": "NHOPIPopShare", "correlation": 0.022474545009895625, "correlation_label": "0.02"}, {"variable": "SNAPPopShare", "variable2": "AIANPopShare", "correlation": 0.08514295414213996, "correlation_label": "0.09"}, {"variable": "SNAPPopShare", "variable2": "OMultirPopShare", "correlation": 0.26788647420949874, "correlation_label": "0.27"}, {"variable": "SNAPPopShare", "variable2": "HispanicPopShare", "correlation": 0.26425647729412693, "correlation_label": "0.26"}, {"variable": "SNAPPopShare", "variable2": "HUNVPopShare", "correlation": 0.4607312721409357, "correlation_label": "0.46"}, {"variable": "SNAPPopShare", "variable2": "SNAPPopShare", "correlation": 1.0, "correlation_label": "1.00"}]}}, {"mode": "vega-lite"});
</script>



#### Differences in Population Means
Below, I explore the relationship between the population means when comparing populations that are low income and low access with populations that are not.


```python
lila_1_black = df[df['LILATracts_1And10'] == 1]['BlackPopShare']
lila_0_black = df[df['LILATracts_1And10'] == 0]['BlackPopShare']
lila_1_white = df[df['LILATracts_1And10'] == 1]['WhitePopShare']
lila_0_white = df[df['LILATracts_1And10'] == 0]['WhitePopShare']
print('Low Access, Low Income: Black Population Mean: {:.2}'.format(lila_1_black.mean()))
print('Not Low Access, Low Income: Black Population Mean: {:.2}'.format(lila_0_black.mean()))
print('Low Access, Low Income: White Population Mean: {:.2}'.format(lila_1_white.mean()))
print('Not Low Access, Low Income: White Population Mean: {:.2}'.format(lila_0_white.mean()))
```

    Low Access, Low Income: Black Population Mean: 0.23
    Not Low Access, Low Income: Black Population Mean: 0.12
    Low Access, Low Income: White Population Mean: 0.63
    Not Low Access, Low Income: White Population Mean: 0.73


The means appear to suggest that, low access, low income communities have, on average, a higher percentage of Black population than thoese communities that are not low access, low income.

In addition, low access, low income communities have, on average, a lower percentage of White population than thoese communities that are not low access, low income.

This bears out in the following t-tests:

Test the null hypothesis that the population share means are equal for Black populations, vs. the alternative hypothesis that low access, low income populations have greater Black population means than non-low access, low income populations:


```python
ttest_ind(lila_1_black.tolist(), lila_0_black.tolist(), equal_var=False, alternative='greater')
```




    Ttest_indResult(statistic=36.07918837717623, pvalue=1.45167200558015e-269)



At a level of significance of .05, we reject the null hypothesis in favor of the alternative.

Test the null hypothesis that the population share means are equal for White populations, vs. the alternative hypothesis that low access, low income populations have lower White population means than non-low access, low income populations:


```python
ttest_ind(lila_1_white.tolist(), lila_0_white.tolist(), equal_var=False, alternative='less')
```




    Ttest_indResult(statistic=-32.08791001570348, pvalue=9.128973228515656e-217)



At a level of significance of .05, we reject the null hypothesis in favor of the alternative.

As a result of the above tests, we can likely say that Black and White population means will likely be good features for our binary classification model.

#### Population Demographics
Below, I further explore the population demographics as they relate to low access, low income communities.


```python
bins = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0]
names = ['0%-10%', '10%-20%', '20%-30%', '30%-40%', '40%-50%', '50%-60%', '60%-70%', '70%-80%', '80%-90%', '90%-100%']
```

##### White and Black Communites Compared
Extending from above, I visualized the relationship between White and Black population percentages, and number of low income, low access communities.  We can see clearly here that, as the White population percentage increases, the percentage of low income, low access communites decrease.  Contrasting that, as the Black population percentage increases, the percentage of low income, low access communities decrease.  This again supports our findings above, and also indicates these may be good indicators for our models.


```python
# Segment and calulate Black and White population statistics
eda_df = df.copy()
eda_df['PopShareRange'] = pd.cut(eda_df['BlackPopShare'], bins, labels=names)
b = eda_df[['PopShareRange',label]].groupby(['PopShareRange'], as_index=False).aggregate({label:['sum','count', 'mean']})
b.columns = [f"{x}_{y}" for x, y in b.columns.to_flat_index()]
b['Demographic'] = 'Black'

eda_df['PopShareRange'] = pd.cut(eda_df['WhitePopShare'], bins, labels=names)
w = eda_df[['PopShareRange',label]].groupby(['PopShareRange'], as_index=False).aggregate({label:['sum','count', 'mean']})
w.columns = [f"{x}_{y}" for x, y in w.columns.to_flat_index()]
w['Demographic'] = 'White'

bw = pd.concat([b,w])
```


```python
# Visualize
alt.Chart(bw).mark_bar(size=25).encode(
    x=alt.X('Demographic:N', title='', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('LILATracts_1And10_mean:Q', axis=alt.Axis(format='%'), title='Percent Low Income, Low Access Tracts'),
    color='Demographic:N',
    column=alt.Column('PopShareRange_:O', title='Population Percentage Range')
).properties(
    height=300,
    width=60
).configure_axis(
    labelFontSize=13,
    titleFontSize=15
).configure_header(
    labelFontSize=13,
    titleFontSize=15,
    labelOrient='bottom',
    titleOrient='bottom'
)
```





<div id="altair-viz-e69ed9f2a7ee4a4d9f6e7cd2d61d958b"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-e69ed9f2a7ee4a4d9f6e7cd2d61d958b") {
      outputDiv = document.getElementById("altair-viz-e69ed9f2a7ee4a4d9f6e7cd2d61d958b");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 13, "titleFontSize": 15}, "header": {"labelFontSize": 13, "labelOrient": "bottom", "titleFontSize": 15, "titleOrient": "bottom"}}, "data": {"name": "data-130c1534ce772d2f9455d20d58017946"}, "mark": {"type": "bar", "size": 25}, "encoding": {"color": {"field": "Demographic", "type": "nominal"}, "column": {"field": "PopShareRange_", "title": "Population Percentage Range", "type": "ordinal"}, "x": {"axis": {"labelAngle": -45}, "field": "Demographic", "title": "", "type": "nominal"}, "y": {"axis": {"format": "%"}, "field": "LILATracts_1And10_mean", "title": "Percent Low Income, Low Access Tracts", "type": "quantitative"}}, "height": 300, "width": 60, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-130c1534ce772d2f9455d20d58017946": [{"PopShareRange_": "0%-10%", "LILATracts_1And10_sum": 4584, "LILATracts_1And10_count": 49022, "LILATracts_1And10_mean": 0.09350903675900617, "Demographic": "Black"}, {"PopShareRange_": "10%-20%", "LILATracts_1And10_sum": 1234, "LILATracts_1And10_count": 8426, "LILATracts_1And10_mean": 0.14645145976738666, "Demographic": "Black"}, {"PopShareRange_": "20%-30%", "LILATracts_1And10_sum": 721, "LILATracts_1And10_count": 4098, "LILATracts_1And10_mean": 0.17593948267447534, "Demographic": "Black"}, {"PopShareRange_": "30%-40%", "LILATracts_1And10_sum": 505, "LILATracts_1And10_count": 2469, "LILATracts_1And10_mean": 0.20453624949372215, "Demographic": "Black"}, {"PopShareRange_": "40%-50%", "LILATracts_1And10_sum": 439, "LILATracts_1And10_count": 1699, "LILATracts_1And10_mean": 0.2583872866391995, "Demographic": "Black"}, {"PopShareRange_": "50%-60%", "LILATracts_1And10_sum": 372, "LILATracts_1And10_count": 1296, "LILATracts_1And10_mean": 0.28703703703703703, "Demographic": "Black"}, {"PopShareRange_": "60%-70%", "LILATracts_1And10_sum": 320, "LILATracts_1And10_count": 1026, "LILATracts_1And10_mean": 0.31189083820662766, "Demographic": "Black"}, {"PopShareRange_": "70%-80%", "LILATracts_1And10_sum": 308, "LILATracts_1And10_count": 1018, "LILATracts_1And10_mean": 0.3025540275049116, "Demographic": "Black"}, {"PopShareRange_": "80%-90%", "LILATracts_1And10_sum": 335, "LILATracts_1And10_count": 1186, "LILATracts_1And10_mean": 0.2824620573355818, "Demographic": "Black"}, {"PopShareRange_": "90%-100%", "LILATracts_1And10_sum": 406, "LILATracts_1And10_count": 1759, "LILATracts_1And10_mean": 0.23081296191017622, "Demographic": "Black"}, {"PopShareRange_": "0%-10%", "LILATracts_1And10_sum": 676, "LILATracts_1And10_count": 2872, "LILATracts_1And10_mean": 0.23537604456824512, "Demographic": "White"}, {"PopShareRange_": "10%-20%", "LILATracts_1And10_sum": 417, "LILATracts_1And10_count": 1885, "LILATracts_1And10_mean": 0.22122015915119364, "Demographic": "White"}, {"PopShareRange_": "20%-30%", "LILATracts_1And10_sum": 408, "LILATracts_1And10_count": 2171, "LILATracts_1And10_mean": 0.1879318286503915, "Demographic": "White"}, {"PopShareRange_": "30%-40%", "LILATracts_1And10_sum": 526, "LILATracts_1And10_count": 2828, "LILATracts_1And10_mean": 0.185997171145686, "Demographic": "White"}, {"PopShareRange_": "40%-50%", "LILATracts_1And10_sum": 668, "LILATracts_1And10_count": 4075, "LILATracts_1And10_mean": 0.16392638036809815, "Demographic": "White"}, {"PopShareRange_": "50%-60%", "LILATracts_1And10_sum": 875, "LILATracts_1And10_count": 5182, "LILATracts_1And10_mean": 0.16885372443072172, "Demographic": "White"}, {"PopShareRange_": "60%-70%", "LILATracts_1And10_sum": 1049, "LILATracts_1And10_count": 6621, "LILATracts_1And10_mean": 0.1584352816795046, "Demographic": "White"}, {"PopShareRange_": "70%-80%", "LILATracts_1And10_sum": 1276, "LILATracts_1And10_count": 9601, "LILATracts_1And10_mean": 0.13290282262264347, "Demographic": "White"}, {"PopShareRange_": "80%-90%", "LILATracts_1And10_sum": 1550, "LILATracts_1And10_count": 14759, "LILATracts_1And10_mean": 0.10502066535673149, "Demographic": "White"}, {"PopShareRange_": "90%-100%", "LILATracts_1And10_sum": 1844, "LILATracts_1And10_count": 22398, "LILATracts_1And10_mean": 0.08232877935529959, "Demographic": "White"}]}}, {"mode": "vega-lite"});
</script>



##### KidsPopShare and Low Income, Low Access Communities
From the correlation matrix above, it seemed like KidsPopShare was an interesting feature.  Looking at this data a little closely, we do see that as the kids population increases, we do see a very sharp increase in low income, low access communities.  This, of course, drops to zero at 60%, because there is a logical limit in the percentage of kids vs. parents in any given community.


```python
# Calculate KidsPopShare statistics
eda_df = df.copy()
eda_df['PopShareRange'] = pd.cut(eda_df['KidsPopShare'], bins, labels=names)
k = eda_df[['PopShareRange',label]].groupby(['PopShareRange'], as_index=False).aggregate({label:['sum','count', 'mean']})
k.columns = [f"{x}_{y}" for x, y in k.columns.to_flat_index()]
```


```python
# Visualize
alt.Chart(k).mark_bar(size=25).encode(
    x=alt.X('PopShareRange_:O', title='Population Percentage Range', axis=alt.Axis(labelAngle=-45)),
    y=alt.Y('LILATracts_1And10_mean:Q', axis=alt.Axis(format='%'), title='Percent Low Income, Low Access Tracts'),
).properties(
    height=300,
    width=600
).configure_axis(
    labelFontSize=13,
    titleFontSize=15
)
```





<div id="altair-viz-211112ab5d3a41e99f5a15c955feeccc"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-211112ab5d3a41e99f5a15c955feeccc") {
      outputDiv = document.getElementById("altair-viz-211112ab5d3a41e99f5a15c955feeccc");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 13, "titleFontSize": 15}}, "data": {"name": "data-c50f7f28298f62bd50d50caaaf12e3aa"}, "mark": {"type": "bar", "size": 25}, "encoding": {"x": {"axis": {"labelAngle": -45}, "field": "PopShareRange_", "title": "Population Percentage Range", "type": "ordinal"}, "y": {"axis": {"format": "%"}, "field": "LILATracts_1And10_mean", "title": "Percent Low Income, Low Access Tracts", "type": "quantitative"}}, "height": 300, "width": 600, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-c50f7f28298f62bd50d50caaaf12e3aa": [{"PopShareRange_": "0%-10%", "LILATracts_1And10_sum": 312, "LILATracts_1And10_count": 2552, "LILATracts_1And10_mean": 0.12225705329153605}, {"PopShareRange_": "10%-20%", "LILATracts_1And10_sum": 1389, "LILATracts_1And10_count": 14240, "LILATracts_1And10_mean": 0.09754213483146068}, {"PopShareRange_": "20%-30%", "LILATracts_1And10_sum": 5796, "LILATracts_1And10_count": 46189, "LILATracts_1And10_mean": 0.125484422698045}, {"PopShareRange_": "30%-40%", "LILATracts_1And10_sum": 1691, "LILATracts_1And10_count": 8988, "LILATracts_1And10_mean": 0.18813974187805962}, {"PopShareRange_": "40%-50%", "LILATracts_1And10_sum": 91, "LILATracts_1And10_count": 322, "LILATracts_1And10_mean": 0.2826086956521739}, {"PopShareRange_": "50%-60%", "LILATracts_1And10_sum": 7, "LILATracts_1And10_count": 31, "LILATracts_1And10_mean": 0.22580645161290322}, {"PopShareRange_": "60%-70%", "LILATracts_1And10_sum": 0, "LILATracts_1And10_count": 5, "LILATracts_1And10_mean": 0.0}, {"PopShareRange_": "70%-80%", "LILATracts_1And10_sum": 0, "LILATracts_1And10_count": 1, "LILATracts_1And10_mean": 0.0}, {"PopShareRange_": "80%-90%", "LILATracts_1And10_sum": 0, "LILATracts_1And10_count": 0, "LILATracts_1And10_mean": null}, {"PopShareRange_": "90%-100%", "LILATracts_1And10_sum": 0, "LILATracts_1And10_count": 0, "LILATracts_1And10_mean": null}]}}, {"mode": "vega-lite"});
</script>



##### States and Low Income, Low Access Communities
I plotted the states and Low Income, Low Access communities below because I wanted to explore if there was any kind of geo-spacial relationship driving food desert communities.  Indeed, I did find that the states with the most low income, low access communities appeared to be in one region of the country, and the states with the lowest percentage of low income, low access communities appeared to be in another.  This prompted me to add geographic regions to my data as an engineered feature.


```python
# Calculate State statistics
eda_df = df.copy()
k = eda_df[['State',label]].groupby(['State'], as_index=False).aggregate({label:['sum','count', 'mean']})
k.columns = [f"{x}_{y}" for x, y in k.columns.to_flat_index()]
```


```python
# Visualize
alt.Chart(k).mark_bar(size=10).encode(
    y=alt.Y('State_:N', title='State', sort='-x'),
    x=alt.X('LILATracts_1And10_mean:Q', axis=alt.Axis(format='%'), title='Percent Low Income, Low Access Tracts'),
).properties(
    height=800,
    width=600
).configure_axis(
    labelFontSize=13,
    titleFontSize=15
)
```





<div id="altair-viz-53473a31ea2a4ac687a105ca9e17b73a"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-53473a31ea2a4ac687a105ca9e17b73a") {
      outputDiv = document.getElementById("altair-viz-53473a31ea2a4ac687a105ca9e17b73a");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 13, "titleFontSize": 15}}, "data": {"name": "data-d91e253bc5e9923b47a17ce70dbbaa4b"}, "mark": {"type": "bar", "size": 10}, "encoding": {"x": {"axis": {"format": "%"}, "field": "LILATracts_1And10_mean", "title": "Percent Low Income, Low Access Tracts", "type": "quantitative"}, "y": {"field": "State_", "sort": "-x", "title": "State", "type": "nominal"}}, "height": 800, "width": 600, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-d91e253bc5e9923b47a17ce70dbbaa4b": [{"State_": "Alabama", "LILATracts_1And10_sum": 267, "LILATracts_1And10_count": 1177, "LILATracts_1And10_mean": 0.2268479184367035}, {"State_": "Alaska", "LILATracts_1And10_sum": 33, "LILATracts_1And10_count": 166, "LILATracts_1And10_mean": 0.19879518072289157}, {"State_": "Arizona", "LILATracts_1And10_sum": 257, "LILATracts_1And10_count": 1516, "LILATracts_1And10_mean": 0.1695250659630607}, {"State_": "Arkansas", "LILATracts_1And10_sum": 171, "LILATracts_1And10_count": 686, "LILATracts_1And10_mean": 0.24927113702623907}, {"State_": "California", "LILATracts_1And10_sum": 536, "LILATracts_1And10_count": 8008, "LILATracts_1And10_mean": 0.06693306693306693}, {"State_": "Colorado", "LILATracts_1And10_sum": 172, "LILATracts_1And10_count": 1237, "LILATracts_1And10_mean": 0.13904607922392886}, {"State_": "Connecticut", "LILATracts_1And10_sum": 65, "LILATracts_1And10_count": 826, "LILATracts_1And10_mean": 0.07869249394673124}, {"State_": "Delaware", "LILATracts_1And10_sum": 32, "LILATracts_1And10_count": 214, "LILATracts_1And10_mean": 0.14953271028037382}, {"State_": "District of Columbia", "LILATracts_1And10_sum": 12, "LILATracts_1And10_count": 179, "LILATracts_1And10_mean": 0.0670391061452514}, {"State_": "Florida", "LILATracts_1And10_sum": 550, "LILATracts_1And10_count": 4176, "LILATracts_1And10_mean": 0.13170498084291188}, {"State_": "Georgia", "LILATracts_1And10_sum": 441, "LILATracts_1And10_count": 1956, "LILATracts_1And10_mean": 0.2254601226993865}, {"State_": "Hawaii", "LILATracts_1And10_sum": 32, "LILATracts_1And10_count": 320, "LILATracts_1And10_mean": 0.1}, {"State_": "Idaho", "LILATracts_1And10_sum": 41, "LILATracts_1And10_count": 298, "LILATracts_1And10_mean": 0.13758389261744966}, {"State_": "Illinois", "LILATracts_1And10_sum": 319, "LILATracts_1And10_count": 3115, "LILATracts_1And10_mean": 0.10240770465489567}, {"State_": "Indiana", "LILATracts_1And10_sum": 291, "LILATracts_1And10_count": 1506, "LILATracts_1And10_mean": 0.19322709163346613}, {"State_": "Iowa", "LILATracts_1And10_sum": 85, "LILATracts_1And10_count": 823, "LILATracts_1And10_mean": 0.10328068043742406}, {"State_": "Kansas", "LILATracts_1And10_sum": 139, "LILATracts_1And10_count": 764, "LILATracts_1And10_mean": 0.18193717277486912}, {"State_": "Kentucky", "LILATracts_1And10_sum": 153, "LILATracts_1And10_count": 1109, "LILATracts_1And10_mean": 0.13796212804328223}, {"State_": "Louisiana", "LILATracts_1And10_sum": 258, "LILATracts_1And10_count": 1127, "LILATracts_1And10_mean": 0.22892635314995563}, {"State_": "Maine", "LILATracts_1And10_sum": 30, "LILATracts_1And10_count": 351, "LILATracts_1And10_mean": 0.08547008547008547}, {"State_": "Maryland", "LILATracts_1And10_sum": 131, "LILATracts_1And10_count": 1387, "LILATracts_1And10_mean": 0.09444844989185291}, {"State_": "Massachusetts", "LILATracts_1And10_sum": 112, "LILATracts_1And10_count": 1464, "LILATracts_1And10_mean": 0.07650273224043716}, {"State_": "Michigan", "LILATracts_1And10_sum": 339, "LILATracts_1And10_count": 2749, "LILATracts_1And10_mean": 0.1233175700254638}, {"State_": "Minnesota", "LILATracts_1And10_sum": 192, "LILATracts_1And10_count": 1331, "LILATracts_1And10_mean": 0.14425244177310292}, {"State_": "Mississippi", "LILATracts_1And10_sum": 208, "LILATracts_1And10_count": 658, "LILATracts_1And10_mean": 0.3161094224924012}, {"State_": "Missouri", "LILATracts_1And10_sum": 248, "LILATracts_1And10_count": 1391, "LILATracts_1And10_mean": 0.17828900071890727}, {"State_": "Montana", "LILATracts_1And10_sum": 36, "LILATracts_1And10_count": 270, "LILATracts_1And10_mean": 0.13333333333333333}, {"State_": "Nebraska", "LILATracts_1And10_sum": 55, "LILATracts_1And10_count": 530, "LILATracts_1And10_mean": 0.10377358490566038}, {"State_": "Nevada", "LILATracts_1And10_sum": 49, "LILATracts_1And10_count": 680, "LILATracts_1And10_mean": 0.07205882352941176}, {"State_": "New Hampshire", "LILATracts_1And10_sum": 38, "LILATracts_1And10_count": 292, "LILATracts_1And10_mean": 0.13013698630136986}, {"State_": "New Jersey", "LILATracts_1And10_sum": 108, "LILATracts_1And10_count": 1999, "LILATracts_1And10_mean": 0.054027013506753374}, {"State_": "New Mexico", "LILATracts_1And10_sum": 126, "LILATracts_1And10_count": 498, "LILATracts_1And10_mean": 0.25301204819277107}, {"State_": "New York", "LILATracts_1And10_sum": 194, "LILATracts_1And10_count": 4855, "LILATracts_1And10_mean": 0.03995880535530381}, {"State_": "North Carolina", "LILATracts_1And10_sum": 353, "LILATracts_1And10_count": 2171, "LILATracts_1And10_mean": 0.1625978811607554}, {"State_": "North Dakota", "LILATracts_1And10_sum": 17, "LILATracts_1And10_count": 205, "LILATracts_1And10_mean": 0.08292682926829269}, {"State_": "Ohio", "LILATracts_1And10_sum": 421, "LILATracts_1And10_count": 2942, "LILATracts_1And10_mean": 0.14309993201903468}, {"State_": "Oklahoma", "LILATracts_1And10_sum": 177, "LILATracts_1And10_count": 1046, "LILATracts_1And10_mean": 0.16921606118546845}, {"State_": "Oregon", "LILATracts_1And10_sum": 100, "LILATracts_1And10_count": 825, "LILATracts_1And10_mean": 0.12121212121212122}, {"State_": "Pennsylvania", "LILATracts_1And10_sum": 237, "LILATracts_1And10_count": 3207, "LILATracts_1And10_mean": 0.07390084190832553}, {"State_": "Rhode Island", "LILATracts_1And10_sum": 13, "LILATracts_1And10_count": 241, "LILATracts_1And10_mean": 0.05394190871369295}, {"State_": "South Carolina", "LILATracts_1And10_sum": 217, "LILATracts_1And10_count": 1089, "LILATracts_1And10_mean": 0.1992653810835629}, {"State_": "South Dakota", "LILATracts_1And10_sum": 32, "LILATracts_1And10_count": 219, "LILATracts_1And10_mean": 0.1461187214611872}, {"State_": "Tennessee", "LILATracts_1And10_sum": 266, "LILATracts_1And10_count": 1487, "LILATracts_1And10_mean": 0.1788836583725622}, {"State_": "Texas", "LILATracts_1And10_sum": 1021, "LILATracts_1And10_count": 5231, "LILATracts_1And10_mean": 0.19518256547505258}, {"State_": "Utah", "LILATracts_1And10_sum": 49, "LILATracts_1And10_count": 584, "LILATracts_1And10_mean": 0.0839041095890411}, {"State_": "Vermont", "LILATracts_1And10_sum": 12, "LILATracts_1And10_count": 183, "LILATracts_1And10_mean": 0.06557377049180328}, {"State_": "Virginia", "LILATracts_1And10_sum": 269, "LILATracts_1And10_count": 1882, "LILATracts_1And10_mean": 0.14293304994686504}, {"State_": "Washington", "LILATracts_1And10_sum": 174, "LILATracts_1And10_count": 1445, "LILATracts_1And10_mean": 0.12041522491349481}, {"State_": "West Virginia", "LILATracts_1And10_sum": 66, "LILATracts_1And10_count": 484, "LILATracts_1And10_mean": 0.13636363636363635}, {"State_": "Wisconsin", "LILATracts_1And10_sum": 133, "LILATracts_1And10_count": 1391, "LILATracts_1And10_mean": 0.09561466570812366}, {"State_": "Wyoming", "LILATracts_1And10_sum": 12, "LILATracts_1And10_count": 131, "LILATracts_1And10_mean": 0.0916030534351145}]}}, {"mode": "vega-lite"});
</script>



##### Regions and Low Income, Low Access Communites
Extending from above, I plotted regions and their relationships to low income, low access communities.  What I noticed above does appear to be supported by the data.


```python
# Calculate division statistics
eda_df = df.copy()
d = eda_df[['Census Division',label]].groupby(['Census Division'], as_index=False).aggregate({label:['sum','count', 'mean']})
d.columns = [f"{x}_{y}" for x, y in d.columns.to_flat_index()]
```


```python
# Visualize
alt.Chart(d).mark_bar(size=20).encode(
    y=alt.Y('Census Division_:N', title='Census Division', sort='-x'),
    x=alt.X('LILATracts_1And10_mean:Q', axis=alt.Axis(format='%'), title='Percent Low Income, Low Access Tracts'),
).properties(
    height=300,
    width=600
).configure_axis(
    labelFontSize=13,
    titleFontSize=15
)
```





<div id="altair-viz-3b84ed226ac245f3a4404a449a37a8c5"></div>
<script type="text/javascript">
  var VEGA_DEBUG = (typeof VEGA_DEBUG == "undefined") ? {} : VEGA_DEBUG;
  (function(spec, embedOpt){
    let outputDiv = document.currentScript.previousElementSibling;
    if (outputDiv.id !== "altair-viz-3b84ed226ac245f3a4404a449a37a8c5") {
      outputDiv = document.getElementById("altair-viz-3b84ed226ac245f3a4404a449a37a8c5");
    }
    const paths = {
      "vega": "https://cdn.jsdelivr.net/npm//vega@5?noext",
      "vega-lib": "https://cdn.jsdelivr.net/npm//vega-lib?noext",
      "vega-lite": "https://cdn.jsdelivr.net/npm//vega-lite@4.17.0?noext",
      "vega-embed": "https://cdn.jsdelivr.net/npm//vega-embed@6?noext",
    };

    function maybeLoadScript(lib, version) {
      var key = `${lib.replace("-", "")}_version`;
      return (VEGA_DEBUG[key] == version) ?
        Promise.resolve(paths[lib]) :
        new Promise(function(resolve, reject) {
          var s = document.createElement('script');
          document.getElementsByTagName("head")[0].appendChild(s);
          s.async = true;
          s.onload = () => {
            VEGA_DEBUG[key] = version;
            return resolve(paths[lib]);
          };
          s.onerror = () => reject(`Error loading script: ${paths[lib]}`);
          s.src = paths[lib];
        });
    }

    function showError(err) {
      outputDiv.innerHTML = `<div class="error" style="color:red;">${err}</div>`;
      throw err;
    }

    function displayChart(vegaEmbed) {
      vegaEmbed(outputDiv, spec, embedOpt)
        .catch(err => showError(`Javascript Error: ${err.message}<br>This usually means there's a typo in your chart specification. See the javascript console for the full traceback.`));
    }

    if(typeof define === "function" && define.amd) {
      requirejs.config({paths});
      require(["vega-embed"], displayChart, err => showError(`Error loading script: ${err.message}`));
    } else {
      maybeLoadScript("vega", "5")
        .then(() => maybeLoadScript("vega-lite", "4.17.0"))
        .then(() => maybeLoadScript("vega-embed", "6"))
        .catch(showError)
        .then(() => displayChart(vegaEmbed));
    }
  })({"config": {"view": {"continuousWidth": 400, "continuousHeight": 300}, "axis": {"labelFontSize": 13, "titleFontSize": 15}}, "data": {"name": "data-f7928f836fd4056d624fd30437c72d8e"}, "mark": {"type": "bar", "size": 20}, "encoding": {"x": {"axis": {"format": "%"}, "field": "LILATracts_1And10_mean", "title": "Percent Low Income, Low Access Tracts", "type": "quantitative"}, "y": {"field": "Census Division_", "sort": "-x", "title": "Census Division", "type": "nominal"}}, "height": 300, "width": 600, "$schema": "https://vega.github.io/schema/vega-lite/v4.17.0.json", "datasets": {"data-f7928f836fd4056d624fd30437c72d8e": [{"Census Division_": "East North Central", "LILATracts_1And10_sum": 1503, "LILATracts_1And10_count": 11703, "LILATracts_1And10_mean": 0.12842860804921816}, {"Census Division_": "East South Central", "LILATracts_1And10_sum": 894, "LILATracts_1And10_count": 4431, "LILATracts_1And10_mean": 0.2017603249830738}, {"Census Division_": "Middle Atlantic", "LILATracts_1And10_sum": 539, "LILATracts_1And10_count": 10061, "LILATracts_1And10_mean": 0.05357320345890071}, {"Census Division_": "Mountain", "LILATracts_1And10_sum": 742, "LILATracts_1And10_count": 5214, "LILATracts_1And10_mean": 0.14230916762562332}, {"Census Division_": "New England", "LILATracts_1And10_sum": 270, "LILATracts_1And10_count": 3357, "LILATracts_1And10_mean": 0.08042895442359249}, {"Census Division_": "Pacific", "LILATracts_1And10_sum": 810, "LILATracts_1And10_count": 10278, "LILATracts_1And10_mean": 0.07880910683012259}, {"Census Division_": "South Atlantic", "LILATracts_1And10_sum": 2059, "LILATracts_1And10_count": 13359, "LILATracts_1And10_mean": 0.15412830301669286}, {"Census Division_": "West North Central", "LILATracts_1And10_sum": 768, "LILATracts_1And10_count": 5263, "LILATracts_1And10_mean": 0.14592437773133193}, {"Census Division_": "West South Central", "LILATracts_1And10_sum": 1627, "LILATracts_1And10_count": 8090, "LILATracts_1And10_mean": 0.20111248454882572}]}}, {"mode": "vega-lite"});
</script>



### Models

#### Test/Train Preparation
Below, I select the features to include in my dataset.

In addition, I incorporate my engineered feature `Census Division` into the training dataset.  Because this engineered feature is categorical, in order for it to be used properly in the model, I expand it into one binary True/False feature per division via Pandas' `get_dummies` method.


```python
# Prepare data
data = df.loc[:, df.columns.isin(['PovertyRate','MedianFamilyIncome','LowIncomeTracts','BlackPopShare','LOWIPopShare','KidsPopShare','SeniorsPopShare','WhitePopShare','AsianPopShare','NHOPIPopShare','AIANPopShare','OMultirPopShare','HispanicPopShare','HUNVPopShare','SNAPPopShare'])]
# Expand the Urban feature into two:  Rural and Urban
data = pd.concat([data, pd.get_dummies(df['Urban'])], axis=1)
data = data.rename({0:'Rural', 1:'Urban'}, axis=1)
# Expand the Census Division feature into one True/False feature per Census Division value.
data = pd.concat([data, pd.get_dummies(df['Census Division'])], axis=1)
# Assign labels
labels = df[label]
```

Using the prepared data, I perform a train/test split, with 80% training data and 20% test data:


```python
# Train test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
print('Training Features Count: {}  Training Labels Count: {}'.format(len(X_train), len(y_train)))
print('Training Labels=1 Count: {}  Training Labels=0 Count: {}'.format(sum(y_train == 1), sum(y_train == 0)))
print('Testing Features Count: {}  Testing Labels Count: {}'.format(len(X_test), len(y_test)))
```

    Training Features Count: 57936  Training Labels Count: 57936
    Training Labels=1 Count: 7431  Training Labels=0 Count: 50505
    Testing Features Count: 14485  Testing Labels Count: 14485


Since, as we found in data cleaning and just above, the labels are not balanced, I rebalance using under sampling:


```python
under_sampler = RandomUnderSampler()
X_res, y_res = under_sampler.fit_resample(X_train, y_train)

print('Resampled Training Features Count: {}  Resampled Training Labels Count: {}'.format(len(X_res), len(y_res)))
print('Resampled Training Labels=1 Count: {}  Resampled Training Labels=0 Count: {}'.format(sum(y_res == 1), sum(y_res == 0)))
```

    Resampled Training Features Count: 14862  Resampled Training Labels Count: 14862
    Resampled Training Labels=1 Count: 7431  Resampled Training Labels=0 Count: 7431


#### Decision Tree Classifier
Below, I start with training a simple Decision Tree Classifier


```python
dcs = DecisionTreeClassifier(max_depth=5, max_leaf_nodes=50)
dcs.fit(X_res, y_res)
```




<style>#sk-container-id-1 {color: black;background-color: white;}#sk-container-id-1 pre{padding: 0;}#sk-container-id-1 div.sk-toggleable {background-color: white;}#sk-container-id-1 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-1 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-1 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-1 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-1 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-1 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-1 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-1 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-1 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-1 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-1 div.sk-item {position: relative;z-index: 1;}#sk-container-id-1 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-1 div.sk-item::before, #sk-container-id-1 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-1 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-1 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-1 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-1 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-1 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-1 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-1 div.sk-label-container {text-align: center;}#sk-container-id-1 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-1 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>DecisionTreeClassifier(max_depth=5, max_leaf_nodes=50)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label sk-toggleable__label-arrow">DecisionTreeClassifier</label><div class="sk-toggleable__content"><pre>DecisionTreeClassifier(max_depth=5, max_leaf_nodes=50)</pre></div></div></div></div></div>




```python
# Calculate scores
y_pred = dcs.predict(X_test)
precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print('Precision: {:.2}  Recall: {:.2}  F1: {:.2}'.format(precision,recall,fbeta_score))
print('Accuracy: {:.2}'.format(accuracy))
print('ROC-AUC: {:.2}'.format(roc_auc))
```

    Precision: 0.3  Recall: 0.83  F1: 0.44
    Accuracy: 0.73
    ROC-AUC: 0.77


These scores are not horrible, but we can do better.

#### AdaBoost
I move to using an AdaBoost model to see if we can get some better scores.


```python
ada = AdaBoostClassifier(learning_rate=1)
ada.fit(X_res, y_res)
```




<style>#sk-container-id-2 {color: black;background-color: white;}#sk-container-id-2 pre{padding: 0;}#sk-container-id-2 div.sk-toggleable {background-color: white;}#sk-container-id-2 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-2 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-2 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-2 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-2 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-2 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-2 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-2 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-2 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-2 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-2 div.sk-item {position: relative;z-index: 1;}#sk-container-id-2 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-2 div.sk-item::before, #sk-container-id-2 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-2 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-2 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-2 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-2 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-2 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-2 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-2 div.sk-label-container {text-align: center;}#sk-container-id-2 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-2 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>AdaBoostClassifier(learning_rate=1)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label sk-toggleable__label-arrow">AdaBoostClassifier</label><div class="sk-toggleable__content"><pre>AdaBoostClassifier(learning_rate=1)</pre></div></div></div></div></div>




```python
# Calculate scores
y_pred = ada.predict(X_test)
precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print('Precision: {:.2}  Recall: {:.2}  F1: {:.2}'.format(precision,recall,fbeta_score))
print('Accuracy: {:.2}'.format(accuracy))
print('ROC-AUC: {:.2}'.format(roc_auc))
```

    Precision: 0.3  Recall: 0.81  F1: 0.44
    Accuracy: 0.74
    ROC-AUC: 0.77


I get similar scores with the AdaBoost model

#### XGBoost
Finally, I experiment with XGBoost, a model that was not covered in the course, however I've seen perform very well in practice.

I expended a bit more rigor with this model, including hyperparameter tuning using grid search.


```python
# Format the train/test data as XGBoost requires
dtrain = DMatrix(X_res, label=y_res)
dtest = DMatrix(X_test, label=y_test)
evallist = [(dtrain, 'train'), (dtest, 'eval')]
```

Below, I perform hyperparameter tuning using a Grid Search.  I'm using ROC-AUC as my scoring metric to determine the best model.  In all, 405 candidate models were trained to attempt to find the best model.


```python
# Set hyperparameters for search
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [5, 10, 15]
        }

# Instatiate the XGBoost classifier and begin grid search
xgb = XGBClassifier(learning_rate=0.02, objective='binary:logistic', nthread=1)
search = GridSearchCV(xgb, params, scoring='roc_auc', n_jobs=4, verbose=0)
search.fit(X_res, y_res)
```




<style>#sk-container-id-3 {color: black;background-color: white;}#sk-container-id-3 pre{padding: 0;}#sk-container-id-3 div.sk-toggleable {background-color: white;}#sk-container-id-3 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-3 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-3 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-3 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-3 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-3 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-3 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-3 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-3 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-3 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-3 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-3 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-3 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-3 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-3 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-3 div.sk-item {position: relative;z-index: 1;}#sk-container-id-3 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-3 div.sk-item::before, #sk-container-id-3 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-3 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-3 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-3 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-3 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-3 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-3 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-3 div.sk-label-container {text-align: center;}#sk-container-id-3 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-3 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-3" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=False, eval_metric=None,
                                     feature_types=None, gamma=None,
                                     gpu_id=None, grow_policy=None,
                                     importance_type=None,
                                     interaction_constraints=None,
                                     learning_rate=0.02, max_b...
                                     max_delta_step=None, max_depth=None,
                                     max_leaves=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     n_estimators=100, n_jobs=None, nthread=1,
                                     num_parallel_tree=None, predictor=None, ...),
             n_jobs=4,
             param_grid={&#x27;colsample_bytree&#x27;: [0.6, 0.8, 1.0],
                         &#x27;gamma&#x27;: [0.5, 1, 1.5, 2, 5], &#x27;max_depth&#x27;: [5, 10, 15],
                         &#x27;min_child_weight&#x27;: [1, 5, 10],
                         &#x27;subsample&#x27;: [0.6, 0.8, 1.0]},
             scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-3" type="checkbox" ><label for="sk-estimator-id-3" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=False, eval_metric=None,
                                     feature_types=None, gamma=None,
                                     gpu_id=None, grow_policy=None,
                                     importance_type=None,
                                     interaction_constraints=None,
                                     learning_rate=0.02, max_b...
                                     max_delta_step=None, max_depth=None,
                                     max_leaves=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     n_estimators=100, n_jobs=None, nthread=1,
                                     num_parallel_tree=None, predictor=None, ...),
             n_jobs=4,
             param_grid={&#x27;colsample_bytree&#x27;: [0.6, 0.8, 1.0],
                         &#x27;gamma&#x27;: [0.5, 1, 1.5, 2, 5], &#x27;max_depth&#x27;: [5, 10, 15],
                         &#x27;min_child_weight&#x27;: [1, 5, 10],
                         &#x27;subsample&#x27;: [0.6, 0.8, 1.0]},
             scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-4" type="checkbox" ><label for="sk-estimator-id-4" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.02, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, nthread=1, num_parallel_tree=None,
              predictor=None, ...)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-5" type="checkbox" ><label for="sk-estimator-id-5" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.02, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, nthread=1, num_parallel_tree=None,
              predictor=None, ...)</pre></div></div></div></div></div></div></div></div></div></div>



##### Best Model Scores


```python
print('Best ROC_AUC score: {:.2}'.format(search.best_score_))
print('Best parameters:')
print(search.best_params_)
```

    Best ROC_AUC score: 0.87
    Best parameters:
    {'colsample_bytree': 0.8, 'gamma': 2, 'max_depth': 10, 'min_child_weight': 1, 'subsample': 0.6}



```python
# Calculate scores
best_model = search.best_estimator_
y_pred = best_model.predict(X_test)
precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print('Precision: {:.2}  Recall: {:.2}  F1: {:.2}'.format(precision,recall,fbeta_score))
print('Accuracy: {:.2}'.format(accuracy))
print('ROC-AUC: {:.2}'.format(roc_auc))
```

    Precision: 0.33  Recall: 0.86  F1: 0.48
    Accuracy: 0.76
    ROC-AUC: 0.8


##### Feature Importance


```python
plot_importance(best_model)
```




    <AxesSubplot: title={'center': 'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_75_1.png)
    


##### ROC-AUC Curve


```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x7fc8f32fce80>




    
![png](output_77_1.png)
    


##### Confusion Matrix


```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fc8f2e55790>




    
![png](output_79_1.png)
    


#### XGBoost, Again
It looks like some of my engineered features (Urban, Rural, and the Census Divisions) are all relatively low performance compared to the others. I'll try the model again, with these features removed.


```python
# Prepare data
data = df.loc[:, df.columns.isin(['PovertyRate','MedianFamilyIncome','LowIncomeTracts','BlackPopShare','LOWIPopShare','KidsPopShare','SeniorsPopShare','WhitePopShare','AsianPopShare','NHOPIPopShare','AIANPopShare','OMultirPopShare','HispanicPopShare','HUNVPopShare','SNAPPopShare'])]
# Assign labels
labels = df[label]
```

Using the prepared data, I perform a train/test split, with 80% training data and 20% test data:


```python
# Train test split
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels)
print('Training Features Count: {}  Training Labels Count: {}'.format(len(X_train), len(y_train)))
print('Training Labels=1 Count: {}  Training Labels=0 Count: {}'.format(sum(y_train == 1), sum(y_train == 0)))
print('Testing Features Count: {}  Testing Labels Count: {}'.format(len(X_test), len(y_test)))
```

    Training Features Count: 57936  Training Labels Count: 57936
    Training Labels=1 Count: 7431  Training Labels=0 Count: 50505
    Testing Features Count: 14485  Testing Labels Count: 14485


Since, as we found in data cleaning and just above, the labels are not balanced, I rebalance using under sampling:


```python
under_sampler = RandomUnderSampler()
X_res, y_res = under_sampler.fit_resample(X_train, y_train)

print('Resampled Training Features Count: {}  Resampled Training Labels Count: {}'.format(len(X_res), len(y_res)))
print('Resampled Training Labels=1 Count: {}  Resampled Training Labels=0 Count: {}'.format(sum(y_res == 1), sum(y_res == 0)))
```

    Resampled Training Features Count: 14862  Resampled Training Labels Count: 14862
    Resampled Training Labels=1 Count: 7431  Resampled Training Labels=0 Count: 7431



```python
# Format the train/test data as XGBoost requires
dtrain = DMatrix(X_res, label=y_res)
dtest = DMatrix(X_test, label=y_test)
evallist = [(dtrain, 'train'), (dtest, 'eval')]
```

Below, I perform hyperparameter tuning using a Grid Search.  I'm using ROC-AUC as my scoring metric to determine the best model.  In all, 405 candidate models were trained to attempt to find the best model.


```python
# Set hyperparameters for search
params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.5, 1, 1.5, 2, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [5, 10, 15]
        }

# Instatiate the XGBoost classifier and begin grid search
xgb = XGBClassifier(learning_rate=0.02, objective='binary:logistic', nthread=1)
search_wo_eng = GridSearchCV(xgb, params, scoring='roc_auc', n_jobs=4, verbose=0)
search_wo_eng.fit(X_res, y_res)
```




<style>#sk-container-id-4 {color: black;background-color: white;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: "▸";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: "▾";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: "";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: "";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id="sk-container-id-4" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>GridSearchCV(estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=False, eval_metric=None,
                                     feature_types=None, gamma=None,
                                     gpu_id=None, grow_policy=None,
                                     importance_type=None,
                                     interaction_constraints=None,
                                     learning_rate=0.02, max_b...
                                     max_delta_step=None, max_depth=None,
                                     max_leaves=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     n_estimators=100, n_jobs=None, nthread=1,
                                     num_parallel_tree=None, predictor=None, ...),
             n_jobs=4,
             param_grid={&#x27;colsample_bytree&#x27;: [0.6, 0.8, 1.0],
                         &#x27;gamma&#x27;: [0.5, 1, 1.5, 2, 5], &#x27;max_depth&#x27;: [5, 10, 15],
                         &#x27;min_child_weight&#x27;: [1, 5, 10],
                         &#x27;subsample&#x27;: [0.6, 0.8, 1.0]},
             scoring=&#x27;roc_auc&#x27;)</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item sk-dashed-wrapped"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-6" type="checkbox" ><label for="sk-estimator-id-6" class="sk-toggleable__label sk-toggleable__label-arrow">GridSearchCV</label><div class="sk-toggleable__content"><pre>GridSearchCV(estimator=XGBClassifier(base_score=None, booster=None,
                                     callbacks=None, colsample_bylevel=None,
                                     colsample_bynode=None,
                                     colsample_bytree=None,
                                     early_stopping_rounds=None,
                                     enable_categorical=False, eval_metric=None,
                                     feature_types=None, gamma=None,
                                     gpu_id=None, grow_policy=None,
                                     importance_type=None,
                                     interaction_constraints=None,
                                     learning_rate=0.02, max_b...
                                     max_delta_step=None, max_depth=None,
                                     max_leaves=None, min_child_weight=None,
                                     missing=nan, monotone_constraints=None,
                                     n_estimators=100, n_jobs=None, nthread=1,
                                     num_parallel_tree=None, predictor=None, ...),
             n_jobs=4,
             param_grid={&#x27;colsample_bytree&#x27;: [0.6, 0.8, 1.0],
                         &#x27;gamma&#x27;: [0.5, 1, 1.5, 2, 5], &#x27;max_depth&#x27;: [5, 10, 15],
                         &#x27;min_child_weight&#x27;: [1, 5, 10],
                         &#x27;subsample&#x27;: [0.6, 0.8, 1.0]},
             scoring=&#x27;roc_auc&#x27;)</pre></div></div></div><div class="sk-parallel"><div class="sk-parallel-item"><div class="sk-item"><div class="sk-label-container"><div class="sk-label sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-7" type="checkbox" ><label for="sk-estimator-id-7" class="sk-toggleable__label sk-toggleable__label-arrow">estimator: XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.02, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, nthread=1, num_parallel_tree=None,
              predictor=None, ...)</pre></div></div></div><div class="sk-serial"><div class="sk-item"><div class="sk-estimator sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-8" type="checkbox" ><label for="sk-estimator-id-8" class="sk-toggleable__label sk-toggleable__label-arrow">XGBClassifier</label><div class="sk-toggleable__content"><pre>XGBClassifier(base_score=None, booster=None, callbacks=None,
              colsample_bylevel=None, colsample_bynode=None,
              colsample_bytree=None, early_stopping_rounds=None,
              enable_categorical=False, eval_metric=None, feature_types=None,
              gamma=None, gpu_id=None, grow_policy=None, importance_type=None,
              interaction_constraints=None, learning_rate=0.02, max_bin=None,
              max_cat_threshold=None, max_cat_to_onehot=None,
              max_delta_step=None, max_depth=None, max_leaves=None,
              min_child_weight=None, missing=nan, monotone_constraints=None,
              n_estimators=100, n_jobs=None, nthread=1, num_parallel_tree=None,
              predictor=None, ...)</pre></div></div></div></div></div></div></div></div></div></div>



##### Best Model Scores


```python
print('Best ROC_AUC score: {:.2}'.format(search_wo_eng.best_score_))
print('Best parameters:')
print(search_wo_eng.best_params_)
```

    Best ROC_AUC score: 0.85
    Best parameters:
    {'colsample_bytree': 0.8, 'gamma': 1.5, 'max_depth': 10, 'min_child_weight': 1, 'subsample': 0.6}



```python
# Calculate scores
best_model_wo_eng = search_wo_eng.best_estimator_
y_pred = best_model_wo_eng.predict(X_test)
precision, recall, fbeta_score, support = precision_recall_fscore_support(y_test, y_pred, pos_label=1, average='binary')
accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print('Precision: {:.2}  Recall: {:.2}  F1: {:.2}'.format(precision,recall,fbeta_score))
print('Accuracy: {:.2}'.format(accuracy))
print('ROC-AUC: {:.2}'.format(roc_auc))
```

    Precision: 0.31  Recall: 0.84  F1: 0.45
    Accuracy: 0.73
    ROC-AUC: 0.78


##### Feature Importance


```python
plot_importance(best_model_wo_eng)
```




    <AxesSubplot: title={'center': 'Feature importance'}, xlabel='F score', ylabel='Features'>




    
![png](output_93_1.png)
    


##### ROC-AUC Curve


```python
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)
display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
display.plot()
```




    <sklearn.metrics._plot.roc_curve.RocCurveDisplay at 0x7fc8f2fc1df0>




    
![png](output_95_1.png)
    


##### Confusion Matrix


```python
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
```




    <sklearn.metrics._plot.confusion_matrix.ConfusionMatrixDisplay at 0x7fc8f310ebe0>




    
![png](output_97_1.png)
    


#### Summary of Results and Analysis
Above, I have attempted four models on the data.  Two (Decision Tree Classifier and AdaBoost), we have see in class, and two attempts at XGBoost models, using grid search to identify the best hyperparameters for both models.

I used ROC-AUC as my primary metric because it takes into account both true positive and false positive rates.

For both the standard Decision Tree and AdaBoost, the ROC-AUC score was relatively low.

However, for the XGBoost model with both the Urban/Rural and Census Division features included, the ROC-AUC score was the highest.  For this reason, I would choose the XGBoost model that included the engineered features as the best model.  Even though this model is more complex than the one without the engineered features included, it did perform better than the one without (ROC-AUC=.8 for the full feature XGBoost model vs. 0.78 for the reduced feature model)

There are a few issues to be aware of that are apparent from some of the results plotted above:
- While the ROC-AUC score is good at .8 for the best XGBoost model, this isn't incredible high.  I suspect that this low score may be simply due to that this data is noisy, real-world data.
- In reviewing the confusion matrix, the model does appear to produce more false positives than would be desired on the test data (3192).  This conclusion is supported by the low ROC-AUC score, as well as the low precision score.

### Discussion and Conclusion

#### Learnings and Takeaways
From the Exploratory Data Analysis, I was able to confirm other reporting that there are indeed correlations between the demographic makeup of a community and whether that community is likely to have low income, low access to affordable, healthy food sources. It was interesting to see some of these relationships we intuitively expect play out very clearly in the data.

In addition, the fact that machine learning methods recognize these relationships further indicate that there is a correlation between demographics and the opportunities available in a community.

Successfully recognizing these relationships is critical so that communities can identify and address the root causes of these issues.

#### What Didn't Work
While the models performed better than I initially expected they would, they aren't perfect.  I suspect a probably cause of that is that this data is real-world data that is inherently messy.

#### Ways to Improve
There are a number of changes I would attempt given more time:
- I would like to explore more with the features and feature engineering of the model.  For instance, adding states as features ionstead of just Census Divisions, exploring other additional data, etc.
- The problems contributing to food deserts and poverty in America is complex.  Political climate, community involvement, local government support, etc all play a part.  Exploring these complexities deeper, and bringing related data into modelling could further improve the ability to accurately model.
- I would like to explore other, different models.  I wonder if additional grid search parameters, deep learning methods, or other methods might perform better.


```python

```
