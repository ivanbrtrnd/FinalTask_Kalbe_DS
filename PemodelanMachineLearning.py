#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[1]:


import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import autocorrelation_plot

import warnings
warnings.filterwarnings('ignore')


# # Read CSV Data

# In[2]:


df_customer = pd.read_csv('Case Study - Customer.csv', delimiter=';')
df_product = pd.read_csv('Case Study - Product.csv', delimiter=';')
df_store = pd.read_csv('Case Study - Store.csv', delimiter=';')
df_transaction = pd.read_csv('Case Study - Transaction.csv', delimiter=';')


# # Data Cleansing

# ## Customer

# In[3]:


df_customer.head


# In[4]:


df_customer.duplicated().sum()


# In[5]:


df_customer.isnull().sum()


# In[6]:


df_customer[df_customer['Marital Status'].isnull()]


# In[7]:


df_customer.dropna(subset=['Marital Status'], inplace=True)


# In[8]:


df_customer.dtypes


# In[9]:


df_customer['Income'] = df_customer['Income'].replace('[,]', '.', regex=True).astype('float')


# ## Product

# In[10]:


df_product.head()


# In[11]:


df_product.duplicated().sum()


# In[12]:


df_product.isnull().sum()


# In[13]:


df_product.dtypes


# ## Store

# In[14]:


df_store.head()


# In[15]:


df_store.duplicated().sum()


# In[16]:


df_store.isnull().sum()


# In[17]:


df_store.dtypes


# In[18]:


df_store['Latitude'] = df_store['Latitude'].replace('[,]', '.', regex=True).astype('float')
df_store['Longitude'] = df_store['Longitude'].replace('[,]', '.', regex=True).astype('float')


# ## Transaction

# In[19]:


df_transaction.head()


# In[20]:


df_transaction.duplicated().sum()


# In[21]:


df_transaction.isnull().sum()


# In[22]:


df_transaction.dtypes


# In[23]:


df_transaction['Date'] = pd.to_datetime(df_transaction['Date'])


# In[24]:


df_transaction[df_transaction.duplicated(['TransactionID'], keep=False)]


# In[25]:


df_transaction.drop_duplicates('TransactionID', keep='last', inplace=True)


# # Data Merging

# In[26]:


df_merge = pd.merge(df_transaction, df_customer, on=['CustomerID'])
df_merge = pd.merge(df_merge, df_product.drop(columns=['Price']), on=['ProductID'])
df_merge = pd.merge(df_merge, df_store, on=['StoreID'])


# In[27]:


df_merge.head()


# # Model Machine Learning Regression (Time Series)

# In[28]:


df_regresi = df_merge.groupby(['Date']).agg({
    'Qty' : 'sum'
}).reset_index()


# In[29]:


decomposed = seasonal_decompose(df_regresi.set_index('Date'))

plt.figure(figsize=(10, 10))

plt.subplot(311)
decomposed.trend.plot(ax=plt.gca(), color='#59A14F')
plt.title('Trend')
plt.subplot(312)
decomposed.seasonal.plot(ax=plt.gca(), color='#59A14F')
plt.title('Seasonality')
plt.subplot(313)
decomposed.resid.plot(ax=plt.gca(), color='#59A14F')
plt.title('Residual')

plt.tight_layout()


# ## Stationarity Test / Augmented Dickey-Fuller Test

# In[30]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(df_regresi['Qty'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))


# In[31]:


plt.figure(figsize=(20,5))
sns.lineplot(data=df_regresi, x=df_regresi['Date'], y=df_regresi['Qty'], color='#59A14F')


# ## Finding the Value of d

# In[32]:


plt.rcParams.update({'figure.figsize':(9,7), 'figure.dpi':120})

fig, (ax1, ax2, ax3) = plt.subplots(3)

ax1.plot(df_regresi.Qty, color='#59A14F'); ax1.set_title('Original Series'); ax1.axes.xaxis.set_visible(False)

ax2.plot(df_regresi.Qty.diff(), color='#59A14F'); ax2.set_title('1st Order Differencing'); ax2.axes.xaxis.set_visible(False)

ax3.plot(df_regresi.Qty.diff().diff(), color='#59A14F'); ax3.set_title('2nd Order Differencing')

plt.show()


# In[33]:


from statsmodels.graphics.tsaplots import plot_acf
fig, (ax1, ax2, ax3) = plt.subplots(3)
plot_acf(df_regresi.Qty, color='#59A14F', ax=ax1); ax1.axes.xaxis.set_visible(False)
plot_acf(df_regresi.Qty.diff().dropna(), color='#59A14F', ax=ax2); ax2.axes.xaxis.set_visible(False)
plot_acf(df_regresi.Qty.diff().diff().dropna(), color='#59A14F', ax=ax3);


# ## Finding the Value of p

# In[34]:


from statsmodels.graphics.tsaplots import plot_pacf
plot_pacf(df_regresi.Qty.diff().dropna(), color='#59A14F');


# ## Finding the Value of q

# In[35]:


plot_acf(df_regresi.Qty.diff().dropna(), color='#59A14F');


# In[54]:


from pmdarima.arima import auto_arima
model = auto_arima(df_train, trace=True, error_action='ignore', suppress_warnings=True)
model.fit(df_train)
forecast = model.predict(n_periods=len(df_test))
forecast = pd.DataFrame(forecast, index=df_test.index, columns=['Predictions'])
print(forecast)


# In[36]:


cut_off = round(df_regresi.shape[0]*0.8)
df_train = df_regresi[:cut_off]
df_test = df_regresi[cut_off:].reset_index(drop=True)
df_train.shape, df_test.shape


# In[37]:


plt.figure(figsize=(20,5))
sns.lineplot(data=df_train, x=df_train['Date'], y=df_train['Qty'], color='#F28E2B')
sns.lineplot(data=df_test, x=df_test['Date'], y=df_test['Qty'], color='#59A14F')


# In[38]:


autocorrelation_plot(df_regresi['Qty'], color='#59A14F')


# In[39]:


def rmse(y_actual, y_pred):
    """
    function to calculate RMSE
    """
    
    print(f'RMSE value {mean_squared_error(y_actual, y_pred)**0.5}')
    
def eval(y_actual, y_pred):
    """
    function to eval machine learning modelling
    """
    
    rmse(y_actual, y_pred)
    print(f'MAE value {mean_absolute_error(y_actual, y_pred)}')


# In[40]:


y = df_train['Qty']

df_train = df_train.set_index('Date')
df_test = df_test.set_index('Date')


# In[41]:


ARIMAmodel = ARIMA(y, order=(50, 0, 2))
ARIMAmodel = ARIMAmodel.fit()

y_pred = ARIMAmodel.get_forecast(len(df_test))

y_pred_df = y_pred.conf_int()
y_pred_df['predictions'] = ARIMAmodel.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = df_test.index
y_pred_out = y_pred_df['predictions']
eval(df_test['Qty'], y_pred_out)

plt.figure(figsize=(20, 5))
plt.plot(df_train['Qty'], color='#F28E2B')
plt.plot(df_test['Qty'], color='#59A14F')
plt.plot(y_pred_out, color='#555555', label='ARIMA Predictions')
plt.legend()


# # Model Machine Learning Clustering

# In[42]:


df_merge.head()


# In[43]:


df_merge.corr()


# In[44]:


df_cluster = df_merge.groupby(['CustomerID']).agg({
    'TransactionID' : 'count',
    'Qty' : 'sum',
    'TotalAmount' : 'sum'
}).reset_index()


# In[45]:


df_cluster.head()


# In[46]:


data_cluster = df_cluster.drop(columns=['CustomerID'])

data_cluster_normalize = preprocessing.normalize(data_cluster)


# In[47]:


data_cluster_normalize


# In[48]:


K = range(2, 8)
fits = []
score = []

for k in K:
    model = KMeans(n_clusters = k, random_state=0, n_init='auto').fit(data_cluster_normalize)
    
    fits.append(model)
    
    score.append(silhouette_score(data_cluster_normalize, model.labels_, metric='euclidean'))


# In[49]:


sns.lineplot(x=K, y=score, color='#59A14F')


# In[50]:


fits[1]


# In[51]:


df_cluster['cluster_label'] = fits[1].labels_


# In[52]:


df_cluster.groupby(['cluster_label']).agg({
    'CustomerID' : 'count',
    'TransactionID' : 'mean',
    'Qty' : 'mean',
    'TotalAmount' : 'mean',
})


# In[53]:


import matplotlib.colors as mcolors

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x = np.array(df_cluster['TransactionID'])
y = np.array(df_cluster['Qty'])
z = np.array(df_cluster['TotalAmount'])

cmap = mcolors.LinearSegmentedColormap.from_list("", ["#F28E2B", "#59A14F", "#4E79A7"])
ax.scatter(x, y, z, c=df_cluster['cluster_label'], cmap=cmap)

plt.show()


# In[ ]:




