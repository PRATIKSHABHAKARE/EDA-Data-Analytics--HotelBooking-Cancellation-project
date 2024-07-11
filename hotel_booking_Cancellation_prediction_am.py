#!/usr/bin/env python
# coding: utf-8

# <p style = "font-size : 42px; color : #393e46 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #00adb5; border-radius: 5px 5px;"><strong>Hotel Booking Cancellation EDA and Prediction</strong></p>

# <a id = '0'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Table of Contents</strong></p> 
# 
# * [EDA](#2.0)
#     * [From where the most guests are coming ?](#2.1)
#     * [How much do guests pay for a room per night?](#2.2)
#     * [How does the price vary per night over the year?](#2.3)
#     * [Which are the most busy months?](#2.4)
#     * [How long do people stay at the hotels?](#2.5)
#     
# * [Data Pre Processing](#3.0)
# * [Model Building](#4.0)
#     * [Logistic Regression](#4.1)
# * [Models Comparison](#5.0)
# 

# In[16]:


pip install missingno


# In[17]:


pip install folium


# In[20]:


# importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from xgboost import XGBClassifier


import folium
from folium.plugins import HeatMap
import plotly.express as px

plt.style.use('fivethirtyeight')
get_ipython().run_line_magic('matplotlib', 'inline')
pd.set_option('display.max_columns', 32)


# In[21]:


# reading data
df = pd.read_csv('hotel_bookings.csv')
df.head()


# In[22]:


df.describe()


# In[23]:


df.info()


# In[24]:


# checking for null values 

null = pd.DataFrame({'Null Values' : df.isna().sum(), 'Percentage Null Values' : (df.isna().sum()) / (df.shape[0]) * (100)})
null


# In[25]:


# filling null values with zero

df.fillna(0, inplace = True)


# In[26]:


# visualizing null values
msno.bar(df)
plt.show()


# In[27]:


# adults, babies and children cant be zero at same time, so dropping the rows having all these zero at same time

'''1.the filter variable represents a condition that selects rows from the DataFrame df 
where all three of these conditions are true. This means that the filtered rows only contain cases where 
there are no children, adults, or babies in the given data.
2.The second line of code, df[filter], uses this filter variable to subset the DataFrame df. 
It selects only the rows that meet the specified filter condition, effectively filtering out rows that 
do not satisfy all three criteria of having zero children, zero adults, and zero babies.
'''
filter = (df.children == 0) & (df.adults == 0) & (df.babies == 0)
df[filter]


# In[28]:


'''In this line of code, df = df[~filter], the ~ operator is used to negate the filter condition. 
This means that the code is selecting all the rows in the df DataFrame that do not meet the filter condition.*/
'''
df = df[~filter]
df


# <a id = '2.0'></a>
# <p style = "font-size : 40px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Exploratory Data Analysis (EDA)</strong></p> 

# <a id = '2.1'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>From where the most guests are coming ?.</strong></p> 

# In[29]:


'''
Filtering for non-canceled bookings: The first line of code, df[df['is_canceled'] == 0], 
filters the DataFrame df to only include rows where the is_canceled column is equal to 0. 
This means that only bookings that were not canceled are selected.

Counting guests by country: The second line of code, ['country'].value_counts(), 
counts the number of guests from each country in the filtered DataFrame. 
This creates a Series object with the countries as index and the corresponding guest counts as values.

Resetting index and renaming columns: The third line of code, .reset_index(), 
converts the Series object into a DataFrame and resets the index to create a new column. 
The fourth line of code, .columns = ['country', 'No of guests'], renames the columns of the resulting DataFrame 
to country and No of guests, respectively.

In summary, this code is calculating the number of non-canceled guests from each country 
in the DataFrame df and organizing the results into a new DataFrame with the columns country and No of guests. 
This information could be useful for analyzing guest demographics and identifying patterns in guest origin.
'''
country_wise_guests = df[df['is_canceled'] == 0]['country'].value_counts().reset_index()
country_wise_guests.columns = ['country', 'No of guests']
country_wise_guests


# In[30]:


basemap = folium.Map()
'''This line of code initializes a base map using the folium library. 
folium is an open-source Python library for creating interactive maps. 
The basemap object will serve as the foundation for displaying the guest data on a geographical representation.'''

guests_map = px.choropleth(country_wise_guests, locations = country_wise_guests['country'],
                           color = country_wise_guests['No of guests'], hover_name = country_wise_guests['country'])

'''This line of code creates a choropleth map using the px.choropleth() function from the plotly library. 
plotly is a data visualization library that provides interactive charting capabilities. 
The guests_map object will represent the visualization of guest data on the base map.

The px.choropleth() function takes several parameters:

country_wise_guests: This is the DataFrame containing the guest data, including country names and guest counts.
locations: This specifies the column in the DataFrame that represents the geographical locations (countries) 
           for which the data is being visualized. In this case, it's the country column.
color: This specifies the column in the DataFrame that represents the values to be mapped to color intensity. 
       In this case, it's the No of guests column, indicating the intensity of shading based on the number of guests from each country.
hover_name: This specifies the column in the DataFrame that will be displayed when hovering over a particular 
            location on the map. In this case, it's the country column, providing the country name as a tooltip.
'''
guests_map.show()

'''In short it creates an interactive choropleth map using folium and plotly libraries to visualize 
the geographical distribution of guest data. The map represents the number of guests from each country using 
varying color intensities, and provides tooltips with country names when hovering over specific locations.
'''


# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>People from all over the world are staying in these two hotels. Most guests are from Portugal and other countries in Europe.</strong></p> 

# <a id = '2.2'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>How much do guests pay for a room per night?</strong></p> 

# In[31]:


df.head()


# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>Both hotels have different room types and different meal arrangements.Seasonal factors are also important, So the prices varies a lot.</strong></p> 

# '''This line of code creates a box plot using the px.box() function from the plotly library. 
# The box plot summarizes the distribution of adr (average daily rate) for each reserved_room_type and 
# distinguishes between different hotels using color.
# 
# breakdown of the parameters used in px.box():
# 
# data_frame: This specifies the DataFrame containing the data for the box plot. In this case, 
#             it's the data DataFrame created in the previous step.
# x: This specifies the column in the DataFrame that represents the categorical variable for grouping the box plots. 
#     In this case, it's the reserved_room_type column.
# y: This specifies the column in the DataFrame that represents the numerical variable to be summarized in the box plots. 
#     In this case, it's the adr column.
# color: This specifies the column in the DataFrame that determines the color of each box plot. 
#     In this case, it's the hotel column, assigning different colors to different hotels.
# template: This specifies the visual theme for the box plot. In this case, it's set to plotly_dark for a dark color scheme.
# 
# The px.box() function generates the box plot and displays it in a web browser. 
# The box plot shows the median, quartiles, and outliers for each reserved room
# '''
# 

# In[32]:


data = df[df['is_canceled'] == 0]
#only bookings that were not canceled are selected. The filtered DataFrame is stored in the variable data.

px.box(data_frame = data, x = 'reserved_room_type', y = 'adr', color = 'hotel', template = 'plotly_dark')


# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>The figure shows that the average price per room depends on its type and the standard deviation.</strong></p> 

# <a id = '2.3'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>How does the price vary per night over the year?</strong></p> 

# In[33]:


data_resort = df[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 0)]
data_city = df[(df['hotel'] == 'City Hotel') & (df['is_canceled'] == 0)]


# In[34]:


resort_hotel = data_resort.groupby(['arrival_date_month'])['adr'].mean().reset_index()
resort_hotel


# In[35]:


city_hotel=data_city.groupby(['arrival_date_month'])['adr'].mean().reset_index()
city_hotel


# In[36]:


final_hotel = resort_hotel.merge(city_hotel, on = 'arrival_date_month')
final_hotel.columns = ['month', 'price_for_resort', 'price_for_city_hotel']
final_hotel


# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>Now we observe here that month column is not in order, and if we visualize we will get improper conclusions.</strong></p>

# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>So, first we have to provide right hierarchy to month column.</strong></p>

# In[37]:


get_ipython().system('pip install sort-dataframeby-monthorweek')

get_ipython().system('pip install sorted-months-weekdays')


# In[38]:


import sort_dataframeby_monthorweek as sd

def sort_month(df, column_name):
    return sd.Sort_Dataframeby_Month(df, column_name)


# In[39]:


final_prices = sort_month(final_hotel, 'month')
final_prices


# In[40]:


plt.figure(figsize = (17, 8))

px.line(final_prices, x = 'month', y = ['price_for_resort','price_for_city_hotel'],
        title = 'Room price per night over the Months', template = 'plotly_dark')


# <p style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>This plot clearly shows that prices in the Resort Hotel are much higher during the summer and prices of city hotel varies less and is most expensive during Spring and Autumn .</strong></p>

# <a id = '2.4'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>Which are the most busy months?</strong></p> 

# In[41]:


resort_guests = data_resort['arrival_date_month'].value_counts().reset_index()
resort_guests.columns=['month','no of guests']
resort_guests


# In[42]:


city_guests = data_city['arrival_date_month'].value_counts().reset_index()
city_guests.columns=['month','no of guests']
city_guests


# In[43]:


final_guests = resort_guests.merge(city_guests,on='month')
final_guests.columns=['month','no of guests in resort','no of guest in city hotel']
final_guests


# In[44]:


final_guests = sort_month(final_guests,'month')
final_guests


# In[45]:


px.line(final_guests, x = 'month', y = ['no of guests in resort','no of guest in city hotel'],
        title='Total no of guests per Months', template = 'plotly_dark')


# <ul>
#     <li style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>The City hotel has more guests during spring and autumn, when the prices are also highest, In July and August there are less visitors, although prices are lower.</strong></li>
#     <li style = "font-size : 20px; color : #810000 ; font-family : 'Comic Sans MS'; "><strong>Guest numbers for the Resort hotel go down slighty from June to September, which is also when the prices are highest. Both hotels have the fewest guests during the winter.</strong></li>
# </ul>

# <a id = '2.5'></a>
# <p style = "font-size : 35px; color : #34656d ; font-family : 'Comic Sans MS'; "><strong>How long do people stay at the hotels?</strong></p> 

# In[46]:


filter = df['is_canceled'] == 0
data = df[filter]
data.head()


# In[47]:


data['total_nights'] = data['stays_in_weekend_nights'] + data['stays_in_week_nights']
data.head()


# In[48]:


stay = data.groupby(['total_nights', 'hotel']).agg('count').reset_index()
stay = stay.iloc[:, :3]
stay = stay.rename(columns={'is_canceled':'Number of stays'})
stay


# In[49]:


px.bar(data_frame = stay, x = 'total_nights', y = 'Number of stays', color = 'hotel', barmode = 'group',
        template = 'plotly_dark')


# The bar chart can reveal several insights about the booking patterns and hotel preferences among the guests.
# 
# Stay Duration and Hotel Preference: The height of each bar represents the total number of stays for a particular hotel. Taller bars indicate that more guests chose to stay at that hotel. By comparing the bar heights, you can identify which hotels are more popular among the guests.
# 
# Average Stay Length: The grouping of bars based on hotel provides an overview of the average stay length for each hotel. Hotels with longer bars suggest that guests tend to stay for a longer duration at that hotel, while shorter bars indicate shorter stays on average.
# 
# Hotel Diversity and Guest Preferences: The number of bars represents the diversity of hotels chosen by guests. If there are many bars with varying heights, it suggests that guests have a wider range of preferences and choose different hotels based on various factors.
# 
# Overall Stay Patterns: The overall pattern of the bar chart can reveal trends in guest preferences for stay durations. If the bars are evenly distributed, it suggests that guests have varying preferences for the length of their stays. However, if there is a clear pattern of taller or shorter bars, it indicates a preference for longer or shorter stays, respectively.
# 
# Hotel Popularity and Market Share: By analyzing the relative heights of the bars, you can assess the relative popularity of each hotel and their share in the overall market. Taller bars indicate a higher share of bookings, while shorter bars suggest a lower share.
# 
# Impact of Hotel Characteristics: If information about hotel characteristics, such as amenities, location, or price range, is available, you can correlate those factors with the stay duration and hotel preference observed in the bar chart. This can provide insights into how these factors influence guest decisions.
# 
# Potential for Segmentation: The bar chart can help identify potential customer segments based on stay duration preferences. For instance, hotels with longer stays may attract guests seeking extended vacations or business travelers, while those with shorter stays may cater to weekend getaways or shorter business trips.
# 
# Marketing Strategies and Hotel Positioning: Hotel managers can use the bar chart to inform their marketing strategies and positioning. Understanding guest preferences for stay durations can help tailor promotions and offerings to attract the desired customer segments.
# 
# Seasonal Trends and Fluctuations: Analyzing the bar chart over different time periods can reveal seasonal trends in booking patterns and stay durations. This information can help hotels anticipate demand and optimize their operations accordingly.
# 
# Comparative Analysis and Benchmarking: By comparing the bar chart with data from other hotels or industry benchmarks, hoteliers can assess their performance and identify areas for improvement. This can help them gain a competitive edge and attract more guests.

# In[54]:


df_numerical_features = df.select_dtypes(include='number')


# <a id = '3.0'></a>
# <p style = "font-size : 40px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Data Pre Processing</strong></p> 

# In[55]:


plt.figure(figsize = (24, 12))

corr = df_numerical_features.corr()
#This line calculates the correlation matrix for the DataFrame df. 
#The correlation matrix represents the pairwise correlations between all variables in the DataFrame.

sns.heatmap(corr, annot = True, linewidths = 1)
'''
This line creates a heatmap using the sns.heatmap() function from the seaborn library. 
The seaborn library provides advanced data visualization functions. 
The heatmap() function takes the correlation matrix as input and generates a color-coded representation of the correlations.

annot: This parameter sets whether to display correlation values as annotations on the heatmap. 
annot=True shows the correlation values in each cell.

linewidths: This parameter sets the width of the cell borders in the heatmap. 
linewidths=1 provides a visible border around each cell.'''
plt.show()


# The heatmap allows you to visually assess the strength and direction of correlations between variables. 
# Darker and more saturated colors indicate stronger correlations, while lighter colors indicate weaker correlations. 
# The color hue indicates whether the correlation is positive (red) or negative (blue).

# In[56]:


correlation = df_numerical_features.corr()['is_canceled'].abs().sort_values(ascending = False)
correlation


# In[57]:


# dropping columns that are not useful

useless_col = ['days_in_waiting_list', 'arrival_date_year', 'arrival_date_year', 'assigned_room_type', 'booking_changes',
               'reservation_status', 'country', 'days_in_waiting_list']

df.drop(useless_col, axis = 1, inplace = True)


# In[58]:


df.head()


# In[59]:


# creating numerical and categorical dataframes

cat_cols = [col for col in df.columns if df[col].dtype == 'O']
cat_cols


# This condition checks if the data type of the current column (col) is 'O'. 
# The 'O' data type in Pandas represents object data, which typically includes categorical values like strings or text.
# 
# The resulting cat_cols list contains the names of all columns in the DataFrame df that have an object data type, 
# indicating that they are categorical columns. This information is useful for further analysis and 
# processing of categorical data.

# In[60]:


cat_df = df[cat_cols]
cat_df.head()


# In[61]:


cat_df['reservation_status_date'] = pd.to_datetime(cat_df['reservation_status_date'])

cat_df['year'] = cat_df['reservation_status_date'].dt.year
cat_df['month'] = cat_df['reservation_status_date'].dt.month
cat_df['day'] = cat_df['reservation_status_date'].dt.day


# In[62]:


cat_df.drop(['reservation_status_date','arrival_date_month'] , axis = 1, inplace = True)


# In[63]:


cat_df.head()


# In[64]:


# printing unique values of each column
for col in cat_df.columns:
    print(f"{col}: \n{cat_df[col].unique()}\n")


# In[65]:


# encoding categorical variables

cat_df['hotel'] = cat_df['hotel'].map({'Resort Hotel' : 0, 'City Hotel' : 1})

cat_df['meal'] = cat_df['meal'].map({'BB' : 0, 'FB': 1, 'HB': 2, 'SC': 3, 'Undefined': 4})

cat_df['market_segment'] = cat_df['market_segment'].map({'Direct': 0, 'Corporate': 1, 'Online TA': 2, 'Offline TA/TO': 3,
                                                           'Complementary': 4, 'Groups': 5, 'Undefined': 6, 'Aviation': 7})

cat_df['distribution_channel'] = cat_df['distribution_channel'].map({'Direct': 0, 'Corporate': 1, 'TA/TO': 2, 'Undefined': 3,
                                                                       'GDS': 4})

cat_df['reserved_room_type'] = cat_df['reserved_room_type'].map({'C': 0, 'A': 1, 'D': 2, 'E': 3, 'G': 4, 'F': 5, 'H': 6,
                                                                   'L': 7, 'B': 8})

cat_df['deposit_type'] = cat_df['deposit_type'].map({'No Deposit': 0, 'Refundable': 1, 'Non Refund': 3})

cat_df['customer_type'] = cat_df['customer_type'].map({'Transient': 0, 'Contract': 1, 'Transient-Party': 2, 'Group': 3})

cat_df['year'] = cat_df['year'].map({2015: 0, 2014: 1, 2016: 2, 2017: 3})


# In[66]:


cat_df.head()


# In[67]:


num_df = df.drop(columns = cat_cols, axis = 1)
num_df.drop('is_canceled', axis = 1, inplace = True)
num_df


# In[68]:


num_df.var()


# In[69]:


# normalizing numerical variables

num_df['lead_time'] = np.log(num_df['lead_time'] + 1)
num_df['arrival_date_week_number'] = np.log(num_df['arrival_date_week_number'] + 1)
num_df['arrival_date_day_of_month'] = np.log(num_df['arrival_date_day_of_month'] + 1)
num_df['agent'] = np.log(num_df['agent'] + 1)
num_df['company'] = np.log(num_df['company'] + 1)
num_df['adr'] = np.log(num_df['adr'] + 1)


# In[70]:


num_df.var()


# In[71]:


num_df['adr'] = num_df['adr'].fillna(value = num_df['adr'].mean())


# In[72]:


num_df.head()


# # Following code is preparing the data for training a machine learning model.

# line 1: combines two DataFrames, cat_df and num_df, into a single DataFrame named X. 
# 
# The axis=1 parameter indicates that the DataFrames are concatenated horizontally, meaning that the columns are combined while maintaining their rows.
# 
# The cat_df DataFrame likely contains categorical data, such as strings or text, while the num_df DataFrame likely contains numerical data, such as integers or floats. 
# 
# By combining these two DataFrames, you create a single feature matrix X that includes both types of data.
# 
# Line2:This line extracts the target variable 'is_canceled' from the original DataFrame df. 
# The target variable represents the outcome you want to predict, in this case, whether a booking is canceled or not. 
# The y variable will be used to train the machine learning model to predict the value of 'is_canceled' for new bookings.

# # The resulting X DataFrame contains the features used for prediction, and the y variable contains the corresponding target values. This data preparation step is essential for training and evaluating machine learning models.
# 

# In[73]:


X = pd.concat([cat_df, num_df], axis = 1)
y = df['is_canceled']


# In[74]:


X.shape, y.shape


# In[75]:


# splitting data into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# In[76]:


X_train.head()


# In[77]:


X_test.head()


# In[78]:


y_train.head(), y_test.head()


# <a id = '4.0'></a>
# <p style = "font-size : 45px; color : #34656d ; font-family : 'Comic Sans MS'; text-align : center; background-color : #f9b208; border-radius: 5px 5px;"><strong>Model Building</strong></p> 

# <a id = '4.1'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Logistic Regression</strong></p> 

# In[79]:


lr = LogisticRegression()
lr.fit(X_train, y_train)

y_pred_lr = lr.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
conf = confusion_matrix(y_test, y_pred_lr)
clf_report = classification_report(y_test, y_pred_lr)

print(f"Accuracy Score of Logistic Regression is : {acc_lr}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# In machine learning, a classification report is a performance evaluation metric that summarizes the accuracy of a classification model. It provides a detailed breakdown of the model's performance on different classes or categories in the target variable.
# 
# The classification report typically includes the following metrics:
# 
# Precision: The proportion of positive predictions that are actually correct.
# 
# Recall: The proportion of actual positive cases that are correctly identified.
# 
# F1-score: The harmonic mean of precision and recall, providing a balanced measure of both.
# 
# Support: The total number of true instances in each class.
# 
# The classification report is typically presented as a table, with each metric calculated for each class and an overall summary for all classes. It allows you to assess the model's performance on individual classes and identify areas where the model may be struggling.

# <a id = '4.2'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>KNN</strong></p> 

# In[80]:


knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

y_pred_knn = knn.predict(X_test)

acc_knn = accuracy_score(y_test, y_pred_knn)
conf = confusion_matrix(y_test, y_pred_knn)
clf_report = classification_report(y_test, y_pred_knn)

print(f"Accuracy Score of KNN is : {acc_knn}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.3'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Decision Tree Classifier</strong></p> 

# In[81]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train, y_train)

y_pred_dtc = dtc.predict(X_test)

acc_dtc = accuracy_score(y_test, y_pred_dtc)
conf = confusion_matrix(y_test, y_pred_dtc)
clf_report = classification_report(y_test, y_pred_dtc)

print(f"Accuracy Score of Decision Tree is : {acc_dtc}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.4'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Random Forest Classifier</strong></p> 

# In[82]:


rd_clf = RandomForestClassifier()
rd_clf.fit(X_train, y_train)

y_pred_rd_clf = rd_clf.predict(X_test)

acc_rd_clf = accuracy_score(y_test, y_pred_rd_clf)
conf = confusion_matrix(y_test, y_pred_rd_clf)
clf_report = classification_report(y_test, y_pred_rd_clf)

print(f"Accuracy Score of Random Forest is : {acc_rd_clf}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.7'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>XgBoost Classifier</strong></p> 

# In[83]:


xgb = XGBClassifier(booster = 'gbtree', learning_rate = 0.1, max_depth = 5, n_estimators = 180)
xgb.fit(X_train, y_train)

y_pred_xgb = xgb.predict(X_test)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
conf = confusion_matrix(y_test, y_pred_xgb)
clf_report = classification_report(y_test, y_pred_xgb)

print(f"Accuracy Score of Ada Boost Classifier is : {acc_xgb}")
print(f"Confusion Matrix : \n{conf}")
print(f"Classification Report : \n{clf_report}")


# <a id = '4.11'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>ANN</strong></p> 

# In[84]:


from tensorflow.keras.utils import to_categorical

X = pd.concat([cat_df, num_df], axis = 1)
y = to_categorical(df['is_canceled'])


# In[85]:


# splitting data into training set and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)


# In[86]:


import keras
from keras.layers import Dense
from keras.models import Sequential

model  = Sequential()
model.add(Dense(100, activation = 'relu', input_shape = (26, )))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(2, activation = 'sigmoid'))
model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model_history = model.fit(X_train, y_train, validation_data = (X_test, y_test),
                          epochs = 100)


# In[87]:


plt.figure(figsize = (12, 6))

train_loss = model_history.history['loss']
val_loss = model_history.history['val_loss'] 
epoch = range(1, 101)

loss = pd.DataFrame({'train_loss' : train_loss, 'val_loss' : val_loss})

px.line(data_frame = loss, x = epoch, y = ['val_loss', 'train_loss'], title = 'Training and Validation Loss',
        template = 'plotly_dark')


# In[88]:


plt.figure(figsize = (12, 6))

train_acc = model_history.history['accuracy']
val_acc = model_history.history['val_accuracy'] 
epoch = range(1, 101)


accuracy = pd.DataFrame({'train_acc' : train_acc, 'val_acc' : val_acc})

px.line(data_frame = accuracy, x = epoch, y = ['val_acc', 'train_acc'], title = 'Training and Validation Accuracy',
        template = 'plotly_dark')


# In[89]:


acc_ann = model.evaluate(X_test, y_test)[1]

print(f'Accuracy of model is {acc_ann}')


# <a id = '5.0'></a>
# <p style = "font-size : 34px; color : #fed049 ; font-family : 'Comic Sans MS'; text-align : center; background-color : #007580; border-radius: 5px 5px;"><strong>Models Comparison</strong></p> 

# In[90]:


models = pd.DataFrame({
    'Model' : ['Logistic Regression', 'KNN', 'Decision Tree Classifier', 'Random Forest Classifier','XgBoost','ANN'],
    'Score' : [acc_lr, acc_knn, acc_dtc, acc_rd_clf, acc_xgb, acc_ann]
})


models.sort_values(by = 'Score', ascending = False)


# In[91]:


px.bar(data_frame = models, x = 'Score', y = 'Model', color = 'Score', template = 'plotly_dark', title = 'Models Comparison')


# <p style = "font-size : 30px; color : #03506f ; font-family : 'Comic Sans MS'; "><strong>We got accuracy score of 99.5% which is quite impresive.</strong></p> 

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




