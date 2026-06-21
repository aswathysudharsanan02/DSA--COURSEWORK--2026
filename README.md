# Beer Servings Alcohol Prediction App

## Project Overview

This project predicts Total Litres of Pure Alcohol Consumption using machine learning.

## Dataset Overview

Beer Servings Dataset.It contains alcohol consumption statistics for countries around the world. The dataset provides information on the average consumption of different alcoholic beverages and the corresponding total litres of pure alcohol consumed per person. 

Number of Records: 193 countries
Number of Input Features: 5
Data Types:
Numerical Features: Beer Servings, Spirit Servings, Wine Servings
Categorical Features: Country, Continent
Target Variable: Total Litres of Pure Alcohol

## Input Features

* Country(Name)
* Beer Servings (Average number of beer servings consumed per person annually)
* Spirit Servings(Average number of spirit servings consumed per person annually)
* Wine Servings(Average number of wine servings consumed per person annually)
* Continent(Continent to which the country belongs)

## Target Variable

* Total Litres of pure Alcohol consumed per person per year. This is the variable predicted by the machine learning models. 


## Models Used

1. Linear Regression
2. Random Forest Regressor

## Hyperparameter Tuning

GridSearchCV was used to tune the Random Forest model using different values of n_estimators and max_depth.

## Best Model

Linear Regression

R² Score: 0.9468

## Web Application

The prediction application was developed using Streamlit and includes interactive visualizations and prediction functionality.

## Conclusion

In this project, the Beer Servings Dataset was analyzed to predict the **Total Litres of Pure Alcohol Consumption** using machine learning techniques. Data preprocessing was performed to handle missing values and prepare the dataset for model development.Two regression models, **Linear Regression** and **Random Forest Regressor**, were developed and evaluated using the R² score as the performance metric. Hyperparameter tuning was performed on the Random Forest model using GridSearchCV to identify the optimal model parameters.The results showed that **Linear Regression achieved the highest R² score of approximately 0.9468**. Therefore, Linear Regression was selected as the final model for deployment.
A user-friendly web application was developed using Streamlit, allowing users to enter country, continent, beer servings, spirit servings, and wine servings to predict the total litres of pure alcohol consumption. Four Interactive visualizations were also included to provide insights into alcohol consumption patterns across different continents and countries.
The project successfully demonstrated the complete machine learning workflow, including data preprocessing, model development, hyperparameter tuning, model evaluation, visualization, and deployment.
