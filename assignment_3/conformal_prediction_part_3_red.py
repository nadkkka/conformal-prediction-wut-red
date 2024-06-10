import pandas as pd
from mapie.regression import MapieRegressor
from sklearn.ensemble import RandomForestRegressor
import catboost as cb
import numpy as np
import matplotlib.pyplot as plt
from numpy import random

#file_path = r'C:\Users\Kamil\Desktop\II rok\warsztaty\Allegro\niepotrzebne chyba\kaggle_raw_data\train_kaggle_raw.csv'
file_path = r'C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/kaggle_raw_data/train_kaggle_raw.csv'
train_data = pd.read_csv(file_path) # wczytanie danych z pliku - zbiór treningowy

#file_path_2 = r'C:\Users\Kamil\Desktop\II rok\warsztaty\Allegro\niepotrzebne chyba\kaggle_raw_data\test_kaggle_raw.csv'
file_path_2 = r'C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/kaggle_raw_data/test_kaggle_raw.csv'
test_data = pd.read_csv(file_path_2) # wczytanie danych z pliku - zbiór testowy

#wczytanie zbudowanego modelu
model = cb.CatBoostClassifier() 
model.load_model('C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/kaggle_raw_data/model.cbm')

X_train = train_data.drop(columns=['TARGET']) # dane treningowe - tylko features
target_train = train_data['TARGET'] # dane treningowe - targety

X_test = test_data.drop(columns=['TARGET'])

target_test = test_data['TARGET']


# Dopasowanie modelu kagglowego do danych treningowych
model.fit(X_train, target_train)

# prawdopodobienstwa, ze klienci naleza do danej klasy, czyli f(x), czyli scores
predicted_test = model.predict_proba(X_test)[:,0]

# Obliczanie residuów
residuals = np.abs(target_test - predicted_test)
n = len(train_data)
alpha = 0.2 

qhat = np.quantile(residuals,np.ceil((n+1)*(1-alpha))/n)
muhat, stdhat = (predicted_test, residuals.std())

lower_bound = muhat - qhat*stdhat
upper_bound = muhat + qhat*stdhat

bounds = np.array([lower_bound, upper_bound])
prob_pred_test = predicted_test/(1-predicted_test)
prob_test = np.array([prob_pred_test])

#wizualizacja
ind = X_test.index
k = 100
var = 'AMT_INCOME_TOTAL'

for i in range(k):

    selected = random.randint(0, len(X_test)-1)

    plt.plot(X_test[var][ind[selected]],muhat[ind[selected]], 'o', color = 'blue')
    plt.plot(X_test[var][ind[selected]],bounds[0, selected], 'o', color = 'r')
    plt.plot(X_test[var][ind[selected]],bounds[1, selected], 'o', color = 'purple')
    plt.fill_between((X_test[var][ind[selected]], X_test[var][ind[selected]]), bounds[0, selected], bounds[1, selected], color='gray', alpha=0.2)

plt.plot(X_test[var][ind[selected]],bounds[0, selected], label = 'Prediction of the lower bound', color = 'r')
plt.plot(X_test[var][ind[selected]],bounds[1, selected], label = 'Prediction of the upper bound', color = 'purple')
plt.plot(X_test[var][ind[selected]],muhat[ind[selected]], color = 'blue',label = 'Prediction')

plt.legend()
plt.xlabel(f'Value for selected variable')
plt.ylabel('Scores')
plt.title(f'Interval - conformal regression')
plt.show()