import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#file_path = r'C:\Users\Kamil\Desktop\II rok\warsztaty\Allegro\data_red\train_red.csv'
file_path = r'C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/data_red/train_red.csv'
train_data = pd.read_csv(file_path) # wczytanie danych z pliku - zbiór treningowy

#file_path_2 = r'C:\Users\Kamil\Desktop\II rok\warsztaty\Allegro\data_red\test_red.csv'
file_path_2 = r'C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/data_red/test_red.csv'
test_data = pd.read_csv(file_path_2) # wczytanie danych z pliku - zbiór testowy

alpha = 0.2
n = len(train_data)

residuals = train_data['TARGET'] - train_data['PRED']
scores = abs(residuals)/residuals.std()

qhat = np.quantile(scores,np.ceil((n+1)*(1-alpha))/n)
muhat, stdhat = (train_data['PRED'], residuals.std())
pred_test = test_data['PRED']
prob_pred_test = pred_test/(1-pred_test)

test_data['lower_bound']=prob_pred_test-qhat
test_data['upper_bound']=prob_pred_test+qhat

prob_test = np.array([prob_pred_test])

def skala_log10(x):
    return np.log10(x/(1-x))


plt.plot(skala_log10(prob_pred_test),prob_pred_test, label = 'Predykcja')
plt.fill_between(skala_log10(prob_pred_test), test_data['lower_bound'],test_data['upper_bound'], color='b', alpha=.25, label = f'Przedział predykcji {np.round(1-alpha,2)}')
plt.plot(skala_log10(prob_pred_test),test_data['upper_bound'], label = 'Górna wartość przedziału predykcji', color = 'g', linestyle = '-', alpha = 0.5)
plt.plot(skala_log10(prob_pred_test),test_data['lower_bound'], label = 'Dolna wartość przedziału predykcji', color = 'b', linestyle = '-', alpha = 0.5)
plt.legend()
plt.xlabel('Skala logistyczna')
plt.ylabel('Wyniki')
plt.title('Przedziały conformal regresji dla wyników predykcji')
plt.show()
