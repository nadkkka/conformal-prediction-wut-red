import pandas as pd

#file_path = r'C:\Users\Kamil\Desktop\II rok\warsztaty\Allegro\data_red\train_red.csv'
file_path = r'C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/data_red/train_red.csv'
train_data = pd.read_csv(file_path) # wczytanie danych z pliku - zbiór treningowy

#file_path_2 = r'C:\Users\Kamil\Desktop\II rok\warsztaty\Allegro\data_red\test_red.csv'
file_path_2 = r'C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/data_red/test_red.csv'
test_data = pd.read_csv(file_path_2) # wczytanie danych z pliku - zbiór testowy

"""
print("Pierwsze 5 wierszy danych treningowych:")
print(train_data)
"""
import numpy as np
import matplotlib.pyplot as plt

# Wyznaczenie conformity scores
conformity_scores = np.abs(train_data["PRED"] - train_data["TARGET"])

alphas = [0.05, 0.2, 0.4]

# Kwantyle 1-alpha dla wyznaczonych alpha dla rozkładu conformity scores
quantiles = [np.quantile(conformity_scores, 1 - alpha) for alpha in alphas]

# Histogram naszych conformity scores
plt.hist(conformity_scores, bins=20, alpha=0.2, color='g', edgecolor='black')
plt.xlabel('Conformity Scores')
plt.ylabel('')
plt.title('Rozkład conformity scores')

# Nanosimy kwantyle dla 1-alpha
colors = ['r', 'b', 'm']
for quantile, alpha, color in zip(quantiles, alphas, colors):
    plt.axvline(x=quantile, color=color, linestyle='--', label=f'1-alpha={1-alpha}, alpha={alpha}')
plt.legend()
plt.show()

