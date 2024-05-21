import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from mapie.classification import MapieClassifier
from mapie.metrics import classification_coverage_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np

#file_path = r'C:\Users\Kamil\Desktop\II rok\warsztaty\Allegro\data_red\train_red.csv'
file_path = r'C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/data_red/train_red.csv'
train_data = pd.read_csv(file_path) # wczytanie danych z pliku - zbiór treningowy

#file_path_2 = r'C:\Users\Kamil\Desktop\II rok\warsztaty\Allegro\data_red\test_red.csv'
file_path_2 = r'C:/Users/nadia/OneDrive/Pulpit/studia/II stopień/warsztaty badawcze/Allegro/data_red/train_red.csv'
test_data = pd.read_csv(file_path_2) # wczytanie danych z pliku - zbiór testowy

#print("Pierwsze 5 wierszy danych treningowych:")
#print(train_data)

# Podzial danych
X_train = train_data.iloc[:, 6:]  # Od siódmej kolumny do końca - cechy znormalizowane 
y_train = train_data["TARGET"] # etykiety

X_test = test_data.iloc[:, 6:]   # Od siódmej kolumny do końca - cechy znormalizowane
y_test = test_data["TARGET"]

# Inicjalizacja klasyfikatora - TU NIE WIEM CO SIĘ DZIEJE (estimator inaczej na pewno - musimy korzystać z przewidywan, które już mamy w danych)
clf = RandomForestClassifier(random_state=42) # Tworzenie modelu Random Forest
clf.fit(X_train, y_train) # Trenowanie modelu
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)
y_pred_proba_max = np.max(y_pred_proba, axis=1)
cp = MapieClassifier(estimator=clf, cv="prefit", method="lac") # inicjacja klasyfikatora conformal prediction ("lac" to nowa nazwa dla "score")

# Dopasowanie estymatora do naszych danych
cp.fit(X_train, y_train)

# Conformal prediction na zbiorze testowym
alphas = [0.2, 0.1, 0.05]
y_pred, y_set = cp.predict(X_test, alphas)  # alpha określa poziom ufności

# Funkcja rysująca wykres rozkładu klasyfikacji

def plot_scores(alphas, scores, quantiles, bins=10):
    colors = {0: "#1f77b4", 1: "#ff7f0e", 2: "#2ca02c"}
    plt.figure(figsize=(7, 5))
    plt.hist(scores, bins=bins)
    for i, quantile in enumerate(quantiles):
        plt.vlines(
            x=quantile,
            ymin=0,
            ymax = max(np.histogram(scores, bins=bins)[0]) * 1.1,
            color=colors[i],
            ls="dashed",
            label=f"alpha = {alphas[i]}"
        )
    plt.title("Distribution of scores")
    plt.legend()
    plt.xlabel("Scores")
    plt.ylabel("Count")
    plt.show()

scores = cp.conformity_scores_
n = len(cp.conformity_scores_)
quantiles = cp.quantiles_
alphas = [0.5, 0.1, 0.05]
plot_scores(alphas, scores, quantiles, 3)

alpha = 0.2
cut_off_thresholds = {
    0: np.percentile(scores[y_train == 0], 100 * (1 - alpha)),
    1: np.percentile(scores[y_train == 1], 100 * (1 - alpha))
}
lower_bounds = y_set[:, 0]
upper_bounds = y_set[:, 1]
