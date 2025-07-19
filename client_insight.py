import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# Funkcje pomocnicze

def kategoria_dochodu(x):
    if x < 3817:
        return "niski"
    elif x < 5850:
        return "średni"
    else:
        return "wysoki"

def czy_potencjalnie_niezadowolony(wiersz):
    # jeśli ktoś dał niską ocenę i komentarz zawiera negatywne słowo – uznajemy, że jest potencjalnie niezadowolony
    negatywne_slowa = ['słaba', 'okropne', 'nie polecam', 'zła', 'tragiczna', 'rozczarowany']
    komentarz = wiersz['Komentarz'].lower()
    if wiersz['Zadowolony'] == 0 and any(sl in komentarz for sl in negatywne_slowa):
        return True
    return False

# Wczytanie danych

df = pd.read_csv('dane_do_analizy.csv')

print(df.head())
print(df.info())
print(df.isnull().sum())

# Czyszczenie danych

df = df.drop(columns=['ID'])
df['Komentarz'] = df['Komentarz'].fillna('brak')
print(df.isnull().sum())

# Tworzymy nową kolumnę binarną: 1 jeśli ocena 4 lub 5, inaczej 0
df['Zadowolony'] = df['Zadowolenie (1-5)'].apply(lambda x: 1 if x >= 4 else 0)

# Inżynieria cech

df['czy_senior'] = df['Wiek'].apply(lambda x: x > 65)
df['Kategoria_dochodu'] = df['Dochód (zł)'].apply(kategoria_dochodu)
df['Potencjalnie_niezadowolony'] = df.apply(czy_potencjalnie_niezadowolony, axis=1)

print(df['Dochód (zł)'].describe())
print(df[df['Wiek'] > 100])  # podejrzany wiek
print(df['Potencjalnie_niezadowolony'].value_counts())

# Proste analizy wizualne

# sprawdzamy czy seniorzy są mniej zadowoleni
sns.countplot(x='czy_senior', hue='Zadowolony', data=df)
plt.title('Zadowolenie a wiek (senior/niesenior)')
plt.xlabel('Czy senior?')
plt.ylabel('Liczba osób')
plt.legend(title='Zadowolony')
plt.show()

# zadowolenie w zależności od dochodu
sns.countplot(x='Kategoria_dochodu', hue='Zadowolony', data=df)
plt.title('Zadowolenie a dochód')
plt.xlabel('Kategoria dochodu')
plt.ylabel('Liczba osób')
plt.legend(title='Zadowolony')
plt.show()

# zadowolenie wg płci
sns.countplot(x='Płeć', hue='Zadowolony', data=df)
plt.title('Zadowolenie a płeć')
plt.xlabel('Płeć')
plt.ylabel('Liczba osób')
plt.legend(title='Zadowolony')
plt.show()

print(df['Płeć'].value_counts()) # sprawdzamy rozkład płci (dane generowane losowo)

# Korelacje

df['czy_senior_num'] = df['czy_senior'].astype(int) # True/False zamieniamy na 1/0

kolumny_korelacja = ['Wiek', 'Dochód (zł)', 'Zadowolenie (1-5)', 'Zadowolony', 'czy_senior_num']
macierz_korelacji = df[kolumny_korelacja].corr()
print(macierz_korelacji)

# wykres korelacji
plt.figure(figsize=(8, 6))
sns.heatmap(macierz_korelacji, annot=True, cmap='coolwarm')
plt.title('Macierz korelacji')
plt.show()

# Prosty model predykcyjny

# chcemy przewidzieć zadowolenie na podstawie wieku i dochodu
X = df[['Wiek', 'Dochód (zł)']]
y = df['Zadowolony']

# dzielimy dane – 80% trenujemy, 20% testujemy
X_tren, X_test, y_tren, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# trenujemy model regresji logistycznej
model = LogisticRegression()
model.fit(X_tren, y_tren)

# przewidujemy i sprawdzamy, jak dobrze działa
y_pred = model.predict(X_test)

print("Dokładność modelu:", accuracy_score(y_test, y_pred))
print("Raport klasyfikacji:")
print(classification_report(y_test, y_pred))
