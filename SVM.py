# Això no va i no sé perquè, esteu convidats a intentar-ho fer anar.

import numpy as np
from sklearn import svm
import pandas as pd
from sklearn import metrics

# S'ha de veure decission trees (et dona interpretabilitat), svm, gaussaian noseque
# La llibreria pot fer dirèctament un gràfic.

# Aquest és el directori local al meu ordinador, no funcionarà en general.
input_file = "C:\\Users\\marcg\\Desktop\\Programes python\\Datathons\\2025\\dataset.csv"

# Aquest serà el tamany del split per a fer l'entrenament. Recomano experimentar!
split_size = 0.7

df: pd.DataFrame = pd.read_csv(input_file, header = 0)

original_headers = list(df.columns.values)

df: pd.DataFrame = df._get_numeric_data()

numeric_headers = list(df.columns.values)

numpy_array: np.ndarray = df.to_numpy()

numeric_headers.reverse()
reverse_df: pd.DataFrame = df[numeric_headers]

# Ara tenim bé les dades, a partir d'aquí cal entrenar el decission tree.
transposed_array = numpy_array.transpose()

svc = svm.SVC(kernel='rbf')


x_values = transposed_array[0:-1].transpose()
y_values = df["target_variable"]
size = len(y_values)

x_train = x_values[0: int(size * split_size)]
y_train = y_values[0: int(size * split_size)]

print(x_train, "\n\n\n")
print(y_train, "\n\n\n")
result = svc.fit(x_train, y_train)

print(result)

tests = 0
succeses = 0

"""
for i in range(int(size * split_size) + 1, size):
    training_example: np.ndarray = x_values[i]
    expected_result = y_values[i]
    training_example = training_example.reshape(1, -1)
    Calculted_result = result.predict(training_example)
    if i % 1000 == 0:
        print(f"Example {training_example}, with expectance {expected_result}, gave output {Calculted_result}")
    tests += 1
    if abs(Calculted_result - expected_result) <= 1e-9:
        succeses += 1
print(f"Final succes rate: {succeses / tests * 100}%")
"""

predictions = result.predict(x_values[int(size * split_size) + 1: size])
f1 = metrics.f1_score(y_values[int(size * split_size) + 1: size], predictions)

for i in range(0, len(predictions), 100):
    print(predictions[i], end=", ")
print(f"Final f1 score: {f1}")

