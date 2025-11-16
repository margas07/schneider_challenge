import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_graphviz
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt


# Aquí s'ha de canviar el directori, això només va al meu ordinador!
input_file = "C:\\Users\\marcg\\Desktop\\Programes python\\Datathons\\2025\\dataset.csv"
df = pd.read_csv(input_file)


X = df.drop(columns=['target_variable', 'id', 'product_A', 'product_C', 'product_D'])
y = df['target_variable']

# Fem split en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Creem i entrenem el DecisionTree, els parametres els he generat a la força.
clf = DecisionTreeClassifier(
    criterion="entropy",   # es pot fer "entropy" o "gini"
    max_depth=None,        # limita profunditat per evitar overfitting
    min_samples_leaf=1,
    min_samples_split=2,
    random_state=42
)
clf.fit(X_train, y_train)


# Prediccions
y_pred = clf.predict(X_test)

# Visualització de l’arbre
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=np.unique(y).astype(str), filled=True, fontsize=None)
plt.savefig("C:\\Users\\marcg\\Desktop\\Programes python\\Datathons\\2025\\tree.png", dpi = 600)

print("\n\nFeature importances:")
print(clf.tree_.compute_feature_importances())

# Exporta el arbre, si cal:
# export_graphviz(clf, "C:\\Users\\marcg\\Desktop\\Programes python\\Datathons\\2025\\tree.DOT")
