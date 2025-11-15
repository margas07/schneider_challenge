import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt


# Aquí s'ha de canviar el directori, això només va al meu ordinador!
input_file = "C:\\Users\\marcg\\Desktop\\Programes python\\Datathons\\2025\\dataset.csv"
df = pd.read_csv(input_file)


X = df.drop(columns=['target_variable'])
y = df['target_variable']

# Fem split en train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
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

# Mètriques
print("F1 score:", f1_score(y_test, y_pred, average="weighted"))
print(classification_report(y_test, y_pred))

# Visualització de l’arbre
plt.figure(figsize=(20,10))
plt.plot_tree(clf, feature_names=X.columns, class_names=np.unique(y).astype(str), filled=True, fontsize=None)
plt.savefig("C:\\Users\\marcg\\Desktop\\Programes python\\Datathons\\2025\\tree.png", dpi = 600)

print("\n\nFeature importances:")
print(clf.tree_.compute_feature_importances())
""" Output d'aquesta ultima línia:
[2.07137632e-01 3.59961519e-02 9.66195540e-02 1.37315566e-02
 6.77208135e-03 3.70339567e-04 8.49076844e-04 2.46160700e-01
 1.57464755e-01 4.12689462e-02 1.11583725e-01 2.05434531e-02
 1.92903020e-02 1.87834390e-04 2.10537544e-02 2.09701375e-02]
 
 Això es pot interpretar com a aquestes importàncies relatives segons paràmetres:
 -- Nom paràmetre --           -- Importància --
 id                                 0,20714
 product_A_sold_in_the_past         0,03599
 product_B_sold_in_the_past         0,09661
 product_A_recommended              0,01373
 product_A                          0,00677
 product_C                          0,00070
 product_D                          0,00085
 cust_hitrate                       0,24611
 cust_interactions                  0,15746
 cust_contracts                     0,04127
 opp_month                          0,11584
 opp_old                            0,02054
 competitor_Z                       0,01929
 competitor_X                       0,00019
 competitor_Y                       0,02105
 cust_in_iberia                     0,02970
 """
