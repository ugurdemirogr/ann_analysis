import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, accuracy_score
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


data = pd.read_excel(r'') # Tırnak işareti içine verisetinizi ekleyin.

X = StandardScaler().fit_transform(df.drop('', axis=1)) #tırnak işareti içine kendi veri setinizin çıkış değişkenini ekleyin.
y = LabelEncoder().fit_transform(df['']) #tırnak işareti içine kendi veri setinizin çıkış değişkenini ekleyin.

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.52, random_state=42)   #Parametreleri kendi verisetinize göre ayarlaryın.
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.38, random_state=42)

model = Sequential([ #Parametreleri kendi verisetinize göre ayarlaryın.
    Dense(15, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
y_pred = model.predict(X_test)
conf_matrix = confusion_matrix(y_test, (y_pred > 0.5).astype(int))
report = classification_report(y_test, (y_pred > 0.5).astype(int), output_dict=True)

precision = report['1']['precision']
recall = report['1']['recall']
f1_score = report['1']['f1-score']

all_accuracies = []
accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
all_accuracies.append(accuracy)
avg_accuracy = np.mean(all_accuracies)
conf_matrix = confusion_matrix(y_test, (y_pred > 0.5).astype(int))
class_report = classification_report(y_test, (y_pred > 0.5).astype(int), zero_division=1)

all_conf_matrices, all_roc_aucs = [], []
total_conf_matrix = np.zeros((2, 2))
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

all_accuracies.append(accuracy)
all_roc_aucs.append(roc_auc)

print(f"Classification Report:\n{classification_report(y_test, (y_pred > 0.5).astype(int))}\n{'=' * 40}\n")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Validation Accuracy: {avg_accuracy:.4f}")

# Save the false positive rate (fpr) and true positive rate (tpr) for logistic regression
fpr_lr, tpr_lr, _ = roc_curve(y_test_binary, y_prob)

# Save the false positive rate (fpr) and true positive rate (tpr) for neural network
fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, lw=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

