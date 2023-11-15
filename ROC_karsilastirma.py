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
from sklearn.linear_model import LogisticRegression

# İlk olarak ANN modeli ekleniyor.
data = pd.read_excel(r'') # Tırnak işareti içine verisetinizi ekleyin.

X = StandardScaler().fit_transform(df.drop('OAP_Aralıklı', axis=1))
y = LabelEncoder().fit_transform(df['OAP_Aralıklı'])

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.52, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.38, random_state=42)

model = Sequential([    # Kendi ANN modelinizi tanımlayın.
    Dense(15, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(X_train, y_train, epochs=1000, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0) #Earlystopping olduğu için 1000 epoch çalışmayabilir.
y_pred = model.predict(X_test)

fpr_nn, tpr_nn, _ = roc_curve(y_test, y_pred)
roc_auc_nn = auc(fpr_nn, tpr_nn)

# Burası regresyon analizini içerir.
data = pd.read_excel(r' ') # ' ' işareti olan yere kendi verisetinizin dizinini yazın.

X = data.iloc[:, 0:8] # giriş verisi
y = data.iloc[:, 8] # Çıkış verisi
X = pd.get_dummies(X, columns=['']) #Kategorik değişkenleri belirtebilirsiniz.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_lr = LogisticRegression()
model_lr.fit(X_train, y_train)

y_pred_prob_lr = model_lr.predict_proba(X_test)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_prob_lr)
roc_auc_lr = auc(fpr_lr, tpr_lr)

# Tek bir grafikte ROC eğrilerini çizdirilir. ANN ve Regrosyon ROC eğrisi karşılaştırılır.
plt.figure(figsize=(8, 6))
plt.plot(fpr_nn, tpr_nn, lw=2, label=f'NN ROC Curve (AUC = {roc_auc_nn:.4f})')
plt.plot(fpr_lr, tpr_lr, color='darkorange', lw=2, label=f'Logistic Regression ROC Curve (AUC = {roc_auc_lr:.4f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()
