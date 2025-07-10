---
title: "Machine Learning cơ bản cho người mới bắt đầu"
published: 2025-06-29
description: "Hướng dẫn chi tiết về machine learning từ lý thuyết đến thực hành"
tags: [Machine Learning, AI, Python, Scikit-learn, Data Science]
category: "AI & Machine Learning"
draft: false
lang: "vi"
---

## Mục lục

1. [Các loại Machine Learning](#1-các-loại-machine-learning)
    - [Supervised Learning (Học có giám sát)](#supervised-learning-học-có-giám-sát)
    - [Unsupervised Learning (Học không giám sát)](#unsupervised-learning-học-không-giám-sát)
2. [Các thuật toán cơ bản](#2-các-thuật-toán-cơ-bản)
    - [Linear Regression (Hồi quy tuyến tính)](#linear-regression-hồi-quy-tuyến-tính)
    - [Logistic Regression (Hồi quy logistic)](#logistic-regression-hồi-quy-logistic)
    - [Decision Tree (Cây quyết định)](#decision-tree-cây-quyết-định)
3. [Xử lý dữ liệu](#3-xử-lý-dữ-liệu)
    - [Data Preprocessing](#data-preprocessing)
    - [Feature Engineering](#feature-engineering)
4. [Cross-Validation](#4-cross-validation)
5. [Hyperparameter Tuning](#5-hyperparameter-tuning)
6. [Model Evaluation](#6-model-evaluation)
7. [Deep Learning với TensorFlow](#7-deep-learning-với-tensorflow)
8. [Kết luận](#kết-luận)

# Machine Learning cơ bản cho người mới bắt đầu 

Machine Learning (ML) là một nhánh của Artificial Intelligence (AI) cho phép máy tính học và cải thiện từ dữ liệu mà không cần được lập trình rõ ràng. Trong bài viết này, tôi sẽ giới thiệu những khái niệm cơ bản và thực hành với Python.

## 1. Các loại Machine Learning

### Supervised Learning (Học có giám sát)

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Ví dụ: Dự đoán giá nhà
# Dữ liệu mẫu
data = {
    'diện_tích': [100, 150, 200, 250, 300],
    'số_phòng': [2, 3, 3, 4, 4],
    'giá': [500, 750, 1000, 1250, 1500]
}

df = pd.DataFrame(data)
X = df[['diện_tích', 'số_phòng']]
y = df['giá']

# Chia dữ liệu
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Huấn luyện model
model = LinearRegression()
model.fit(X_train, y_train)

# Dự đoán
y_pred = model.predict(X_test)

# Đánh giá
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse:.2f}')
print(f'R²: {r2:.2f}')
```

### Unsupervised Learning (Học không giám sát)

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Ví dụ: Phân cụm khách hàng
# Dữ liệu mẫu
customer_data = np.array([
    [25, 50000],  # [tuổi, thu nhập]
    [30, 60000],
    [35, 70000],
    [40, 80000],
    [45, 90000],
    [50, 100000]
])

# Chuẩn hóa dữ liệu
scaler = StandardScaler()
scaled_data = scaler.fit_transform(customer_data)

# Phân cụm
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Visualize
plt.scatter(customer_data[:, 0], customer_data[:, 1], 
           c=clusters, cmap='viridis')
plt.xlabel('Tuổi')
plt.ylabel('Thu nhập')
plt.title('Phân cụm khách hàng')
plt.show()
```

## 2. Các thuật toán cơ bản

### Linear Regression (Hồi quy tuyến tính)

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Tạo dữ liệu mẫu
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 0.5

# Huấn luyện model
model = LinearRegression()
model.fit(X, y)

# Dự đoán
y_pred = model.predict(X)

# Visualize
plt.scatter(X, y, alpha=0.5, label='Dữ liệu thực')
plt.plot(X, y_pred, color='red', label='Dự đoán')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

print(f'Coefficient: {model.coef_[0][0]:.2f}')
print(f'Intercept: {model.intercept_[0]:.2f}')
```

### Logistic Regression (Hồi quy logistic)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Dữ liệu mẫu: Dự đoán spam email
X = np.array([
    [1, 0, 1, 0],  # [có_link, có_số_điện_thoại, có_từ_khẩn, có_viết_hoa]
    [0, 1, 0, 1],
    [1, 1, 1, 1],
    [0, 0, 0, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
])

y = np.array([1, 1, 1, 0, 0, 0])  # 1: spam, 0: không spam

# Huấn luyện model
model = LogisticRegression()
model.fit(X, y)

# Dự đoán
y_pred = model.predict(X)

# Đánh giá
accuracy = accuracy_score(y, y_pred)
print(f'Độ chính xác: {accuracy:.2f}')
print(classification_report(y, y_pred))
```

### Decision Tree (Cây quyết định)

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Dữ liệu mẫu: Dự đoán mua hàng
data = {
    'tuổi': [25, 35, 45, 55, 25, 35, 45, 55],
    'thu_nhập': ['thấp', 'thấp', 'cao', 'cao', 'cao', 'thấp', 'cao', 'thấp'],
    'mua_hàng': [0, 0, 1, 1, 1, 0, 1, 0]
}

df = pd.DataFrame(data)

# Chuyển đổi dữ liệu
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['thu_nhập_encoded'] = le.fit_transform(df['thu_nhập'])

X = df[['tuổi', 'thu_nhập_encoded']]
y = df['mua_hàng']

# Huấn luyện model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# Visualize cây quyết định
plt.figure(figsize=(10, 8))
plot_tree(model, feature_names=['Tuổi', 'Thu nhập'], 
          class_names=['Không mua', 'Mua'], filled=True)
plt.show()
```

## 3. Xử lý dữ liệu

### Data Preprocessing

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Tạo dữ liệu mẫu với missing values
data = {
    'tuổi': [25, 30, None, 35, 40],
    'thu_nhập': [50000, 60000, 70000, None, 90000],
    'thành_phố': ['Hà Nội', 'TP.HCM', 'Đà Nẵng', 'Hà Nội', None],
    'mua_hàng': [1, 0, 1, 0, 1]
}

df = pd.DataFrame(data)

# Xử lý missing values
# Số: thay bằng median
imputer_num = SimpleImputer(strategy='median')
df[['tuổi', 'thu_nhập']] = imputer_num.fit_transform(df[['tuổi', 'thu_nhập']])

# Categorical: thay bằng mode
imputer_cat = SimpleImputer(strategy='most_frequent')
df['thành_phố'] = imputer_cat.fit_transform(df[['thành_phố']])

# Encode categorical variables
le = LabelEncoder()
df['thành_phố_encoded'] = le.fit_transform(df['thành_phố'])

print("Dữ liệu sau khi xử lý:")
print(df)
```

### Feature Engineering

```python
# Tạo features mới
df['tuổi_nhóm'] = pd.cut(df['tuổi'], bins=[0, 30, 40, 100], 
                         labels=['trẻ', 'trung niên', 'cao tuổi'])

# One-hot encoding
df_encoded = pd.get_dummies(df, columns=['tuổi_nhóm'])

# Scaling
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_encoded[['tuổi', 'thu_nhập']])

print("Features sau khi engineering:")
print(df_encoded.head())
```

## 4. Cross-Validation

```python
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Sử dụng cross-validation để đánh giá model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# 5-fold cross-validation
scores = cross_val_score(model, X, y, cv=5)

print(f'Độ chính xác trung bình: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})')
```

## 5. Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Tìm hyperparameters tốt nhất
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best score: {grid_search.best_score_:.3f}')
```

## 6. Model Evaluation

```python
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns

# Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Thực tế')
plt.xlabel('Dự đoán')
plt.show()

# ROC Curve
y_pred_proba = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
auc = roc_auc_score(y_test, y_pred_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
```

## 7. Deep Learning với TensorFlow

```python
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# Tạo neural network đơn giản
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Huấn luyện
history = model.fit(X_train, y_train,
                    epochs=100,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0)

# Visualize training
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
```

## Kết luận

Machine Learning là một lĩnh vực rộng lớn và thú vị. Để thành công trong lĩnh vực này, bạn cần:

### 1. **Nền tảng vững chắc**
- Toán học: Linear Algebra, Calculus, Statistics
- Lập trình: Python, R, SQL
- Kiến thức domain: Hiểu rõ lĩnh vực ứng dụng

### 2. **Thực hành thường xuyên**
- Làm các project thực tế
- Tham gia competitions (Kaggle, etc.)
- Đọc papers và implement

### 3. **Cập nhật xu hướng**
- Deep Learning
- Reinforcement Learning
- AutoML
- MLOps

### 4. **Tools và Frameworks**
- **Scikit-learn**: ML cơ bản
- **TensorFlow/PyTorch**: Deep Learning
- **Pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

Hãy bắt đầu với những khái niệm cơ bản và dần dần tiến tới các thuật toán phức tạp hơn! 🚀 