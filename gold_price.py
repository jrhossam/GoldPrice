import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from scipy.stats import mode
import folium

# 1. قراءة البيانات
file_path = r"D:\\kkd\\gld_price_data.csv"
data = pd.read_csv(file_path)

# 2. اختيار جزء صغير من البيانات (على سبيل المثال، 10% فقط)
data_sample = data.sample(frac=0.1, random_state=42)

# استعراض أول 5 صفوف
print("First 5 rows of the dataset:")
print(data_sample.head())

# 3. تنظيف البيانات
# التعامل مع القيم المفقودة
print("\nMissing values before cleaning:")
print(data_sample.isnull().sum())

# استبدال القيم الفارغة بـ NaN
data_sample = data_sample.fillna(np.nan)

# حذف الصفوف المكررة
data_sample = data_sample.drop_duplicates()

print("\nData after cleaning:")
print(data_sample.info())

# 4. التحليل الإحصائي
# تحديد الأعمدة الرقمية فقط
numeric_data = data_sample.select_dtypes(include=[np.number])

print("\nStatistical Analysis:")
print(numeric_data.describe())

# حساب المتوسط، الوسيط، القيمة الأكثر تكرارًا
mean_values = numeric_data.mean()
median_values = numeric_data.median()
mode_values = numeric_data.mode().iloc[0]

print("\nMean values:\n", mean_values)
print("\nMedian values:\n", median_values)
print("\nMode values:\n", mode_values)

# 5. التصورات الأولية
plt.figure(figsize=(8, 8))
correlation = numeric_data.corr()
sns.heatmap(correlation, annot=True, cmap='Blues', fmt='.2f')
plt.title("Heatmap of Correlations")
plt.show()

# توزيع القيم
plt.figure(figsize=(10, 6))
sns.histplot(data_sample['GLD'], kde=True, bins=30, color='blue')
plt.title("Distribution of GLD Prices")
plt.xlabel("GLD Price")
plt.ylabel("Frequency")
plt.show()

# 6. إزالة القيم الشاذة
Q1 = numeric_data.quantile(0.25)
Q3 = numeric_data.quantile(0.75)
IQR = Q3 - Q1
non_outliers = ~((numeric_data < (Q1 - 1.5 * IQR)) | (numeric_data > (Q3 + 1.5 * IQR))).any(axis=1)

# إنشاء نسخة من البيانات بعد إزالة القيم الشاذة
data_cleaned = data_sample[non_outliers].copy()

print("\nData after removing outliers:")
print(data_cleaned.describe())

# 7. التصنيف
# تحويل أسعار الذهب إلى فئات
Y_class = (data_cleaned['GLD'] > data_cleaned['GLD'].median()).astype(int)
X_class = data_cleaned.drop(['Date', 'GLD'], axis=1)

X_train_class, X_test_class, Y_train_class, Y_test_class = train_test_split(X_class, Y_class, test_size=0.2, random_state=2)

# Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train_class, Y_train_class)
Y_pred_class = rf_classifier.predict(X_test_class)

print("\nClassification Report:")
print(metrics.classification_report(Y_test_class, Y_pred_class))

print("\nConfusion Matrix:")
print(metrics.confusion_matrix(Y_test_class, Y_pred_class))

# Decision Tree Classifier
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_class, Y_train_class)

plt.figure(figsize=(12, 8))
plot_tree(dt_classifier, filled=True, feature_names=X_class.columns, class_names=['Low', 'High'])
plt.title("Decision Tree Classifier")
plt.show()

# 8. التجميع (Clustering)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_class)
kmeans = KMeans(n_clusters=3, random_state=0)
kmeans.fit(X_scaled)

# تعديل طريقة تعيين التصنيفات باستخدام loc لتجنب التحذير
data_cleaned.loc[:, 'Cluster'] = kmeans.labels_

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_cleaned['SPX'], y=data_cleaned['GLD'], hue=data_cleaned['Cluster'], palette='Set2', s=100, alpha=0.7)
plt.title("Clustering Visualization")
plt.xlabel("SPX")
plt.ylabel("GLD Price")
plt.show()

# 9. الانحدار
Y = data_cleaned['GLD']
X = data_cleaned.drop(['Date', 'GLD'], axis=1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

# Random Forest Regressor
rf_regressor = RandomForestRegressor(n_estimators=100)
rf_regressor.fit(X_train, Y_train)
Y_pred = rf_regressor.predict(X_test)

print("\nR^2 Score (Random Forest):", metrics.r2_score(Y_test, Y_pred))

# Decision Tree Regressor
dt_regressor = DecisionTreeRegressor()
dt_regressor.fit(X_train, Y_train)

plt.figure(figsize=(12, 8))
plot_tree(dt_regressor, filled=True, feature_names=X.columns)
plt.title("Decision Tree Regressor")
plt.show()

# 10. خريطة جغرافية (Map Visualization)
# إذا كانت هناك بيانات مواقع جغرافية (فرضية)
# إضافة بيانات عشوائية للموقع
data_cleaned['Latitude'] = np.random.uniform(low=24.0, high=32.0, size=len(data_cleaned))  # مثال عشوائي
data_cleaned['Longitude'] = np.random.uniform(low=30.0, high=35.0, size=len(data_cleaned))  # مثال عشوائي

# إنشاء الخريطة باستخدام مكتبة folium
m = folium.Map(location=[data_cleaned['Latitude'].mean(), data_cleaned['Longitude'].mean()], zoom_start=5)

# إضافة النقاط للموقع
for index, row in data_cleaned.iterrows():
    folium.CircleMarker(location=[row['Latitude'], row['Longitude']], radius=5, color='blue', fill=True).add_to(m)

# حفظ الخريطة في المسار الكامل
output_path = r"D:\\kkd\\map.html"  # قم بتحديد المسار المناسب
m.save(output_path)

# طباعة تأكيد
print(f"Map has been saved to {output_path}")
