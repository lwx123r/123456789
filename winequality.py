import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import time
data = pd.read_csv('data.csv')
print('数据集大小：', len(data))

pd.set_option('display.max_columns', None)
print("列名：\n", data.columns)
print("\n前5行数据：\n", data.head(5))
print("\n统计描述：\n", data.describe())

data.drop("id", axis=1, inplace=True)
#显示出三种不同葡萄酒的数量
plt.figure(figsize=(6, 4))
sns.countplot(x=data['class'], label='Count')
plt.title("class count")
plt.show()
#分析13种特征和种类之间的相关性，分析哪种成分是分类的特征标签
corr = data.corr()['class']
correlations = corr.sort_values(ascending=False)
print(correlations)

correlations.plot(kind='bar')#绘制按特征与种类相关系数从大到小的排列条形图# #plt.title('')
plt.show()

plt.figure(figsize=(10,6))
sns.heatmap(data.corr(), annot=True, fmt='.0%')
plt.title("heatmap")
plt.show()

print(abs(correlations) > 0.1)
#从所有成分中选择Alclinity_of_ash、Nonflavanoid phenols、Malicacid、Color intensity四个成分。


for column in data.columns:
    plt.figure(figsize=(8, 5))
    sns.boxplot(x='class', y=data[column], data=data)
    plt.title(f'{column}')
    plt.show()

#算法
x = data.drop('class', axis=1)
Y = data['class']

x_train1, x_test1, Y_train1, Y_test1 = train_test_split(x, Y, test_size=0.4, random_state=42)
pd.unique(Y)

model = SVC(C=1.0, kernel='linear', gamma=0.1)
model.fit(x_train1, Y_train1)
Y_test_pred1 = model.predict(x_test1)
Y_train_pred1 = model.predict(x_train1)
print("测试准确率SVC(pca前):", accuracy_score(Y_test1, Y_test_pred1))
print("训练准确率SVC(pca前)：", accuracy_score(Y_train1, Y_train_pred1))

cm = confusion_matrix(Y_test1, Y_test_pred1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("predicted class")
plt.ylabel("real class")
plt.title("SVC Confusion Matrix")
plt.show()


scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components=6)
x_pca = pca.fit_transform(x_scaled)
x_train, x_test, Y_train, Y_test = train_test_split(x_pca, Y, test_size=0.4, random_state=42)
pd.unique(Y)

model_pca2 = SVC(C=1.0, kernel='linear', gamma=0.1)
model_pca2.fit(x_train, Y_train)
Y_test_pred = model_pca2.predict(x_test)
Y_train_pred = model_pca2.predict(x_train)

start = time.time()
model_pca2.fit(x_train, Y_train)
svc_train_time = time.time() - start

start = time.time()
model_pca2.predict(x_test)
svc_pred_time = time.time() - start

print("测试准确率SVC:", accuracy_score(Y_test, Y_test_pred))
print("训练准确率SVC：", accuracy_score(Y_train, Y_train_pred))

cm = confusion_matrix(Y_test, Y_test_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("predicted class")
plt.ylabel("real class")
plt.title("SVC pca Confusion Matrix")
plt.show()

n_components = range(1, 13, 1)
accuracies = []

for n in n_components:
    pca = PCA(n_components=n)
    x_pca_n = pca.fit_transform(x_scaled)
    x_train_pca, x_test_pca, Y_train_pca, Y_test_pca = train_test_split(x_pca_n, Y, test_size=0.4, random_state=42)
    svm = SVC(kernel='linear', C=1.0, gamma=0.1)
    svm.fit(x_train_pca, Y_train_pca)
    Y_pred_pca = svm.predict(x_test_pca)
    acc = accuracy_score(Y_test_pca, Y_pred_pca)
    accuracies.append(acc)

# 绘制准确率变化曲线
plt.figure(figsize=(8, 5))
plt.plot(n_components, accuracies, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Accuracy')
plt.title('SVM Accuracy vs. PCA Components')
plt.grid(True)
plt.show()

#逻辑斯蒂回归
lr_clf = LogisticRegression(solver='liblinear')
lr_clf.fit(x_train1, Y_train1)

y_lr_pred = lr_clf.predict(x_test1)
Y_lr_pred = lr_clf.predict(x_train1)
print('测试集准确率Logistic(pca前):', accuracy_score(Y_test1, y_lr_pred))
print('训练集准确率Logistic(pca前)：', accuracy_score(Y_train1, Y_lr_pred))

cm = confusion_matrix(Y_test1, y_lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("predicted class")
plt.ylabel("real class")
plt.title("LogisticRegression Confusion Matrix")
plt.show()

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

n_components1 = range(1, 13, 1)
accuracies1 = []

for n in n_components1:
    pca = PCA(n_components=n)
    x_pca_n1 = pca.fit_transform(x_scaled)
    x_train_pca1, x_test_pca1, Y_train_pca1, Y_test_pca1 = train_test_split(x_pca_n1, Y, test_size=0.4, random_state=0)
    lr_clf1 = LogisticRegression(solver='liblinear')
    lr_clf1.fit(x_train_pca1, Y_train_pca1)
    Y_pred_pca1 = lr_clf1.predict(x_test_pca1)
    acc1 = accuracy_score(Y_test_pca1, Y_pred_pca1)
    accuracies1.append(acc1)

# 绘制准确率变化曲线
plt.figure(figsize=(8, 5))
plt.plot(n_components1, accuracies1, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Principal Components')
plt.ylabel('Accuracy')
plt.title('Logistic Accuracy vs. PCA Components')
plt.grid(True)
plt.show()

lr_clf1 = LogisticRegression(solver='liblinear')
lr_clf1.fit(x_train, Y_train)

start = time.time()
lr_clf1.fit(x_train, Y_train)
lr_train_time = time.time() - start

start = time.time()
lr_clf1.predict(x_test)
lr_pred_time = time.time() - start

y_lr_pred1 = lr_clf1.predict(x_test)
Y_lr_pred1 = lr_clf1.predict(x_train)
print('测试集准确率Logistic:', accuracy_score(Y_test, y_lr_pred1))
print('训练集准确率Logistic：', accuracy_score(Y_train, Y_lr_pred1))

cm = confusion_matrix(Y_test, Y_test_pred1)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2], yticklabels=[0, 1, 2])
plt.xlabel("predicted class")
plt.ylabel("real class")
plt.title("Logistic pca Confusion Matrix")
plt.show()

print(f"SVC训练时间：{svc_train_time: .8f}s, 预测时间：{svc_pred_time: .8f}s")
print(f"逻辑回归训练时间：{lr_train_time:.8f}s, 预测时间：{lr_pred_time:.8f}s")