# CHANCE OF STROKE - DATASET FROM KAGGLE
# objetivo: prever se os pacientes terão ou não um AVC

# importando as bibliotecas básicas
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('max_columns', None)
sns.set(style='darkgrid') # colocando o tema 'darkgrid' para os gráficos que serão feitos

# importando o dataset
df = pd.read_csv('stroke.csv')
df.set_index('id', inplace=True, drop=True) # como a coluna id não possui valor algum, vamos transformar no nosso índice

# conhecendo os dados
df.info() # existem dados faltantes na coluna de IMC (bmi) e tipo de dado de idade está como float, vamos transformar em inteiros
df.isnull().sum()

df.describe()
print('As 10 menores idades:\n', df.nsmallest(10, 'age')['age']) # crianças de até 2 anos estão com idade decimal para idade em meses, vamos arredondar

# Arrendondando idades
df['age'] = df['age'].apply(lambda x : round(x))
print('As 10 menores idades:\n', df.nsmallest(10, 'age')['age'])

# Retirando sexo diferente de 'Male' e 'Female'
# df.drop(df[df['gender'] == 'Other'].index, inplace=True)

# Tratando BMI
sns.boxplot(data=df, x='bmi').set(title='Boxplot BMI') # vemos alguns outliers que provavelmente são erros de digitação

df['bmi'] = df['bmi'].apply(lambda x: x if x < 61 else np.nan) # todos bmi abaixo de 12 e acima de 61 vao ser transformados em NaN

# agora vamos preencher esses NaN com o método forward fill
# organiza-se o dataset por gênero e idade e completa os dados faltantes exatamente igual ao dado anterior
# isso garante que o dado vai ser de alguém da mesma faixa etária e de idade 
df.sort_values(['gender', 'age'], inplace=True)
df.reset_index(inplace=True, drop=True)
df.fillna(method='ffill', inplace=True)

# ANALISE EXPLORATORIA
palette = ['#3399FF', '#FF6666'] # definindo palete de cores

# Analisando variavel target
ax = sns.countplot(data=df, x='stroke',  palette=palette)
ax.set(title='O paciente já teve um AVC?', xlabel='', ylabel='', xticklabels=['Não', 'Sim']) # os dados estão desbalanceados

# vamos calcular quantos % dos pacientes já tiveram um AVC
stroke_perc = (len(df[df['stroke'] == 1])/len(df['stroke']))*100
print(f"Pacientes que já tiveram AVC: {stroke_perc:.2f}%\nPacientes que nunca tiveram AVC {100-stroke_perc:.2f}%")

# Relação do genero com chance de AVC
# Retirando sexo diferente de 'Male' e 'Female'
df.drop(df[df['gender'] == 'Other'].index, inplace=True)

nostroke = df[df['stroke'] == 0].gender.value_counts() # quantas ocorrencias de cada status para quem nunca teve avc
stroke = df[df['stroke'] == 1].gender.value_counts() # quantas ocorrencias de cada status para quem já teve avc
plt.bar(nostroke.index, height = nostroke.values, width = 0.2, color=['#3399FF'])
plt.bar(np.arange(len(stroke.index)) , height = stroke.values, width = 0.5,color=['#FF6666'])
plt.title('Gênero x AVC', fontsize=14)
plt.legend(labels=['Nunca teve AVC', 'Já teve AVC'], prop={'size': 12},  fontsize=20)
plt.tight_layout()

# verificando qual a porcentagem para cada gênero
print(f"\nMulheres que já tiveram AVC: {(stroke[0]/(stroke[0]+nostroke[0])*100):.2f}%\nHomens que já tiveram AVC {(stroke[1]/(stroke[1]+nostroke[1])*100):.2f}%")
# os dois tem em média 5% de indivíduos que já tiveram stroke, assim como revelou o cálculo do dataset todo (stroke_perc)

# Analisando variáveis numéricas e target
# HEATMAP
numerical = ['age', 'avg_glucose_level', 'bmi']
ax = sns.heatmap(pd.concat([df[numerical], df['stroke']], axis=1).corr(), vmax=0.8, cmap='viridis') # nenhuma variável se correlaciona muito com o target
ax.set(title='Heatmap variaveis numéricas e target\n') 

# Gráfico KDE de AVC x idade
plt.figure(figsize=(10, 6))
ax = sns.kdeplot(data=df[df.stroke==1], x='age', label='Stroke', shade=True, color='#FF6666', alpha=1)
sns.kdeplot(data=df[df.stroke==0], x='age', label='No-stroke', shade=True, color='#3399FF', alpha=0.5, ax=ax)
plt.title('KDE: Idade x AVC', fontsize=16)
plt.legend(loc='upper left')
# é possível ver como a chance de se ter um AVC aumenta de acordo que a idade aumenta

# Distribuição 
fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))
sns.kdeplot(data=df[df.stroke==1],x='avg_glucose_level', label='Stroke', ax=ax[0], shade=True, color='#FF6666', alpha=1)
sns.kdeplot(data=df[df.stroke==0],x='avg_glucose_level', label='No-stroke', ax=ax[0], shade=True, color='#3399FF', alpha=0.5)
sns.kdeplot(data=df[df.stroke==1],x='bmi', label='Stroke', ax=ax[1],shade=True,color='#FF6666',alpha=1)
sns.kdeplot(data=df[df.stroke==0],x='bmi', label='No-stroke', ax=ax[1], shade=True, color='#3399FF',alpha=0.5)
plt.tight_layout()
plt.title('KDE: BMI e glicose x AVC\n', y=2.17, fontsize=14)
plt.legend()

# Analisando variáveis categóricas
fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))
ax[0].set_title('Contagem de variáveis booleanas\n', fontsize=19)
ax[0] = sns.countplot(data=df, y='hypertension', hue='stroke', ax=ax[0], palette=palette).set(xlabel='')
ax[1] = sns.countplot(data=df, y='heart_disease', hue='stroke', ax=ax[1], palette=palette).set(xlabel='')
ax[2] = sns.countplot(data=df, y='ever_married', hue='stroke', ax=ax[2], palette=palette).set(xlabel='')

# Comparando quantidade de AVCs em diferentes grupos amostrais
fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(8, 12))
ax[0].set_title('Probabilidade de uma pessoa ter um AVC\n', fontsize=19)
ax[0] = sns.countplot(data=df, y='stroke', ax=ax[0], palette=palette).set(xlabel='')
ax[1].set_title('\nDoença cardíaca x AVC', fontsize=14)
ax[1] = sns.countplot(data=df[df['heart_disease']==1], y='stroke', ax=ax[1], palette=palette).set(xlabel='')
ax[2].set_title('Hipertensão x AVC', fontsize=14)
ax[2] = sns.countplot(data=df[df['hypertension']==1], y='stroke', ax=ax[2], palette=palette).set(xlabel='')
ax[3].set_title('Casamento x AVC', fontsize=14)
ax[3] = sns.countplot(data=df[df['ever_married']=='Yes'], y='stroke', ax=ax[3], palette=palette).set(xlabel='')
fig.tight_layout()

# Vendo qual % das pessoas com comorbidade já tiveram um AVC
var = ['hypertension', 'heart_disease']
for v in var:
    group = df.loc[df[v] == 1]
    stroke_perc = (len(group[group['stroke'] == 1])/len(group))*100
    print(f"\nPacientes com '{v}' que já tiveram AVC: {stroke_perc:.2f}%\nPacientes com '{v}' que nunca tiveram AVC: {100-stroke_perc:.2f}%")
# uma pessoa com hipertensão tem mais que o dobro de chance de ter um AVC quando comparado com uma pessoa que não tem
# uma pessoa com doença cardíaca tem mais que o triplo da chance de ter um AVC quando comparado com uma pessoa que naão tem

# Analisando variável de fumantes
nostroke = df[df['stroke'] == 0].smoking_status.value_counts() # quantas ocorrencias de cada status para quem nunca teve avc
stroke = df[df['stroke'] == 1].smoking_status.value_counts() # quantas ocorrencias de cada status para quem já teve avc
plt.bar(nostroke.index, height = nostroke.values, width = 0.2, color=['#3399FF'])
plt.bar(np.arange(len(stroke.index)) , height = stroke.values, width = 0.5,color=['#FF6666'])
plt.title('Smoking_status x AVC', fontsize=14)
plt.legend(labels=['Nunca teve AVC', 'Já teve AVC'], prop={'size': 12},  fontsize=20)
plt.tight_layout()


# Vamos copiar o dataset e isolar a variável target
X = df.copy()
y = X.pop('stroke')

# Realizar o encoder das variáveis categoricas
from sklearn.preprocessing import LabelEncoder, StandardScaler
encoder = LabelEncoder()
X['gender'] = encoder.fit_transform(X['gender'])
X['ever_married'] = encoder.fit_transform(X['ever_married'])
X['Residence_type'] = encoder.fit_transform(X['Residence_type'])
X['work_type'] = encoder.fit_transform(X['work_type'])
X['smoking_status'] = encoder.fit_transform(X['smoking_status'])

# Padronização da escala dos dados
scaler = StandardScaler() 
X = scaler.fit_transform(X)

# Como os dados estão desbalanceados, vamos utilizar o SMOTE para balanceá-los.
# SMOTE é um método de over-sampling 
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
smote = SMOTE()
X_train_bal, y_train_bal = smote.fit_resample(X, y)
X_train,  X_test, y_train, y_test = train_test_split(X_train_bal, y_train_bal, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, plot_confusion_matrix


# REGRESSÃO LOGÍSTICA  
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

print('#'*60)
print(' '*20,'REGRESSÃO LOGISTICA\n')
print(classification_report(y_test, lr_pred))
plot_confusion_matrix(lr, X_test, y_test, cmap='cividis')
plt.grid(False)
plt.title('REGRESSÃO LOGISTICA')
print('#'*60)

# KNN 
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

print('#'*60)
print(' '*27, 'KNN\n')
print(classification_report(y_test, knn_pred))
plot_confusion_matrix(knn, X_test, y_test, cmap='cividis')
plt.grid(False)
plt.title('KNN')
print('#'*60)

# SVC
svc = SVC()
svc.fit(X_train, y_train)
svc_pred = svc.predict(X_test)

print('#'*60)
print(' '*27,'SVC\n')
print(classification_report(y_test, svc_pred))
plot_confusion_matrix(svc, X_test, y_test, cmap='cividis')
plt.grid(False)
plt.title('SVM')
print('#'*60)

# ARVORE DE DECISAO
tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_pred = tree.predict(X_test)

print('#'*60)
print(' '*20,'ARVORE DE DECISÃO\n')
print(classification_report(y_test, tree_pred))
plot_confusion_matrix(tree, X_test, y_test, cmap='cividis')
plt.grid(False)
plt.title('ARVORE DE DECISÃO')
print('#'*60)

# FLOERSTA RANDOMICA
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)

print('#'*60)
print(' '*20,'RANDOM FOREST\n')
print(classification_report(y_test, rf_pred))
plot_confusion_matrix(rf, X_test, y_test, cmap='cividis')
plt.grid(False)
plt.title('RANDOM FOREST')
print('#'*60)

# XGBOOST
xgb = XGBClassifier()
xgb.fit(X_train, y_train)
xgb_pred = xgb.predict(X_test)

print('#'*60)
print(' '*29,'XGBOOST\n')
print(classification_report(y_test, xgb_pred))
plot_confusion_matrix(xgb, X_test, y_test, cmap='cividis')
plt.grid(False)
plt.title('XGBOOST')
print('#'*60)

