import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from ucimlrepo import fetch_ucirepo

#Завантаження даних (ID 292 - Wholesale customers)
print('=== 1. ЗАГРУЗКА ДАННЫХ ===')
wholesale_customers = fetch_ucirepo(id=292)
X = wholesale_customers.data.features
y = wholesale_customers.data.targets

#Об'єднуємо ознаки та цільові змінні в єдиний DataFrame
if y is not None and not y.empty:
  df = pd.concat([X, y], axis=1)
else:
  df = X

#Аналіз даних (вивід базової інформації)
print('\n=== 2. АНАЛИЗ ДАННЫХ ===')
print('Метадані:')
print(wholesale_customers.metadata)
print('\nПерші 5 рядків датасету:')
print(df.head())
print('\nОписова статистика:')
print(df.describe())

#Побудова та збереження 4 різних графіків

#Гістограма витрат на свіжі продукти (Fresh)
plt.figure(figsize=(8, 6))
sns.histplot(df['Fresh'], bins=20, kde=True, color='green')
plt.title('Розподіл річних витрат на свіжі продукти (Fresh)')
plt.xlabel('Витрати (грошові одиниці)')
plt.ylabel('Частота')
plt.savefig('histogram_wholesale.png')
plt.close()

#Діаграма розсіювання (Milk vs Grocery)
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='Milk',
    y='Grocery',
    hue='Channel' if 'Channel' in df.columns else None,
    data=df,
    palette='Set1',
)
plt.title('Залежність між витратами на молоко та бакалію')
plt.xlabel('Молоко (Milk)')
plt.ylabel('Бакалія (Grocery)')
plt.legend(title='Канал збуту')
plt.savefig('scatterplot_wholesale.png')
plt.close()

#Ящик з вусами (Box plot) для порівняння витрат за категоріями
plt.figure(figsize=(10, 6))
cols_to_box = [
    c
    for c in ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper']
    if c in df.columns
]
sns.boxplot(data=df[cols_to_box], palette='Set3')
plt.title('Порівняння витрат за категоріями товарів')
plt.xlabel('Категорія товару')
plt.ylabel('Витрати (му)')
plt.savefig('boxplot_wholesale.png')
plt.close()

#теплова карта кореляцій (Heatmap)
plt.figure(figsize=(8, 6))
correlation_matrix = df.select_dtypes(include=['number']).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Теплова карта кореляцій витрат оптових клієнтів')
plt.savefig('heatmap_wholesale.png')
plt.close()

print(
    '\nУспішно! Усі 4 графіки для датасету Wholesale customers збережено у'
    ' корінь репозиторію.'
)
