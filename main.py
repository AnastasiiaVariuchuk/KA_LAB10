import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Зчитування даних з файлу
gdp_data = pd.read_csv('ny.gdp.mktp.kd.zg_Indicator_en_csv_v2.csv')

# Вибірка стовпців за роками
columns = [str(year) for year in range(1961, 2013)]
gdp_columns = [col for col in gdp_data.columns if col in columns]
gdp_data = gdp_data[['Country Name'] + gdp_columns]

# Видалення рядків з відсутніми значеннями
gdp_data.dropna(inplace=True)

# Виконання кластеризації
k = 5  # Кількість кластерів
km_gdp = KMeans(n_clusters=k, random_state=42).fit(gdp_data.iloc[:, 1:])

# Виведення результатів
labels = km_gdp.labels_
print(labels)
gdp_data['Cluster'] = labels

# Візуалізація
plt.scatter(gdp_data['Country Name'], gdp_data['2010'], c=gdp_data['Cluster'], cmap='viridis')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('GDP in 2010')
plt.title('Cluster Analysis of Countries by GDP')
plt.show()

# Зчитати дані з файлу CSV
unemployment_data = pd.read_csv('sl.uem.totl.zs_Indicator_en_csv_v2.csv')


# Вибрати стовпці з рівнями безробіття за певний період
columns = ['Country Name', '2000', '2001', '2002', '2003', '2004', '2005']
unemployment_data = unemployment_data[columns]
unemployment_data.dropna(inplace=True)


# Виконати кластеризацію
k = 3  # кількість кластерів
km_unemployment = KMeans(n_clusters=k, random_state=42).fit(unemployment_data.iloc[:, 1:])

# Отримати мітки кластерів для кожної країни
cluster_labels = km_unemployment.labels_
print(cluster_labels)

# Побудувати графік кластеризації
plt.scatter(unemployment_data['2000'], unemployment_data['2005'], c=cluster_labels)
plt.xlabel('Unemployment Rate 2000')
plt.ylabel('Unemployment Rate 2005')
plt.title('Clustering of Countries by Unemployment Rate')
plt.show()


