import pandas as pd
import numpy as np

# Dosyamızı okuyup sütun isimlerini veriyoruz ve özellikleri hakkında bilgi alacağız 
# Hem de kullanacağımız şekilde düzenleyeceğiz.

df = pd.read_csv('HUMIRSentimentDatasets.csv', encoding ='iso-8859-9', sep='\t')
column = ['id', 'tür', 'yorum', 'sonuc', 'test/train','non']
df.columns=column
print("Dataframe:".upper())
print(df.head())



print("\ndf Satır/Sütun".upper())
print(df.shape)
print("\ndf hakkında bilgi(info)".upper())
df.info()
print("\ndf hakkında bilgi(describe)".upper())
print(df.describe())


print("\ntür colonunda kaç farklı değer var".upper())
print(df['tür'].unique())

print("\ntest/train colonunda kaç farklı değer var".upper())
print(df['test/train'].unique())

print("\nsonuc colonunda kaç farklı değer var".upper())
print(df['sonuc'].unique())

print("\nnon colonunda kaç farklı değer var".upper())
print(df['non'].unique())


# Datasetimizde hem hotel hem film yorumları var 
# biz sadece filmler ile işlem yapacağız 

df_Moive = df[df['tür'] == 'Movie Review']
df_Moive = df_Moive.drop(["non"], axis = 1)
df_Moive = df_Moive.drop(["tür"], axis = 1)


# test ve train olarak ayırıyoruz  ve kullanmayacağımız featureları siliyoruz 

df_Moive_test=df_Moive[df_Moive['test/train'] == 'test']
df_Moive_test = df_Moive_test.drop(["test/train"] , axis = 1)


df_Moive_train=df_Moive[df_Moive['test/train'] == 'train']
df_Moive_train = df_Moive_train.drop(["test/train"], axis = 1)


print("\ndf_Moive_test Satır/Sütun".upper())
print(df_Moive_test.shape)
print("\ndf_Moive_test hakkında bilgi".upper())
df_Moive_test.info()


print("\ndf_Moive_train Satır/Sütun".upper())
print(df_Moive_train.shape)
print("\ndf_Moive_train hakkında bilgi".upper())
df_Moive_train.info()

print("\ndf_Moive_train head".upper())
print(df_Moive_train.head())

# eksik veri olup olmadığını kontrol edelim 
print("\ndf_Moive_train eksik veri olup olmadığını kontrol edelim ".upper())
print(df_Moive_train.isnull().sum())

# sonuç kısmını kontrol edelim 
print("\ndf_Moive_train sonuç kısmını(positive/negative) kontrol edelim ".upper())
print(df_Moive_train['sonuc'].value_counts().index)


df_Moive_train_vc = df_Moive_train['sonuc'].value_counts()
print("\ndf_Moive_train_vc ".upper())
print(df_Moive_train_vc)



# Görselleştirmeler 
import matplotlib.pyplot as plt

# Kaç negatif kaç pozitif yorum var

fig, ax = plt.subplots(1,1,figsize=(8,6))

ax.bar(range(2), df_Moive_train_vc)
ax.set_xticks(range(2))
ax.set_xticklabels(df_Moive_train_vc.index, fontsize=16)


for rect, c, value in zip(ax.patches, ['b', 'r'], df_Moive_train_vc.values):
    rect.set_color(c)
    height = rect.get_height()
    width = rect.get_width()
    x_loc = rect.get_x()
    ax.text(x_loc + width/2, 0.9*height, value, ha='center', va='center', fontsize=18, color='white')
    
plt.show()    


    
# yorumlarda kelime sayısı öğrenme 
# anlamadığım şekilde 2 fazla sayıyor düzeltemediğim için 2 ekleyip çıkarttığım yerler var :D
document_lengths = np.array(list(map(len, df_Moive_train['yorum'].str.split(' ')))) 

    
print("\ndf_Moive_train Yorumları kontrol edelim ".upper())
print("\nBu dokümandaki Yorumlarda kullanıla ortalama kelime sayısı: {}.".format(np.mean(document_lengths)-2))
print("Bu dokümandaki Yorumlarda kullanıla minumun kelime sayısı: {}.".format(min(document_lengths)-2))
print("Bu dokümandaki Yorumlarda kullanıla maksimum kelime sayısı: {}.".format(max(document_lengths)-2))


import seaborn as sns

#kelime sayısının dağılımı 
fig, ax = plt.subplots(figsize=(15,6))

ax.set_title("Sözcüklerin dağılımı", fontsize=16)
ax.set_xlabel("Kelime sayısı")
sns.distplot(document_lengths-2, bins=50, ax=ax);
plt.show()

print("\n250 den fazla kelime içeren yorum sayısı: {}".format(sum(document_lengths > 250+2)).upper())


# 250 den fazla kelime olan çok yorum yok o yüzden 250 den az kelime içerenleri yakından bakalım 
shorter_documents = document_lengths[document_lengths <= 250+2]
                                     
fig, ax = plt.subplots(figsize=(15,6))

ax.set_title("Sözcüklerin dağılımı", fontsize=16)
ax.set_xlabel("Kelime sayısı")
sns.distplot(shorter_documents-2, bins=50, ax=ax);
plt.show()


print("\n5 den az kelime içeren yorum sayısı: {}".format(sum(document_lengths < 5+2)).upper())


#1, 2, 3 kelimeye sahip yorumlardan bazılarını görelim
print("\n1 kelime olan yorumlardan bazıları ".upper())
one_lengths = df_Moive_train[document_lengths == 1+2]
print(one_lengths.head())
print("\n2 kelime olan yorumlardan bazıları ".upper())
two_lengths = df_Moive_train[document_lengths == 2+2]
print(two_lengths.head())
print("\n3 kelime olan yorumlardan bazıları ".upper())
three_lengths = df_Moive_train[document_lengths == 3+2]
print(three_lengths.head())



