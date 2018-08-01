## Word2Vec ve LDA


Humır veri seti ile Word2Vec ve LDA kullanarak metin analizi(pozitif-negatif) yapılan bir örnek 

Genel olarak 4 ana adımla ilerleyeceğiz(adımları yaptıkça ekleyeceğim); 
* Özellik Denetimi
* Özellik Dluşturma 
* Model Eğitimi
* Model Seçimi


###### 1) Özellik denetimi
Kullanacağımız veri setinde hızlı bir keşif yaptığımız aşama;
* Datasetin okunması 
* Sütun isimlerinin verilmesi
* Örnek olarak 5 tane içeriğe bakılması

  ![df_head(1)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/df_head(1).PNG)
* Satır ve sütun sayılarının öğrenilmesi
  
  ![df_shape(2)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/df_shape(2).PNG)
* Dataset hakkında genel bilgilerin öğrenilmesi(info, describe)
  
  ![df_info_desciribe(3)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/df_info_desciribe(3).PNG)
* Sütunların bazılarnının içeriğinde kaç farklı değerin olduğuna bakılması 
  
  ![df_unique(4)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/df_unique(4).PNG)
* Datasette kullanmayacağımız verilerin ayrıştırılması
* Dataseti train ve test olarak ayırmak 
* Datasetin test kısmında bilgilerin öğrenilmesi
  
  ![test_info(5)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/test_info(5).PNG)
* Datasetin train kısmında bilgilerin öğrenilmesi
  
  ![train_info(6)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/train_info(6).PNG)
  
  ![train_info2(7)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/train_info2(7).PNG)
* Pozitif ve negatif yorumların sayısının görselleştirilmesi 
  
  ![figure1](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/figure1.PNG)
* Yorumlarda kullanılan kelimelerin sayılayılarının dağılımlarının görselleştirilmesi
  
  ![figure2](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/figure2.PNG)
  ![figure3](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/figure3.PNG)
* Yorumlarda kullanıla max, min, ortalama kelime sayıları ve 250 den fazla kelime içerenlerin yorumlar

  ![train_kelime(8)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/train_kelime(8).PNG)
  
* 1, 2, 3 kelimeye sahip yorumlardan bazılarına bakılması 

  ![train_kelime(9)](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Denetimi/train_kelime(9).PNG)

###### Yararlanılan Kaynaklar 
* [Classification combining LDA and Word2Vec
](https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec)


