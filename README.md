## Word2Vec ve LDA


Humır veri seti ile Word2Vec ve LDA kullanarak metin analizi(pozitif-negatif) yapılan bir örnek 

Genel olarak 4 ana adımla ilerleyeceğiz(adımları yaptıkça ekleyeceğim); 
* Özellik Denetimi
* Özellik Oluşturma 
* Model Eğitimi
* Model Seçimi


###### 1) Özellik Denetimi
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


###### 2) Özellik Oluşturma
Doğal dil işlemede olağan yaklaşım, metni ilk olarak temizlemektir. 
Modelimizin benzerlikleri, iki farklı kelime benzer şeyler ifade ettiği zaman anladığından emin olmalıyız.
Ham metni bir modele girip her şeyi anlamasını bekleyemeyiz.
Bunun için biraz temizlik yapmamız gerekir.
Genel olarak yapılanlar:
* Metni tokinize etmek (cümle cümle sonra kelime kelime ayırıyoruz, boşlukları temizliyoruz, istenmeyen karakterleri kaldırıyoruz)
* Kelime sıklıklarının görselleştirilmesi

  ![figure1](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Olusturma/figure1.PNG)
* Stop word leri kaldırıyoruz (türkçedeen çok kullanılan yan kelimeler)
* Stemming(genel olarak aynı ifadeye denk gelen kelimelerin birleştirilmesi)
* Sözcükleri vektörize etmek
* Tüm işlemler bittikten sonra tekrar kelime sıklıklarının görselleştirilmesi

  ![figure2](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Olusturma/figure2.PNG)
* Negatif ve Pozitif yorumlarda en çok kullanılan kelimelerin görselleştirilmesi
  
  ![figure3](https://github.com/FatmaKaya/Word2Vec-LDA/blob/master/Ozellik%20Olusturma/figure3.PNG)

###### Yararlanılan Kaynaklar 
* [Classification combining LDA and Word2Vec
](https://www.kaggle.com/vukglisovic/classification-combining-lda-and-word2vec)


