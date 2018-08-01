import pandas as pd
import numpy as np

df = pd.read_csv('HUMIRSentimentDatasets.csv', encoding ='iso-8859-9', sep='\t')
column = ['id', 'tür', 'yorum', 'sonuc', 'test/train','non']
df.columns=column


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



#metnimizi temiizlioruz
import re

def get_good_tokens(sentence):
    replaced_punctation = list(map(lambda token: re.sub('[^0-9A-Za-z!?]+', '', token), sentence))
    removed_punctation = list(filter(lambda token: token, replaced_punctation))
    return removed_punctation

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize


def w2v_preprocessing(df_fon):
    """ Word2vec için tüm ön işlem adımları bu işlevde yapılır.
        Tüm mutasyonlar veri çerçevesinin kendisinde yapılır.
        Yani bu işlev hiçbir şey döndürmez.
    """
    df_fon['yorum'] = df_fon['yorum'].str.lower()
    df_fon['document_sentences'] = df_fon['yorum'].str.split('.')  # metni cümleler halinde ayırıyor
    df_fon['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(nltk.word_tokenize, sentences)),
                                         df_fon.document_sentences))  # tokenize cümleler
    df_fon['tokenized_sentences'] = list(map(lambda sentences:
                                         list(map(get_good_tokens, sentences)),
                                         df_fon.tokenized_sentences))  # istenmeyen karakterler kaldırılır
    df_fon['tokenized_sentences'] = list(map(lambda sentences:
                                         list(filter(lambda lst: lst, sentences)),
                                         df_fon.tokenized_sentences))  # boş listeler kaldırılır

w2v_preprocessing(df_Moive_train)


from collections import Counter

def lda_get_good_tokens(df_fon):
    df_fon['yorum'] = df_fon.yorum.str.lower()
    df_fon['tokenized_text'] = list(map(nltk.word_tokenize, df_fon.yorum))
    df_fon['tokenized_text'] = list(map(get_good_tokens, df_fon.tokenized_text))

lda_get_good_tokens(df_Moive_train)

tokenized_only_dict = Counter(np.concatenate(df_Moive_train.tokenized_text.values))

tokenized_only_df = pd.DataFrame.from_dict(tokenized_only_dict, orient='index')
tokenized_only_df.rename(columns={0: 'count'}, inplace=True)

tokenized_only_df.sort_values('count', ascending=False, inplace=True)

import matplotlib.pyplot as plt
import seaborn as sns 

def word_frequency_barplot(df_fon, nr_top_words=50):
    """
    """
    fig, ax = plt.subplots(1,1,figsize=(20,5))

    sns.barplot(list(range(nr_top_words)), df_fon['count'].values[:nr_top_words], palette='hls', ax=ax)

    ax.set_xticks(list(range(nr_top_words)))
    ax.set_xticklabels(df_fon.index[:nr_top_words], fontsize=14, rotation=90)
    return ax
    
ax = word_frequency_barplot(tokenized_only_df)
ax.set_title("Kelime Sıklıkları", fontsize=16);
plt.show()


## stop word lerin silinmesi
def remove_stopwords(df_fon):

    stopwords = open('turkce-stop-words', 'r').read().split()

    df_fon['stopwords_removed'] = list(map(lambda doc:
                                       [word for word in doc if word not in stopwords],
                                       df_fon['tokenized_text']))

remove_stopwords(df_Moive_train)


##Stemming
def stem_words(df_fon):
    lemm = nltk.stem.WordNetLemmatizer()
    df_fon['lemmatized_text'] = list(map(lambda sentence:
                                     list(map(lemm.lemmatize, sentence)),
                                     df_fon.stopwords_removed))

    p_stemmer = nltk.stem.porter.PorterStemmer()
    df_fon['stemmed_text'] = list(map(lambda sentence:
                                  list(map(p_stemmer.stem, sentence)),
                                  df_fon.lemmatized_text))

stem_words(df_Moive_train)



from gensim.corpora import Dictionary

#Sözcükleri vektörle
dictionary = Dictionary(documents=df_Moive_train.stemmed_text.values)

print("Bulunan kelimeler: {}".format(len(dictionary.values())))

dictionary.filter_extremes(no_above=0.8, no_below=3)

dictionary.compactify()  # Filtrelemeden sonra kalan kelimeleri yeniden indeksler
print("Kalan kelimeler: {}".format(len(dictionary.values())))



#her belge için bir BOW

def document_to_bow(df_fon):
    df_fon['bow'] = list(map(lambda doc: dictionary.doc2bow(doc), df_fon.stemmed_text))
    
document_to_bow(df_Moive_train)



def lda_preprocessing(df_fon):
    """ LDA için tüm ön işlem adımları bu işlevde birleştirilir.
        Tüm mutasyonlar veri çerçevesinin kendisinde yapılır.
        Yani bu işlev hiçbir şey döndürmez.
    """
    lda_get_good_tokens(df_fon)
    remove_stopwords(df_fon)
    stem_words(df_fon)
    document_to_bow(df_fon)



cleansed_words_df = pd.DataFrame.from_dict(dictionary.token2id, orient='index')
cleansed_words_df.rename(columns={0: 'id'}, inplace=True)

cleansed_words_df['count'] = list(map(lambda id_: dictionary.dfs.get(id_), cleansed_words_df.id))
del cleansed_words_df['id']

cleansed_words_df.sort_values('count', ascending=False, inplace=True)

ax = word_frequency_barplot(cleansed_words_df)
ax.set_title("Belge Sıklıkları (Bir sözcüğün içerdiği belge sayısı)", fontsize=16);
plt.show()


negative_words = list(np.concatenate(df_Moive_train.loc[df_Moive_train.sonuc == 'Negative', 'stemmed_text'].values))
positive_words = list(np.concatenate(df_Moive_train.loc[df_Moive_train.sonuc == 'Positive', 'stemmed_text'].values))


negative_word_frequencies = {word: negative_words.count(word) for word in cleansed_words_df.index[:50]}
positive_word_frequencies = {word: positive_words.count(word) for word in cleansed_words_df.index[:50]}


frequencies_df = pd.DataFrame(index=cleansed_words_df.index[:50])


frequencies_df['negative_freq'] = list(map(lambda word:
                                      negative_word_frequencies[word],
                                      frequencies_df.index))
frequencies_df['negative_positive_freq'] = list(map(lambda word:
                                          negative_word_frequencies[word] + positive_word_frequencies[word],
                                          frequencies_df.index))
    
    
    
    
fig, ax = plt.subplots(1,1,figsize=(20,5))

nr_top_words = len(frequencies_df)
nrs = list(range(nr_top_words))
sns.barplot(nrs, frequencies_df['negative_positive_freq'].values, color='g', ax=ax, label="Positive")
sns.barplot(nrs, frequencies_df['negative_freq'].values, color='r', ax=ax, label="Negative")

ax.set_title("Negative - Positive başına kelime sıklığı", fontsize=16)
ax.legend(prop={'size': 16})
ax.set_xticks(nrs)
ax.set_xticklabels(frequencies_df.index, fontsize=14, rotation=90);
plt.show()



