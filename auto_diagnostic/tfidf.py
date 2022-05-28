import nltk
import  numpy as np
from typing import List

from nltk.tokenize import word_tokenize
from auto_diagnostic.preprocess import get_stopwords


class TFIDF():
    def __init__(self, word_set: List = None, sentences: List = None):
        self.word_set = [] if not word_set else word_set
        self.sentences = [] if not sentences else sentences
        self.stopwords = get_stopwords()
        self.index_dict = {}
            
    def get_wordset_from_text(self, text_list: List[str]):
        self.word_set = []
        self.sentences = []
        for sent in text_list:
            x = [i.lower() for i in word_tokenize(sent) if i.isalpha()]
            #x = [word for word in x if word not in set_Stopwords]
            self.sentences.append(x)
            for word in x:
                if word not in self.word_set and word not in self.stopwords:
                    self.word_set.append(word)
        self.word_set = set(self.word_set)
        self.index_dict = self.get_wordset_index_dict()
        return self.word_set
    
    def get_wordset_index_dict(self):
        index_dict = {}
        for index, word in enumerate(self.word_set):
            index_dict[word] = index
        return index_dict

    def count_dict(self):
        word_count = {}
        for word in self.word_set:
            word_count[word] = 0
            for sent in self.sentences:
                if word in sent:
                    word_count[word] += 1
        return word_count

    def tf(self, word):
        # tam = len(document)
        tam = sum(self.count_dict().values())
        document = self.count_dict()
        ocorrencia = sum([count for token, count in document.items() if token == word])
        return ocorrencia/tam

    def idf(self,word):
        word_count = self.count_dict()
        total_documentos = len(self.sentences)
        try:
            palavras_ = word_count[word] + 1
        except:
            palavras_ = 1
        return np.log(total_documentos/palavras_)

    def tf_idf(self):
        tf_idf_vec = np.zeros(len(self.word_set),)
        # word_list = [word for sentence in self.sentences for word in sentence]
        word_list = list(self.word_set)
        for word in word_list:
            tf = self.tf(word)
            idf = self.idf(word)
            result = tf * idf
            tf_idf_vec[self.index_dict[word]] = result
        return  tf_idf_vec
    
    def get_tuple_index_word_tfidf(self, vectors: List):
        indexed_tfidf = [(v, vectors[v]) for v in self.index_dict.values()]
       
        only_tfidf_values = [tf for tf in list(zip(*indexed_tfidf))[1]]
        tuple_index_word_tfidf = [
            (word, index, tf) for index, word, tf
            in zip(self.index_dict.keys(), self.index_dict.values(), only_tfidf_values)
        ]
        return tuple_index_word_tfidf

    def select_n_largests(self, vectors: List, n: int = 5):
        return sorted(
            self.get_tuple_index_word_tfidf(vectors=vectors),
            key=lambda x: x[2],
            reverse=True
        )[:n]

    def get_closest_neighbors(self, vectors: List, n: int = 5):
        tuple_iwt = self.get_tuple_index_word_tfidf(vectors)
        key_neighbors_list = []
        for tup in self.select_n_largests(vectors=vectors, n=n):
            neighbors_list = [
                (*t[:2], self.tf(tup[1]))
                for t in tuple_iwt if abs(t[0] - tup[0]) <= 2
            ]
            key_neighbors_list.append(
                {'key': tup, 'neighbors': [t for t in neighbors_list if t != tup]}
            )
        return key_neighbors_list


if __name__ == '__main__':
    
    from dataloader.makedata import Dataloader
    from auto_diagnostic.lemmatization import Lemmatization
    from auto_diagnostic.preprocess import tokenize
    
    
    df = Dataloader().make_csv()

    tokenized = tokenize(df['text_column'])

    model = Lemmatization()
    lemmatized_list = model.lemmatize(tokenized)

    tfidf = TFIDF()
    words = tfidf.get_wordset_from_text(lemmatized_list)
    
    vectors = tfidf.tf_idf()
    
    neighbors = tfidf.get_closest_neighbors(vectors)
        
    print(neighbors)
