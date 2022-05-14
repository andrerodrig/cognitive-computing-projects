import nltk
import  numpy as np

from nltk.tokenize import word_tokenize


class TFIDF():
    def __init__(self,word_set,sentences):
        self.word_set = word_set
        self.sentences = sentences
        self.index_dict = {}
        i = 0
        for word in word_set:
            self.index_dict[word] = i
            i += 1
            
    def get_wordset_from_text(self, text: str):
        self.word_set = []
        self.sentences = []
        for sent in text:
            x = [i.lower() for  i in word_tokenize(sent) if i.isalpha()]
            #x = [word for word in x if word not in set_Stopwords]
            self.sentences.append(x)
            for word in x:
                if word not in self.word_set:
                    self.word_set.append(word)
        self.word_set = set(self.word_set)
        
        return self.word_set
    
    def get_wordset_index_dict(self):
        self.index_dict = {}
        for index, word in enumerate(self.word_set):
            self.index_dict[word] = index
        
        return self.index_dict

    def count_dict(self):
        word_count = {}
        for word in self.word_set:
            word_count[word] = 0
            for sent in self.sentences:
                if word in sent:
                    word_count[word] += 1
        return word_count

    def tf(self,document,word):
        tam = len(document)
        ocorrencia = len([token for token in document if token == word])
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
        word_list = [word for sentence in self.sentences for word in sentence]

        for word in word_list:
            tf = self.tf(word_list, word)
            idf = self.idf(word)

            result = tf * idf
            tf_idf_vec[self.index_dict[word]] = result
        return  tf_idf_vec

    # def select_n_largets(self, tf_idf_vec):
    #     retorno = list()
    #     for vector in tf_idf_vec:
    #         if max(vector) != 0 and max(vector)>0:
    #             retorno.append(max(vector))
    #     return retorno

    def select_n_largests(self, vectors):
        indexed_tfidf = [(v, vectors[v]) for v in self.index_dict.values()]
        max_tfidf_list = [(k, indexed_tfidf[v][1]) for k, v in self.index_dict.items()]
        return sorted(max_tfidf_list, key=lambda x: x[1], reverse=True)[:5]
