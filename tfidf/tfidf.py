import nltk
import  numpy as np



class TFIDF():
    def __init__(self,word_set,sentence):
        self.word_set = word_set
        self.sentence = sentence
        self.index_dict = {}
        i = 0
        for word in word_set:
            self.index_dict[word] = i
            i += 1

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
        total_documentos = len(self.sentence)
        try:
            palavras_ = word_count[word] + 1
        except:
            palavras_ = 1

        return np.log(total_documentos/palavras_)


    def tf_idf(self):
        tf_idf_vec = np.zeros(len(self.word_set),)
        for word in self.sentence:
            tf = self.tf(self.sentence,word)
            idf = self.idf(word)

            result = tf*idf
            tf_idf_vec[self.index_dict[word]] = result
        return  tf_idf_vec

    def select_n_largets(self, tf_idf_vec):
        retorno = list()
        for vector in tf_idf_vec:
            if max(vector) != 0 and max(vector)>0:
                retorno.append(max(vector))
        return retorno
