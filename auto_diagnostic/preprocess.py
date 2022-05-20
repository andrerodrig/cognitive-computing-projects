import nltk
from typing import List


def tokenize(corpus_list: List[str]) -> List[str]:
    tokenizer = nltk.RegexpTokenizer(r'\w+')
    tokenized_text = list(map(lambda x: tokenizer.tokenize(x), corpus_list))
    return tokenized_text

def get_stopwords(language: str = 'portuguese') -> List[str]:
    nltk.download('stopwords')
    custom_stopwords = [
        'i', 'ii', 'iii', 'iv', 'ate', 'sera', 'vs',
        'instituto', 'vs', 'of', 'respectivamente'
    ]
    nltk_stopwords = nltk.corpus.stopwords.words(language)
    nltk_stopwords.extend(custom_stopwords)    
    return list(set(nltk_stopwords))