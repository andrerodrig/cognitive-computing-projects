import nltk

from auto_diagnostic.tfidf import TFIDF
from auto_diagnostic.lemmatization import Lemmatization
from auto_diagnostic.preprocess import tokenize

from dataloader.makedata import Dataloader


def main():
    tfidf = TFIDF()
    df = Dataloader().make_csv()
    tokenized = tokenize(df['text_column'])
    model = Lemmatization()
    lemmatized_list = model.lemmatize(tokenized)
    word_set = tfidf.get_wordset_from_text(lemmatized_list)
    tfidf_vectors = tfidf.tf_idf()
    closest = tfidf.get_closest_neighbors(vectors=tfidf_vectors)
    for key in closest:
        print(key)

if __name__=="__main__":
    main()