import argparse

from auto_diagnostic.tfidf import TFIDF
from auto_diagnostic.lemmatization import Lemmatization
from auto_diagnostic.preprocess import tokenize

from graphs import word_graph
from dataloader.makedata import Dataloader


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-f', '--filepath',
        help="Generate the word graph for the given pdf_file.",
        type=str,
    )
    arg = parser.parse_args()
    
    dataloader = Dataloader(arg.filepath) if arg.filepath else Dataloader()
    df = dataloader.make_csv()
    tfidf = TFIDF()
    tokenized = tokenize(df['text_column'])
    model = Lemmatization()
    lemmatized_list = model.lemmatize(tokenized)
    _ = tfidf.get_wordset_from_text(lemmatized_list)
    tfidf_vectors = tfidf.tf_idf()
    closest = tfidf.get_closest_neighbors(vectors=tfidf_vectors)
    word_graph.graph(closest)

if __name__=="__main__":
    main()
