from typing import List, Union, Any
from auto_diagnostic.preprocess import get_stopwords
import stanza


class Lemmatization:
    
    def __init__(
        self,
        lang: str ='pt',
        processors: str ='tokenize,mwt,lemma',
        tokenize_pretokenized: bool =True
    ):
        self.model = stanza.Pipeline(
            lang=lang,
            processors=processors,
            tokenize_pretokenized=tokenize_pretokenized,
            logging_level='ERROR'
        )
        
    def lemmatize(self, text: Union[str, List], debug: bool = False) -> List[str]:
        doc = self.model(text)
        if debug:
            print(
                *[
                    f'word: {word.text} \tlemma: {word.lemma}' 
                    for sent in doc.sentences for word in sent.words
                ],
                sep='\n'
            )
        
        stopwords = get_stopwords()
        lemmatized = [
            word.lemma for sentence in doc.sentences
            for word in sentence.words if word.lemma not in stopwords
        ]
        return lemmatized

    def lemmatize_iter(
        self,
        text: Union[str, List],
        iterations: int = 2,
        debug: bool = False
    ) -> Any: 
        doc = text
        for _ in range(iterations):
            doc = self.lemmatize(doc, debug=debug)
        return doc


if __name__ == '__main__':

    from auto_diagnostic.lemmatization import Lemmatization
    from auto_diagnostic.preprocess import tokenize

    from dataloader.makedata import Dataloader
    
    df = Dataloader().make_csv()
    tokenized = tokenize(df['text_column'])
    model = Lemmatization()
    lemmatized_list = model.lemmatize(tokenized)
    
    print(lemmatized_list)
