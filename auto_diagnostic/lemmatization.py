from typing import List, Union
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
            tokenize_pretokenized=tokenize_pretokenized
        )
        
    def lemmatize(self, text: Union[str, List]):
        doc = self.model(text)
        print(
            *[
                f'word: {word.text} \tlemma: {word.lemma}' 
                for sent in doc.sentences for word in sent.words
            ],
            sep='\n'
        )
        return [word.lemma for sentence in doc.sentences for word in sentence.words]