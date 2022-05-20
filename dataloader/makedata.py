import glob
from pathlib import Path
import fitz
import unidecode
import pandas as pd
from nltk.corpus import stopwords

from auto_diagnostic.config import root_path

class Dataloader():
    """
        Metodo principal make_csv

    """
    def __init__(self, path=str(root_path() / 'data/raw/')):
        self.path = path
        self.stop = stopwords.words('portuguese')

    def _load(self,caminho) -> str:
        """

        Returns:
            conteudo : string
        """
        conteudo = ""
        with fitz.open(caminho) as ff:
            for pagina in ff:
                conteudo += pagina.get_text()
        return conteudo

    def _padronizacao(self,df, text_field):
        for i in range(len(df)):
            df[text_field][i] = unidecode.unidecode(df[text_field].iloc[i])
        df[text_field] = df[text_field].str.lower()
        df[text_field] = df[text_field].str.replace(r"[()\;\,\%\-\/\--\.!?@\'\`\"\_\n]", " ")
        df['txt_sem_stopwords'] = df[text_field].apply(lambda x: ' '.join([word for word in x.split() if word not in (self.stop)]))

        return df

    def load(self) -> list:
        retorno = []
        for filename in glob.glob(str(Path(self.path) / '*.pdf')):
            retorno.append(self._load(filename))

        return retorno

    def make_csv(self, save: bool = True):
        data = self.load()
        _df = pd.DataFrame(data=data)
        _df.reset_index(inplace=True)
        _df.rename(columns={0:'text_column','index':'txt_idx'},inplace=True)
        _df = self._padronizacao(_df, 'text_column')
        if save:
            _df.to_csv(root_path() / 'data/processed/processed_data1.csv', index=False)
        return _df


if __name__ == '__main__':
    data = Dataloader()
    print(data.make_csv())