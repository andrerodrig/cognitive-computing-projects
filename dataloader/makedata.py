import glob
import fitz
import unidecode
import pandas as pd

class Dataloader():
    """
        Metodo principal make_csv

    """
    def __init__(self, path):
        self.path = path

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
            df[text_field][i]=unidecode.unidecode(df[text_field].iloc[i])
        df[text_field] = df[text_field].str.lower()
        df[text_field] = df[text_field].str.replace(r"[()\;\,\%\-\/\--\.!?@\'\`\"\_\n]", " ")

        return df

    def load(self) -> list:
        retorno = []
        for filename in glob.glob(self.path+'*.pdf'):
            retorno.append(self._load(filename))

        return retorno

    def make_csv(self):
        data = self.load()
        _df = pd.DataFrame(data=data)
        _df.reset_index(inplace=True)
        _df.rename(columns={0:'text_column','index':'txt_idx'},inplace=True)
        _df = self._padronizacao(_df, 'text_column')
        return _df