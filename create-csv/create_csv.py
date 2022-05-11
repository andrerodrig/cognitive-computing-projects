import pandas as pd

def create_csv(list_tokens, list_TF, list_TF_IDF):
    """
        Funcao responsavel por gerar um arquivo csv com as colunas de tokens, TF e TF-IDF
        Arguments:
          list_tokens: str list -- Uma lista de str com todos os tokens
          list_TF: float list -- Uma lista dos valores de TF
          list_TF_IDF: float list -- Uma lista lista dos valores de TF-IDF
        
        Return:
          
    """

    df = pd.DataFrame({'Tokens': list_tokens,
                       'TF': list_TF,
                       'TF-IDF': list_TF_IDF})
    
    df.to_csv('dataframe.csv',index=False)

