from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Um transformador para remover colunas indesejadas
class DropColumns(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        # Primeiro realizamos a cópia do DataFrame 'X' de entrada
        data = X.copy()
        # Retornamos um novo dataframe sem as colunas indesejadas
        return data.drop(labels=self.columns, axis='columns')

class CategorizeColumns(BaseEstimator, TransformerMixin):
    def __init__(self): # Respectivamente
        return
        
    def try_float(self, number):
        try:
            number = float(number)
            return True
        except ValueError:
            return False
    
    def correct_float(self, number):
        try:
            number = float(number)
            return number
        except ValueError:
            return 0
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        
        data['CHECKING_BALANCE_CATEGORICAL'] = (X['CHECKING_BALANCE'].apply(self.try_float) == False).apply(lambda x: 1 if x else 0)
        data['CHECKING_BALANCE'] = X['CHECKING_BALANCE'].apply(self.correct_float).fillna(0.0)
        
        data['EXISTING_SAVINGS_CATEGORICAL'] = (X['EXISTING_SAVINGS'].apply(self.try_float) == False).apply(lambda x: 1 if x else 0)
        data['EXISTING_SAVINGS'] = X['EXISTING_SAVINGS'].apply(self.correct_float).fillna(208.53)
        
        return data
    
class EncodeColumns(BaseEstimator, TransformerMixin):
    def __init__(self, encoders, columns): # Respectivamente
        self.enc_col_dict = {col: encoders[i] for i, col in enumerate(columns)}
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        data = X.copy()
        for col in self.enc_col_dict.keys():
            data[col] = self.enc_col_dict[col].transform(data[col])
        return data
