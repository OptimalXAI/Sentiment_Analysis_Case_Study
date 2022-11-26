# import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
# word vectorization 
'''
# Bag of words Vectorizer
class BowVectors(BaseEstimator,TransformerMixin):
    def __init__(self):
        print('calling--init--') 
        self.bow_vectorizer = CountVectorizer()
    def fit(self,df):
        print('calling fit')
        return self
    def transform(self, df):
        print('calling transform')
        features = self.bow_vectorizer.transform(df)
        return self.bow_vectorizer, features 
'''
'''# Tfidf Vectorizer
class TfidfVectors():
    def __init__(self): 
        self.tfidf_vectorizer = TfidfVectorizer(lowercase=False,ngram_range=(1,1))
    def __call__(self, df):
        features = self.tfidf_vectorizer.fit_transform(df)
        return self.tfidf_vectorizer, features

class Vectorizer:
    
    def __init__(self, model):
        
        self.vectorizer = model["VECT"]
                                             
            
    def fit_transform(self, text_array):
        
        try:
            X = self.vectorizer.fit_transform(text_array)
        
        # in case [text_array] was passed as an argument
        except:
            X = self.vectorizer.fit_transform([text_array])
         
        return X
         
    def transform(self, text_array):
        
        try:
            X = self.vectorizer.transform(text_array)
            
        # in case [text_array] was passed as an argument
        except:
            X = self.vectorizer.transform(text_array[0])
        
        return X
'''