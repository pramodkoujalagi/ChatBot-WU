from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

class TfidfEmbeddings:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectors = None

    def fit(self, texts):
        self.vectorizer.fit(texts)

    def embed_documents(self, texts):
        if not hasattr(self.vectorizer, 'vocabulary_'):
            raise ValueError("Vectorizer not fitted. Call fit() before embedding documents.")
        return self.vectorizer.transform(texts).toarray()

    def embed_query(self, text):
        if not hasattr(self.vectorizer, 'vocabulary_'):
            raise ValueError("Vectorizer not fitted. Call fit() before embedding query.")
        return self.vectorizer.transform([text]).toarray()[0]

    def __call__(self, text):
        # This method makes the class callable
        if isinstance(text, str):
            return self.embed_query(text)
        elif isinstance(text, list):
            return self.embed_documents(text)
        else:
            raise ValueError("Input must be a string or a list of strings")

    def embed_many(self, texts):
        # Alias for embed_documents to match expected interface
        return self.embed_documents(texts)