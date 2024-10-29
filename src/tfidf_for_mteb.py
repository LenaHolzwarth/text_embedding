from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any
import torch
import numpy as np
import re
#from mteb.model_meta import ModelMeta
import sys
sys.path.append("../..")
from sparse_mteb.mteb.model_meta import ModelMeta


# tf-idf class in MTEB syntax
class Tfidf():

    def __init__(self):
        #self.model_card_data = ModelCard(name = "Tfidf", revision = "0.1")
        #self.similarity_fn_name = "" #cosine is the default
        self.mteb_model_meta = ModelMeta(name = "Tfidf", 
                                         revision = "1.0",
                                         release_date = "2024-10-01",
                                         languages = [])

    def encode(self, sentences: list[str], vocab: list[str] = [], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        # initialize the model
        if vocab == []:
            vectorizer = TfidfVectorizer()
        else: 
            vectorizer = TfidfVectorizer(vocabulary = vocab)
        # fit on data
        sent_vec = vectorizer.fit_transform(sentences)
        # transform sparse matrix to numpy array
        sent_np = sent_vec.toarray()
        # transform numpy array to tensor
        #sent_np = torch.from_numpy(sent_np)

        return sent_np


# helper function to extract vocab
def get_vocab(text: [str], token_pattern: str = r"(?u)\b\w\w+\b", lowercase: bool = True) -> [str]:
        """return a list of unique words ocurring in text that fulfill the specified token_pattern
        The default token_pattern is the one used in the scikit-learn TfidfVectorizer class
        """
        if lowercase:
            vocab = [word for sent in text for word in re.findall(token_pattern, sent.lower())]
        else:
            vocab = [word for sent in text for word in re.findall(token_pattern, sent)]

        return list(set(vocab))