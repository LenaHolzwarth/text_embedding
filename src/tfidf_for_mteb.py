from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing import Any
import torch
import numpy as np
import re
from mteb.model_meta import ModelMeta


# tf-idf class in MTEB syntax
class Tfidf():

    def __init__(self):
        #self.model_card_data = ModelCard(name = "Tfidf", revision = "0.1")
        #self.similarity_fn_name = "" #cosine is the default
        self.mteb_model_meta = ModelMeta(name = "Tfidf", 
                                         revision = "1.0",
                                         release_date = "2024-10-01",
                                         languages = [])

    def encode(self, sentences: list[str], vocab: list[str] = [], 
               dtype = np.float64, n_documents: int = 1, 
               svd_threshold: int = 1000000, **kwargs: Any
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
            print("no vocab provided, computed by sklearns TfidfVectorizer call")
            vectorizer = TfidfVectorizer(dtype=dtype)
        else: 
            vectorizer = TfidfVectorizer(dtype=dtype, vocabulary = vocab)
        # fit on data
        sent_vec = vectorizer.fit_transform(sentences)
        print(f"tfidf matrix shape{sent_vec.shape}")

        # if sparse rep. or total document number is too large, transform to smaller dim using svd
        if n_documents > svd_threshold or sent_vec.shape[0] > svd_threshold: # 1 mio as arbitrary cutoff, might vary
            print(f"number of total documents ({n_documents}) or currently transformed documents ({sent_vec.shape} too large, use svd reduction)")
            svd = TruncatedSVD(n_components=100, algorithm='arpack', random_state=0)
            sent_np = normalize(svd.fit_transform(sent_vec))

        else:
            # transform sparse matrix to numpy array
            sent_np = sent_vec.toarray()

        print(f"dense matrix shape{sent_np.shape}")
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