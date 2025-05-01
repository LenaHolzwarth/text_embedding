from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from typing import Any, List
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
                                         revision = "rnd500_log",
                                         release_date = "2024-10-01",
                                         languages = [])

    def encode(self, sentences: list[str], vocab: list[str] = [], 
               dtype = np.float64, **kwargs: Any
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
            vectorizer = TfidfVectorizer(dtype=dtype, sublinear_tf=True)
        else: 
            vectorizer = TfidfVectorizer(dtype=dtype, sublinear_tf=True, vocabulary = vocab)
        # fit on data
        sent_vec = vectorizer.fit_transform(sentences)
        print(f"tfidf matrix shape{sent_vec.shape}")

        # transform to smaller dim using random projections
        np.random.seed(42)
        P = 2 * np.random.randint(0, 2, size=(sent_vec.shape[1], 500)) - 1
        sent_np = sent_vec @ P  

        print(f"dense matrix shape{sent_np.shape}")
        # transform numpy array to tensor
        #sent_np = torch.from_numpy(sent_np)

        return sent_np
    