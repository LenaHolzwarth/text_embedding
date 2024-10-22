from sklearn.feature_extraction.text import TfidfVectorizer
from mteb.model_meta import ModelMeta
from typing import Any
import torch
import numpy as np

class ModelCard():
    def __init__(self, name: str, revision: str):
        self.model_name = name
        self.language = "any"
        self.base_model_revision = revision


# tf-idf class in MTEB syntax
class Tfidf():

    def __init__(self):
        #self.model_card_data = ModelCard(name = "Tfidf", revision = "0.1")
        #self.similarity_fn_name = "" #cosine is the default
        self.mteb_model_meta = ModelMeta(name = "Tfidf", 
                                         revision = "1.0",
                                         release_date = "2024-10-01",
                                         languages = [])
        
        
    

    def encode(self, sentences: list[str], **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        # initialize the model
        vectorizer = TfidfVectorizer()
        # fit on data
        sent_vec = vectorizer.fit_transform(sentences)
        # transform sparse matrix to numpy array
        sent_np = sent_vec.toarray()
        # transform numpy array to tensor
        #sent_np = torch.from_numpy(sent_np)

        return sent_np
