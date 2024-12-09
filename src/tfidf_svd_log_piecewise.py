from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
from scipy.sparse import vstack
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
                                         revision = "svd_log_piecewise",
                                         release_date = "2024-10-01",
                                         languages = [])

    def encode(self, sentences: list[str], vocab: list[str], 
               V: np.ndarray = np.array([]), dtype = np.float64, **kwargs: Any
    ) -> torch.Tensor | np.ndarray:
        """Encodes the given sentences using the encoder.
        TF-IDF is performed 'piecewise', for each batch of the dataset.
        the TF-IDF representations of the batches are then concatenated and svd-reduced
        the V matrix is returned and to be used in a later call of this function, 
        where it is multiplied with the tfidf representation of each batch to map them to reduced space

        THIS IS CURRENTLY ONLY COMPATIBLE WITH CLUSTERING TASKS
        ALL OTHER TASKS WILL BREAK

        Args:
            sentences: The sentences to encode.
            V: TruncatedSVD.components_ 
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        # initialize the model
        vectorizer = TfidfVectorizer(dtype=dtype, sublinear_tf=True, vocabulary = vocab)
        # fit on data
        if type(sentences) == list:
            # single batch passed
            print("computing tfidf features for single batch")
            sent_vec = vectorizer.fit_transform(sentences)
        else:
            # whole dataset passed
            print("computing tfidf features for entire dataset")
            sent_vec = None
            for batch in sentences:
                batch_vec = vectorizer.fit_transform(list(batch["sentences"]))
                print(f"tfidf batch matrix shape{batch_vec.shape}")
                if sent_vec == None:
                    sent_vec = batch_vec
                else:
                    sent_vec = vstack([sent_vec, batch_vec])
        
        print(f"tfidf matrix shape{sent_vec.shape}")
        
        
        # check if SVD components are already available
        if V.shape == (0,):
            # this is called if the SVD hasn't been fitted
            print("V not provided, fit SVD to get V")
            svd = TruncatedSVD(n_components=100, algorithm='arpack', random_state=0)
            # fit svd on tfidf representation
            svd.fit(sent_vec)
            # return V (we don't need the transform yet, only the components)
            return svd.components_
        
        else:
            # this is called if components are already available, now we perform the transform
            print("V available, perform SVD dim reduction")
            sent_np = normalize(sent_vec @ V.T)
            print(f"dense matrix shape{sent_np.shape}")
            return sent_np

            
