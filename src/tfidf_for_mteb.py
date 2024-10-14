from sklearn.feature_extraction.text import TfidfVectorizer
from typing import Any

# tf-idf class in MTEB syntax
class Tfidf():
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

        return sent_np
