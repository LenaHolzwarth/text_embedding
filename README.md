# text_embedding
Text embedding benchmarking for TF-IDF methods using an adapted version of MTEB, which can be found at https://github.com/LenaHolzwarth/sparse_mteb. The original MTEB can be found at https://github.com/embeddings-benchmark/mteb. 

Folder structure:

- img: plots for report
- MTEB: results for all experiments
    - clustering comparison: the newest version of MTEB uses different datasets and a different clustering algorithm (hierarchical clustering) to evaluate the models. These are the results for GloVe for both versions of each clustering dataset
    - knn_results: TF-IDF results for k-nearest-neighbor classification on the MTEB clustering datasets
    - knn_results_custom_vocab: this uses a custom function to create the vocabulary instead of the one used by sklearn.feature_extraction.text.TfidfVectorizer, because the scikit-learn one doesn't exclude numbers, which leads to unnessecarily large embeddings as each number is embedded as a vocabulary item. However, to ease comparability with other TF-IDF results, we decided to stick to the scikit-learn implementation.
    - results: MTEB scores for the NN-based models
    - sparse results: MTEB scores for the TF-IDF embeddings
    - typos: results of the typo investigation for the different clustering datasets, to find out why reddit behaves differently to the other datasets. This was not investigated further.
    - vocab_test_results: results for different vocabulary creation set-ups of TF-IDF (varying max features, max df, min df and ngram range)
    - vocabs: the vocabularies that are the result of different vocabulary settings (this was an idea that was not investigated further bc of time constraints)
- notebooks
    - clustering_tasks_comparison: compare old and new version of the MTEB clustering tasks
    - dict_label_to_color_reddit.pkl: color bindings for reddit labels
    - display_results_clustering/ MTEB / vocab: plots
    - knn: run knn classification on MTEB clustering datasets
    - mteb_test (2): getting used to MTEB interface
    - reddit: investigating the label distribution of clustering datasets
    - tfidf_mteb_setup: log of the adaption of MTEB for TF-IDF
    - vocab_comparison: compares the custom vocab function with the one used by sklearns TF-IDF vectorizer
- scripts
    - mteb_run_all_tfidf: run all selected MTEB tasks for all TF-IDF variants
    - mteb_run: query individual models or tasks
    - mteb_vocab_test: run all selected MTEB tasks for different vocabulary creation variants
- src: model configuration files and knn accuracy. Each evaluated version of TF-IDF (different dimensionality reduction techniques and reduction dimensions) has its own file, this is not really necessary (lots of copied code), but this way it is easier to keep an overview of which model was used for which experiment, and you can go back to check how each model was implemented.
