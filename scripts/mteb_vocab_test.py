import mteb
from sentence_transformers import SentenceTransformer
import torch
import gc
import logging 
import src

# set up error logging
logger = logging.getLogger(__name__)

# Remove any existing handlers (console handlers, etc.)
if logger.hasHandlers():
    logger.handlers.clear()

logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("error_logs_vocab.log")
formatter = logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("logger set up successfully")

# select tasks
selection = ["ArguAna", "ArxivClusteringP2P", "BiorxivClusteringP2P", "MedrxivClusteringP2P", "MindSmallReranking",
            "RedditClusteringP2P", "SCIDOCS", "SciDocsRR", "StackExchangeClusteringP2P", "STS15", "STS16",
            "STSBenchmark"]

tasks = mteb.get_tasks(tasks=selection)

# select model
model = src.tfidf_svd_log.Tfidf()

# select vocab hyperparameters
ngram_range_ = [(1,1), (1,2), (2,2)]    # length of sequences to tokenize
max_df_ = [0.05, 0.02, 0.01] # ignore terms that have a document frequency higher than this (uninformative b.c. too frequent)
                                        # -> the lower it is set, the more words are ignored
min_df_ = [1, 0.01, 0.02, 0.05, 0.1]                           # ignore terms that have a document frequency lower than this (uninformative b.c. too infrequent)
                                        # -> the higher it is set, the more words are ignored
max_features_ = [None]

# iterate over vocab settings
for ngram_range in ngram_range_:
    for max_df in max_df_:
        for min_df in min_df_:
            # skip iteration if mindf >= maxdf
            if min_df >= max_df and not min_df==1:
                continue
            for max_features in max_features_:

                # directory for sparse results
                results_folder = f"/gpfs01/berens/user/lholzwarth/text_embedding/MTEB/vocab_test_results/ngram_{ngram_range[0]}-{ngram_range[1]}_maxdf_{int(max_df)}-{int(round(max_df%1, 2)*10)}{int(round(max_df%1, 2)*100)}_mindf_{int(min_df)}-{int(round(min_df%1, 2)*10)}{int(round(min_df%1, 2)*100)}_maxfeatures_{max_features}"
 
                # run each task consecutively
                for task in tasks:
                    logger.info(f"started task {task}")

                    # free GPU memory
                    gc.collect()
                    torch.cuda.empty_cache()

                    evaluation = mteb.MTEB(tasks=[task])
                    results = evaluation.run(model, 
                                            output_folder=results_folder,
                                            encode_kwargs = {"ngram_range": ngram_range, 
                                                             "max_df": max_df, 
                                                             "min_df": min_df, 
                                                             "max_features": max_features})
                    logger.info(f"ended task {task}")


    