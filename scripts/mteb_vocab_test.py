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
file_handler = logging.FileHandler("error_logs.log")
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
ngram_range = (1, 2)
max_df = 1.0
min_df = 1
max_features = None


# directory for sparse results
results_folder = f"/gpfs01/berens/user/lholzwarth/text_embedding/MTEB/vocab_test_results/ngram_{ngram_range[0]}-{ngram_range[1]}_maxdf_{int(max_df)}-{int(round(max_df%1, 2)*10)}_mindf_{int(min_df)}-{int(round(min_df%1, 2)*10)}_maxfeatures_{max_features}"



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


    