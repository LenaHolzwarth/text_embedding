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


# task selection
# all
selection_all = ["ArguAna", "ArxivClusteringP2P", "BiorxivClusteringP2P", "MedrxivClusteringP2P", "MindSmallReranking",
                "RedditClusteringP2P", "SCIDOCS", "SciDocsRR", "StackExchangeClusteringP2P", "STS15", "STS16",
                "STSBenchmark"]
# only clustering   
selection_cluster = ["ArxivClusteringP2P", "BiorxivClusteringP2P", "MedrxivClusteringP2P", "RedditClusteringP2P", "StackExchangeClusteringP2P"]



# free GPU memory
gc.collect()
torch.cuda.empty_cache()

# directory for sparse results
results_folder = "/gpfs01/berens/user/lholzwarth/text_embedding/MTEB/sparse_results"


#"tfidf": src.tfidf_for_mteb.Tfidf(),
#"tfidf_log": src.tfidf_log.Tfidf(),
#"tfidf_svd": src.tfidf_svd.Tfidf(),
#"tfidf_svd_log": src.tfidf_svd_log.Tfidf(),
#"tfidf_svd50_log": src.tfidf_svd50_log.Tfidf(),
#"tfidf_svd200_log": src.tfidf_svd200_log.Tfidf(),
#"tfidf_svd300_log": src.tfidf_svd300_log.Tfidf(),
#"tfidf_svd500_log": src.tfidf_svd500_log.Tfidf(),
#"tfidf_svd_log_piecewise": src.tfidf_svd_log_piecewise.Tfidf(),
#"tfidf_svd_log_novocab": src.tfidf_svd_log_novocab.Tfidf(),
#"tfidf_svd_log_old": src.tfidf_svd_log_old.Tfidf(),
#"tfidf_rnd100_log": src.tfidf_rnd100_log.Tfidf(),
#"tfidf_rnd500_log": src.tfidf_rnd500_log.Tfidf(),
#"tfidf_rnd768_log": src.tfidf_rnd768_log.Tfidf()
model_dict = {
"tfidf_svd768_log": src.tfidf_svd768_log.Tfidf()
}
 

for model_name, model in model_dict.items():

    logger.info(f"evaluating model {model_name}")
    print(model_name)

    # define tasks
    if model_name in ["tfidf_svd_log_piecewise", "tfidf_svd_log_old", "tfidf_svd_log_novocab"]:
        tasks = mteb.get_tasks(tasks=selection_cluster)
    else:
        tasks  = mteb.get_tasks(tasks=selection_all)

    # run each task consecutively
    for task in tasks:
        logger.info(f"started task {task}")
        if model_name == "tfidf_svd500_log" and task.metadata.name == 'MindSmallReranking':
            continue
        evaluation = mteb.MTEB(tasks=[task])
        results = evaluation.run(model, 
                                    output_folder=results_folder,
                                    encode_kwargs = {"ngram_range": (1,1), 
                                                     "max_df": 1.0, 
                                                     "min_df": 1, 
                                                     "max_features": None})
        logger.info(f"ended task {task}")

        # free GPU memory 
        gc.collect()
        torch.cuda.empty_cache()
