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
# dict to map model names to their SentenceTransformer names
model_shortcuts = {"glove": "sentence-transformers/average_word_embeddings_glove.6B.300d",
                   "komninos": "average_word_embeddings_komninos",
                   "mpnet": "microsoft/mpnet-base",
                   "sbert" : "sentence-transformers/all-mpnet-base-v2",
                   "scincl": "malteos/scincl",
                   "simcse_u": "princeton-nlp/unsup-simcse-bert-base-uncased",
                   "specter": "allenai/specter",
                   "t5xxl": "sentence-transformers/sentence-t5-xxl"}


# get user input
model_name = input("\nmodel name (one of 'tfidf', 'glove', 'komninos', 'sbert' or full SentenceTransformer names)")

task_type = input("\ntask type (one of 'b' for benchmark, 't' for task suite or 's' for single task)")

if task_type == "b":
    task_name = input("\nbenchmark name")
    tasks = mteb.get_benchmark(task_name).tasks
elif task_type == "t":
    task_name = input("\ntask subtype(s)").split()
    tasks = mteb.get_tasks(task_types=task_name)
elif task_type == "s":
    task_name = input("\ntask name(s)").split()
    tasks = mteb.get_tasks(tasks=task_name)
elif task_type == "selection":
    selection = ["ArguAna", "ArxivClusteringP2P", "BiorxivClusteringP2P", "MedrxivClusteringP2P", "MindSmallReranking",
                 "RedditClusteringP2P", "SCIDOCS", "SciDocsRR", "StackExchangeClusteringP2P", "STS15", "STS16",
                 "STSBenchmark"]
    tasks = mteb.get_tasks(tasks=selection)
else:
    raise Exception(f"{task_type} is invalid task type. Must be one of 'b', 't' or 's'")

# free GPU memory
gc.collect()
torch.cuda.empty_cache()

# define the model
if model_name in model_shortcuts:
    model_name = model_shortcuts[model_name]

# directory for sparse results
results_folder = "/gpfs01/berens/user/lholzwarth/text_embedding/MTEB/sparse_results"

if model_name == "tfidf":
    model = src.tfidf_for_mteb.Tfidf()
elif model_name == "tfidf_log":
    model = src.tfidf_log.Tfidf()
elif model_name == "tfidf_svd":
    model = src.tfidf_svd.Tfidf()
elif model_name == "tfidf_svd_log":
    model = src.tfidf_svd_log.Tfidf()
elif model_name == "tfidf_svd200_log":
    model = src.tfidf_svd200_log.Tfidf()
elif model_name == "tfidf_svd_log_old":
    model = src.tfidf_svd_log_old.Tfidf()
elif model_name == "tfidf_rnd100_log":
    model = src.tfidf_rnd100_log.Tfidf()
elif model_name == "tfidf_rnd768_log":
    model = src.tfidf_rnd768_log.Tfidf()
else:
    model = mteb.get_model(model_name)
    #results_folder = "/gpfs01/berens/user/lholzwarth/text_embedding/MTEB/results"
logger.info(f"evaluating model {model_name}")

# run each task consecutively
for task in tasks:
    logger.info(f"started task {task}")
    try:
        evaluation = mteb.MTEB(tasks=[task])
        results = evaluation.run(model, 
                                 output_folder=results_folder,
                                 encode_kwargs = {'batch_size': 64})
        logger.info(f"ended task {task}")
    except Exception as e:
        logging.error(traceback.format_exc())
        #or logger.exception()
        raise

    # free GPU memory 
    gc.collect()
    torch.cuda.empty_cache()
