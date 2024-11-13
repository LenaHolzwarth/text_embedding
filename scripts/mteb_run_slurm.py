import mteb
from sentence_transformers import SentenceTransformer
import torch
import gc
import logging 
import sys

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


# check if command line arguments exist
if len(sys.argv) == 1: # no command line args

    # get user input
    model_name = input("\nmodel name (one of 'glove', 'komninos', 'sbert' or full SentenceTransformer names)")
    task_type = input("\ntask type (one of 'b' for benchmark, 't' for task suite or 's' for single task)")
    task_name = input("\ntask name")

elif len(sys.argv) == 4: # correct number of command line args
    model_name, task_type, task_name = sys.argv[1], sys.argv[2], sys.argv[3] 

else: # invalid number of command line args
    raise Exception(f"Invalid user input: {str(sys.argv)}; input should be [script, model_name, task_type, task_name]")

if task_type == "b":
    tasks = mteb.get_benchmark(task_name).tasks
elif task_type == "t":
    tasks = mteb.get_tasks(task_types=task_name.split())
elif task_type == "s":
    tasks = mteb.get_tasks(tasks=task_name.split())
else:
    raise Exception(f"{task_type} is invalid task type. Must be one of 'b', 't' or 's'")

# define the model
if model_name in model_shortcuts:
    model_name = model_shortcuts[model_name]
model = mteb.get_model(model_name)

# run each task consecutively
for task in tasks:
    logger.info(f"started task {task}")
    try:
        evaluation = mteb.MTEB(tasks=[task])
        results = evaluation.run(model, 
                                 output_folder="/home/berens/bep216/text_embedding/MTEB/results",
                                 encode_kwargs = {'batch_size': 64})
        logger.info(f"ended task {task}")
    except Exception as e:
        logging.error(traceback.format_exc())
        #or logger.exception()
        raise

    # free GPU memory 
    gc.collect()
    torch.cuda.empty_cache()
