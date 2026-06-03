from huggingface_hub import login
import os
import logging

import accelerate
import transformers
import torch
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try :
    hf_token = os.getenv("HUGGINGFACE_MODEL_ACCESS_TOKEN")
    logger.info("Hugging face token loaded successfully")

    login(token=hf_token)
    logger.info("Huggingface login successful")
except AttributeError or ValueError as aerr:
    logger.error("Huggingface token not found or the token is invalid")
except HTTPError as herr:
    logger.error("HTTP error occurred while logging into Huggingface")
except GatedRepoError as gerr:
    logger.error("Gated repository. You do not have access to this model/dataset")

logger.info("accelerate:", accelerate.__version__)   # should be 1.x+
logger.info("transformers:", transformers.__version__)