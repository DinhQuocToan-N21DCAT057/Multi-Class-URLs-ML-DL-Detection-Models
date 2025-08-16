from operator import index
from pickle import TRUE
import pandas as pd
import os

from torch import threshold

from url_multi_labels_predictor import URL_PREDICTOR

# from configs.config import Config
# if Config.HF_TOKEN:
#     os.environ.setdefault("HF_TOKEN", Config.HF_TOKEN)
#     os.environ.setdefault("HUGGINGFACE_TOKEN", Config.HF_TOKEN)
#     os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", Config.HF_TOKEN)

URL_PREDICTOR.preload(['llama-32-1b-lora', 'bert_non'])
# # URL_PREDICTOR.preload_scaler()
# URL_PREDICTOR.preload_vectorizers(['xgb_rf'])

test_urls = [
    "https://uis.ptithcm.edu.vn/#/home"
]

for url in test_urls:
    print(f"\n{'='*60}")
    print(f"Testing: {url}")
    print('='*60)
    
    predictor = URL_PREDICTOR(url)
    
    # LLaMA
    predictor.predict_with_LLaMA_LoRA()
    print("LLaMA-LoRA:")
    predictor.print_result()
    
    # BERT  
    predictor.predict_with_TF_BERT()
    print("\nBERT:")
    predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_XGB(threshold=0.85, numerical=True)
# #predictor.df.to_csv("test.csv")
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_XGB(threshold=0.85, numerical=False)
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_RF(threshold=0.85, numerical=True)
# predictor.print_result()

# predictor = URL_PREDICTOR(url1)
# predictor.predict_with_RF(threshold=0.85, numerical=False)
# predictor.print_result()


