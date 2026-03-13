import pandas as pd
import numpy as np

df = pd.read_parquet("Retrieval\\data\\train_ready.parquet")

print("weight nan:", df["weight"].isna().sum())

print("user numeric inf:",
      np.isinf(df[[
        "addtocart_rate",
        "purchase_rate"
      ]]).sum())

print("item numeric inf:",
      np.isinf(df[[
        "cart_rate",
        "purchase_rate_item"
      ]]).sum())