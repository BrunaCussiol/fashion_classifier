import pandas as pd
import json

def read_dfs(train_df_dir, test_df_dir, val_df_dir, map_dir):
  """
  Read train, test and val dataframes and the category dictionary.
  """

  with open(map_dir, 'r') as f:
        cat_map = json.load(f)

  df_test = pd.read_csv(test_df_dir)
  df_train = pd.read_csv(train_df_dir)
  df_val = pd.read_csv(val_df_dir)

  cat_map = {v: k for k,v in cat_map.items()}
  return df_train, df_test, df_val, cat_map

