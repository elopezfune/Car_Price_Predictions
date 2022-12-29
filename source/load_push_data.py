import json
import pandas as pd


# =======================================================================================================
# LOADING THE DATA
# =======================================================================================================
# Loads file provided their path
# ==============================
def load_data(path,index=None):
    # Loads the data
    if path.suffix[-3:] == 'csv':
        df = pd.read_csv(path)
    elif path.suffix[-4:] == 'json':
        with open(path) as f:
            g = json.load(f)
        # Converts json dataset from dictionary to dataframe
        df = pd.DataFrame.from_dict(g)
    else:
        print('No files with this extension exist in the provided directory')
    # Convert column names to title
    #df.columns = df.columns.str.title()
    # Sets index column if needed
    if index !=None:
        df = df.set_index(index)
    return df    
# =======================================================================================================
# LOADING THE DATA
# =======================================================================================================