import logging
from datetime import datetime
import pandas as pd
import numpy as np


current_date = datetime.now().strftime("%Y-%m-%d")

logging.basicConfig(
    filename=f'logs/Isolation_forest_{current_date}.log',
    filemode='a',
    format='%(message)s', 
    level=logging.INFO
)

formatter = logging.Formatter('%(asctime)s ', datefmt='%Y-%m-%d %H:%M:%S')

def log(txt, level="info"):
    txt = str(txt)

    if level == "start":
        # Tworzy rekord loga ręcznie i formatujemy go jako tekst
        record = logging.LogRecord(name="log", level=logging.INFO, pathname="", lineno=0, msg="", args=(), exc_info=None)
        
        log_header = formatter.format(record)
        
        
        num = int((102 - len(txt))/2)
        if(num <= 0):
            num = 1 

        ending ='-' * num
            
        logging.info("\n\n<-" + ending +"[LOG] " + log_header + txt +" [LOG]" + ending + "->")  
    else:
        logging.info(txt)
        print(txt)





def prep_df(df, labels = False, only_a = False):
    df.columns = df.columns.str.strip()
    df.drop(["Fwd Header Length.1"], axis = 1, inplace = True)

    df_labels = pd.DataFrame()
    #df = df.drop_duplicates()
    
    len_1 = len(df)
    nans = df.isna().any(axis=1).sum()
    print("Number of flows: " , len(df))
    print("Number of flows with NAN: " , nans)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    infs = df.isna().any(axis=1).sum() - nans
    print("Number of flows with inf or -inf : " , infs)
    print("Number of flows labled as Attac: ", (~df['Label'].str.contains('BENIGN', na=False)).sum())
    df.dropna(inplace=True)
    len_2 = len_1 - len(df)
    print(len_2, "Flows deleted","\nNumber flows in after preparation : ", len(df))
    print("Number of flows labled as Attac after preperation: ", (~df['Label'].str.contains('BENIGN', na=False)).sum())

    
    if(only_a == True):
        print("\nCreating df with everything that is labled having an attac")
        
        df_a = df[~df['Label'].str.contains('BENIGN', na=False)].copy()
                
        df_a.columns = df_a.columns.str.strip()
        df_a.replace([np.inf, -np.inf], np.nan, inplace=True)
        df_a.dropna(inplace=True)
        df_a.drop(["Label"], axis = 1, inplace = True)
        df_a.reset_index(drop=True, inplace = True)
        
        print("number of flows in Attac only dataframe : ", len(df_a))

    if(labels ==True):
        
        df_labels = df["Label"]
        df_labels.reset_index(drop=True, inplace=True)
          
    df.drop(["Label"], axis = 1, inplace=True)
    df.reset_index(drop=True, inplace = True)

    if(labels == True and only_a == True):
        return df_labels,df_a
    elif(labels == True):
        return df_labels
    elif(only_a == True):
        return df_a
    
        



































