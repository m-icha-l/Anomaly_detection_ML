import logging
from datetime import datetime

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


log("START",level = "start")


def prep_df(df,df_labels):
    df.columns = df.columns.str.strip()
    df = df.drop(["Fwd Header Length.1"], axis = 1)

    #df = df.drop_duplicates()
    
    len_1 = len(df)
    nans = df.isna().any(axis=1).sum()
    print("Number of flows: " , len(df))
    print("Number of flows with NAN: " , nans)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    infs = df.isna().any(axis=1).sum() - nans
    print("Number of flows with inf or -inf : " , infs)
    
    df.dropna(inplace=True)
    len = len_1 - len(df)
    print(len, "Flows deleted ,","flows in df Friday-WorkingHours-Afternoon-DDos after preparation : ", len(df) )
    
    df_labels = df["Label"]
    df = df.drop(["Label"], axis = 1)
    df_labels = df_labels.reset_index(drop=True)



