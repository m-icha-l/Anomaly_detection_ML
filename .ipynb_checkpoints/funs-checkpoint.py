import logging
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report
import seaborn as sns

file_name = "untitled"

def log(txt, level="info"):

    global file_name
    if isinstance(level, (tuple, list)):
        file_name = level[1]
        
    current_date = datetime.now().strftime("%Y-%m-%d")

    logging.basicConfig(
        filename=f'logs/{file_name}_{current_date}.log',
        filemode='a',
        format='%(message)s', 
        level=logging.INFO
    )

    formatter = logging.Formatter('%(asctime)s ', datefmt='%Y-%m-%d %H:%M:%S')
    
    txt = str(txt)

    if level[0] == "start":
        
        print(file_name)
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
################################
    columns_to_keep = [
    'Bwd Packet Length Max',
    'Bwd Packet Length Mean',
    'Bwd Packet Length Std',
    'Flow Bytes/s',
    'Flow Duration',
    'Flow IAT Max',
    'Flow IAT Mean',
    'Flow IAT Min',
    'Flow IAT Std',
    'Fwd IAT Total',
    'Fwd Packet Length Max',
    'Fwd Packet Length Mean',
    'Fwd Packet Length Min',
    'Fwd Packet Length Std',
    'Total Backward Packets',
    'Total Fwd Packets',
    'Total Length of Bwd Packets',
    'Total Length of Fwd Packets',
    'Label',
    ]
    
    # Drop columns NOT in columns_to_keep (inplace)
    cols_to_drop = [col for col in df.columns if col not in columns_to_keep]
    df.drop(columns=cols_to_drop, inplace=True)

##########################################
    
    df_labels = pd.DataFrame()
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

##########################################################################################              2
    # Count duplicate rows and add a column 'Repetition num' with the count
    df_counts = df.groupby(list(df.columns)).size().reset_index(name='Repetition num')
    
    # Replace original df with grouped version (deduplicated with repetition count)
    df.drop(df.index, inplace=True)  # Clear existing rows in df
    for col in df_counts.columns:
        df[col] = df_counts[col].values  # Fill df with new data
##########################################################################################              2
    
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
    print("flag 3")
    df.reset_index(drop=True, inplace = True)

    if(labels == True and only_a == True):
        return df_labels,df_a
    elif(labels == True):
        return df_labels
    elif(only_a == True):
        return df_a
    
def test_IF_model(model_IF,df_scaled,df_labels, name="df"):
    
    preds = model_IF.predict(df_scaled)
    scores = model_IF.score_samples(df_scaled)
    df_scaled_result = pd.DataFrame()
    df_scaled_result['prediction'] = preds
    df_scaled_result['anomaly_score'] = scores
    
    
    anomaly = (df_scaled_result["prediction"] == -1).sum()
    normal = (df_scaled_result["prediction"] == 1).sum()
    
    log(f"3.2 Wizualizacja efektu testu na danych {name}",level=["start","Isolation_forest"])
    log(f"Detection distribution anomaly/normal: {anomaly / len(df_scaled_result) * 100:.4f}% / {normal / len(df_scaled_result) * 100:.4f}%")
    log("!!! Wyniki pokazuje dopasowanie contamination (wyciagania wnioskow z gotowego produktu pracy modelu) NIE DCHYLENIA NORMALNY/ANOMALYJNY ruch sieciowy !!!")
    log(f"Anomaly detected: {anomaly}")
    log(f"Normal traffic: {normal}") 
    
    
    plt.figure(figsize=(14, 6))  # szeroki wykres
    
    plt.scatter(df_scaled_result.index, df_scaled_result['anomaly_score'], color='blue', label='Anomaly Score', s=10)
    
    plt.title('Anomaly Scores for All Data Points (IsolationForest)')
    plt.xlabel('Data Point Index')
    plt.ylabel('Anomaly Score')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.ylim(top=0)
    plt.show()

    try:
        df_scaled_result_check = pd.concat([df_scaled.reset_index(drop=True),
                                        df_scaled_result.reset_index(drop=True),
                                        df_labels.reset_index(drop=True)],
                                       axis=1)
    except Exception as e:
        print("Failed to concatenate dataframes:", e)
        return None
    

    log(f"3.3 Sprawdzenie dokładniści wytrenowanego modelu na {name}",level="start")
    
    log(f"Number of flows in {name}: {len(df_scaled_result_check)}")
    
    df_real_anomaly_count = (df_scaled_result_check["Label"] != "BENIGN").sum()
    
    log("Correct data: ")
    log(f"Real number of anomalies in {name}: {df_real_anomaly_count}")
    log(f"Procentage of anomlys in dataset: {df_real_anomaly_count / len(df_scaled_result_check) * 100:.4f}%")
    
    df_predicted_anomaly_count = (df_scaled_result_check["prediction"] == -1).sum()
    
    log("Predicted data: ")
    log(f"Number of DETECTED anomalies in {name}: {df_predicted_anomaly_count}")
    log(f"Procentage of anomlys in dataset: {df_predicted_anomaly_count / len(df_scaled_result_check) * 100:.4f}%")
    
    
    TP = ((df_scaled_result_check["Label"] != "BENIGN") & (df_scaled_result_check["prediction"] == -1)).sum()
    FP = df_predicted_anomaly_count - TP
    FN = df_real_anomaly_count - TP
    TN = len(df_scaled_result_check) - TP - FP - FN
    
    
    accuracy = (TP + TN) / len(df_scaled_result_check)
    sensitivity = TP / df_real_anomaly_count
    precision = TP / df_predicted_anomaly_count

    if precision == 0 or sensitivity == 0:
        f_measure = 0
    else:
        f_measure = (2 * precision * sensitivity) / (precision + sensitivity)
    
    log(f"\nNumber of correct predictions: {TP}")
    log(f"Accuracy of predictions: {accuracy * 100:.4f}%")
    log(f"Sensitivity of predictions: {sensitivity * 100:.4f}%")
    log(f"Precision of predictions: {precision * 100:.4f}%")
    log(f"F - mesure - harmonic-mean of precision and sensitivity: {f_measure* 100:.4f}")

    unique_labels = df_scaled_result_check["Label"].unique().tolist()

    
    stats = []
    
    for label in unique_labels:
        if label == "BENIGN":
            continue
        
        predicted_num = ((df_scaled_result_check["Label"] == label) & 
                         (df_scaled_result_check["prediction"] == -1)).sum()
        real_num = (df_scaled_result_check["Label"] == label).sum()
    
        stats.append({
            "Label": label,
            "predicted_anomaly_count": predicted_num,
            "real_anomaly_count": real_num
        })
    
    
    df_stats = pd.DataFrame(stats)
    df_stats["predicted_percent"] = 100 * df_stats["predicted_anomaly_count"] / df_stats["real_anomaly_count"]

    plt.figure(figsize=(15, 11))
    sns.barplot(
        x="Label",
        y="predicted_percent",
        hue="Label",            
        data=df_stats,
        palette="viridis",
        legend=False            
    )
    
    plt.axhline(100, color='green', linestyle='-', label='100% (All real anomalies)')
    plt.axhline(50, color='gray', linestyle='--', label='50% (All real anomalies)')
    plt.ylim(0, 110)
    plt.ylabel("Predicted Anomalies (% of Real Anomalies)")
    plt.xlabel("Anomaly Type")
    plt.title("Detection Rate per Anomaly Type")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    df_stats.head()
        
    return df_scaled_result_check































