from utils._1_Imports.reqLibs import *
from utils._2_Cleaning.prepData import *
from utils._3_ModelTraining.trainModel import *
from utils._4_ModelEvaluation.evalModel import *

def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

set_random_seed(10)
file_path = 'CST2208_DataScience_TermProject\Mall Customers Segregration Clustering\Dataset\mall_customers.csv'
df = load_data(file_path)
# check_data(df) Optional
df = train_KmeansModel_with_2features(df)
evaluate_clusters_kmeanswith2features(df)
wss = elbowmethod_to_checkclusters(df)
evaluate_clusters_elbowmethod(wss)
wss = silhouttemethod_to_checkclusters(df,wss) 
evaluate_clusters_silhouettemethod(wss)
Variables3 = train_KmeansModel_with_3features(df)
evaluate_clusters_kmeanswith3features(Variables3)