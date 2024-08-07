import os
import pandas as pd
# import autogluon.eda.auto as auto

class file_load :
    """_summary_

    Requirement :
        module : pandas, xgboost
        linux terminal setting : kaggle. If your envirment doesn't have kaggle command, try this command to your terminal; conda install -c conda-forge kaggle / pip install kaggle
    Returns:
        _type_: _description_
    """
    competition = ""
    directory = None
    reqest = ""
    
    ## pandas dataframe
    df_train = None
    df_test = None
    df_submit = None
        
    def __init__(self, comp, path = None) :
        """_summary_

        Args:
            comp (_type_): _Describe kaggle competition data API. For example "kaggle competitions download -c playground-series-s4e5"_
            path (_type_): _Conditional argument. Directory for data working space._
        """
        self.method = comp.split()[0]
        self.competition = comp.split()[-1]
        
        ## Processing bar input
        if path == None :
            self.directory = "./data"
        elif (path[-1] == "/") or (path[-1] == "\\") :
            self.directory = str(path[:-1])
        else :
            self.directory = str(path)
            
        self.reqest = comp
        
    def __str__(self) :
        return f"competition = {self.competition}\ndirectory = {self.directory}"
        
    def load_tr_tst_sub_data(self) :
        try :
            os.system("chmod 600 /root/.kaggle/kaggle.json")
            os.system(self.reqest)        
            os.system(f"unzip {self.competition} -d {self.directory}")
            self.df_train = pd.read_csv(self.directory+"/train.csv")
            self.df_test = pd.read_csv(self.directory+"/test.csv")
            self.df_submit = pd.read_csv(self.directory+"/sample_submission.csv")
            os.system(f"rm -rf {self.directory}")  ## dangerous code
            os.system(f"rm {self.competition}.zip")
            
            return (self.df_train, self.df_test, self.df_submit)
        
        except :
            os.system(f"rm -rf {self.directory}")  ## dangerous code
            os.system(f"rm {self.competition}.zip")
        
    def submit_file(self, df_submit, commit = None) :
        df_submit.to_csv("submission.csv", index = False)
        if commit == None :
            os.system(f"kaggle competitions submit -c {self.competition} -f submission.csv -m .")
        else :
            os.system(f"kaggle competitions submit -c {self.competition} -f submission.csv -m {commit}")
        
        os.system(f"rm submission.csv")
    
# class eda :
#     def auto_eda(self, train) :

# class xg_fitting :