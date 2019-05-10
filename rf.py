import pandas as pd
import numpy as np
from typing import Union, Optional

class TreeEnsemble():
    def __init__(self, x: pd.DataFrame, y: np.array, 
                       x_valid: pd.DataFrame=pd.DataFrame(), y_valid: pd.DataFrame=pd.DataFrame(),
                       num_trees: int=1, min_leaf: int=5, rf_samples: Union[int, float]=.2):
        """Creates an ensemble of decision trees
           Takes x,y training data, num_trees in ensemble, min_leaf of each
           decision tree and rf_samples as a percentage of samples of training
           data to give to each decision tree
        """
        self.x, self.y, self.x_valid, self.y_valid = x, y, x_valid, y_valid
        self.num_trees, self.min_leaf = num_trees, min_leaf
        self.rf_samples = int(rf_samples * len(self.x))
        
    def fit(self):
        """Creates and fits forest. Afterwards computes predictions on 
           training and validation set (if applicable) and stores various metrics
        """
        self.trees = [self.create_tree() for i in range(self.num_trees)]
        self.preds_trn_mean, self.preds_trn_std = self.predict(self.x)
        self.preds_val_mean, self.preds_val_std = self.predict(self.x_valid) if len(self.x_valid) > 0 else (None, None)
        self.r2_trn = 1 - np.sum((self.preds_trn_mean - self.y)**2) / np.sum((self.y.mean(axis=0) - self.y)**2)
        self.r2_val = 1 - np.sum((self.preds_val_mean - self.y_valid)**2) /\
                      np.sum((self.y_valid.mean(axis=0) - self.y_valid)**2) if len(self.x_valid) > 0 else None
        return self
    
    def create_tree(self):
        """Creates a decision tree with a random sample of training data 
           of length self.rf_samples
        """
        idx_rand = np.random.choice(np.arange(0, len(x_train), 1), size=self.rf_samples, replace=False)
        x_rand = self.x.iloc[idx_rand]
        y_rand = self.y[idx_rand]
        return DecisionTree(x_rand, y_rand, self.min_leaf)
    
    def predict(self, x: pd.DataFrame):
        """Returns a prediction and standard deviation of predictions for a given x
        """
        self.preds = np.array([tree.predict(x) for tree in self.trees])
        return self.preds.mean(axis=0), self.preds.std(axis=0)
    
    @property
    def feature_importance(self):
        """Computes feature importance use training data by randomly shuffling one dep var at
           a time and computing the decrease in
        """
        if not self._fit:
            self.fit()
        if hasattr(self, '_fi'):
            pass
        else:        
            self._fi = pd.DataFrame(columns=['var', 'importance'])
            for var in self.x.columns:
                ##NOTE this is very inefficient because of DecisionTree.predict
                s_tmp = self.x[var].copy()
                s_shuff = self.x[var].sample(frac=1)
                self.x[var] = s_shuff.values #set column to shuffled values
                preds, _ = self.predict(self.x)
                self.x[var] = s_tmp.values #set column back to normal values
                r2 = 1 - np.sum((preds - self.y)**2) / np.sum((self.y.mean(axis=0) - self.y)**2)
                self._fi = self._fi.append({'var':var, 'importance':self.r2_trn-r2}, ignore_index=True)
        return self._fi.sort_values('importance', ascending=False)
            
            
    
    @property
    def r2(self):
        """Outputs training and validation r2 if exists
        """
        if self._fit:
            print("R2 training: {0} R2 validation: {1}".format(self.r2_trn, self.r2_val))
        else:
            print("Please run fit to get R2")
            
    @property
    def _fit(self):
        """Returns boolean if model has been fit or not
        """
        if hasattr(self, 'trees'):
            return True
        else:
            return False
        
class DecisionTree():
    def __init__(self, x: pd.DataFrame, y: np.array, min_leaf: int=5):
        """Creates a decision tree given x,y training data and min_leaf
        """
        self.x, self.y = x,y
        self.min_leaf = min_leaf
        self.val = self.y.mean(axis=0)
        self.score = float('inf')
        self.l_split = None
        self.r_split = None
        self.var = None
        self.split = None
        
        self.split_all()
    
    def split_var(self, var: str):
        """Finds best split for a given variable
        """
        for i in range(len(self.x)):
            l_split = self.x[var] <= self.x[var].iloc[i]
            r_split = ~l_split
            if l_split.sum() < self.min_leaf or r_split.sum() < self.min_leaf: continue
            l_std = self.y[l_split].sum()
            r_std = self.y[r_split].sum()
            score = l_std * l_split.sum() + r_std * l_split.sum() #minimze weighted sum of standard deviation of splits
            if score < self.score:
                #if score improved, store variable, score and split value
                self.var, self.score, self.split = var, score, self.x[var].iloc[i]
    
    def split_var_fast(self, var: str):
        """Finds best split for a given variable, faster than split_var
        """
        idx_sort = np.argsort(self.x[var]) #get the sorted index
        x_sort, y_sort = self.x[var].iloc[idx_sort], self.y[idx_sort]
        r_count, r_sum, r_sum2 = len(self.x), y_sort.sum(), (y_sort**2).sum()
        l_count, l_sum, l_sum2 = 0., 0., 0.
        
        for i in range(r_count - self.min_leaf):
            xi, yi = x_sort.iloc[i], y_sort[i]
            
            #manually increment each value
            l_count += 1 
            l_sum += yi
            l_sum2 += yi**2
            r_count -= 1
            r_sum -= yi
            r_sum2 -= yi**2
            
            if i + 1 < self.min_leaf or xi == x_sort.iloc[i + 1]:
                #skip if we haven't iterated over more than min_leaf indexes
                #or if current x value equals next x value (redundant)
                continue
            
            #np.sqrt((xi - xbar)**2 / n) can be rewritten as np.sqrt((xi**2/n - (x/n)**2))
            l_std = np.sqrt(l_sum2/l_count - (l_sum/l_count)**2)
            r_std = np.sqrt(r_sum2/r_count - (r_sum/r_count)**2)
            score = l_std * l_count + r_std * r_count
            if score < self.score:
                #if score improved, store variable, score and split value
                self.var, self.score, self.split = var, score, xi
            
                
    def split_all(self):
        """Finds best var and value to split on
        """
        for var in self.x.columns:
            self.split_var_fast(var)
        if self.is_leaf:
            return
        l_x_train = self.x.loc[self.x[self.var] <= self.split]
        r_x_train = self.x.loc[self.x[self.var] > self.split]
        l_y_train = self.y[self.x[self.var] <= self.split]
        r_y_train = self.y[self.x[self.var] > self.split]
        self.l_split = DecisionTree(l_x_train, l_y_train, self.min_leaf)
        self.r_split = DecisionTree(r_x_train, r_y_train, self.min_leaf)
    
    def predict_row(self, xi: pd.Series):
        """Makes a prediction on a single row xi
        """
        if self.is_leaf: return self.val
        tree = self.l_split if xi[self.var] <= self.split else self.r_split
        return tree.predict_row(xi)
    
    def predict(self, x: pd.DataFrame):
        """Makes a prediction on a dataframe
        """
        return np.array([self.predict_row(row) for i,row in x.iterrows()])
    
    @property
    def is_leaf(self):
        """Checks if we're done splitting
        """
        return self.score == float('inf')
    
    def __repr__(self):
        if not self.is_leaf:
            return "{0} <= {1} \n samples = {2} \n value = {3}".format(self.var, 
                                                                       self.split, 
                                                                       len(self.x),
                                                                       self.val)
        else:
            return "samples = {0} \n value = {1}".format(len(self.x),
                                                         self.val)
        
