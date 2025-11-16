    
from Tree import BaseDecisionTree
import numpy as np

class DecisionTreeClassifier(BaseDecisionTree):
    def __init__(self, max_depth=3, min_samples_split=2,random_state=42):
        super().__init__(max_depth,min_samples_split,random_state,"classification")
        pass
    
    def _score_function(self,y):
        if len(y) <= 1:
            return 0
        
        _, counts = np.unique(y, return_counts=True)
        
        probabilities = counts / len(y)

        return 1 - np.sum(probabilities ** 2)
    
    def _assign_value(self, y):
        if len(y) == 0:
            return None 
        
        clss, counts = np.unique(y, return_counts=True)
        return clss[np.argmax(counts)]
