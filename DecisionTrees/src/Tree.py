import numpy as np
from abc import ABC,abstractmethod
from Node import Node

class BaseDecisionTree(ABC):
    def __init__(self, max_depth=3, min_samples_split=2, random_state=42, task_type=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.type = task_type
        np.random.seed(random_state)
        
    def _get_current_gain(self,y,left_indices, right_indices, parent_score):
        n = len(y)
        left_score, right_score  = self._score_function(y[left_indices]), self._score_function(y[right_indices])
        n_left, n_right = len(left_indices),len(right_indices)
        split_score = (n_left / n) * left_score + (n_right / n) * right_score
        current_gain = parent_score - split_score
        return current_gain
    
    def _compare_gain(self, current_gain, best_gain):
        if current_gain > best_gain:
            return True
        return False
    
    @abstractmethod
    def _score_function(self,y):
        pass
    
    def _find_best_split(self, x, y):
        best_gain = -1
        best_threshold = None
        best_feature = None
        
        parent_score = self._score_function(y)
        
        n_features = x.shape[1]
        
        for ftr_idx in range(n_features):
            unique_values = sorted(np.unique(x[:, ftr_idx]))
            for threshold in unique_values:
                left_indices = np.where(x[:, ftr_idx] <= threshold)[0]
                right_indices = np.where(x[:, ftr_idx] > threshold)[0]
            
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
            
                current_gain = self._get_current_gain(y,left_indices, right_indices, parent_score)
                
                if self._compare_gain(current_gain,best_gain):
                    best_threshold = threshold
                    best_feature = ftr_idx
                    best_gain = current_gain
                
        return best_feature, best_threshold
    
    @abstractmethod
    def _assign_value(self, y):
        pass

    def _build_tree(self, x, y, depth=0):
        n_samples = x.shape[0]
        if (depth>=self.max_depth or len(np.unique(y)) == 1 or n_samples < self.min_samples_split):
            leaf_value = self._assign_value(y)
            return Node(value=leaf_value)
        
        best_feature, best_threshold = self._find_best_split(x, y)
        
        if best_feature == None:
            leaf_value = self._assign_value(y)
            return Node(value=leaf_value)
        
        left_indices = np.where(x[:,best_feature] <= best_threshold)[0]
        right_indices = np.where(x[:,best_feature] > best_threshold)[0]

        left_x, left_y = x[left_indices, : ], y[left_indices]
        right_x, right_y = x[right_indices, : ], y[right_indices]

        left_child = self._build_tree(left_x, left_y, depth+1)
        right_child = self._build_tree(right_x, right_y, depth+1)
        
        return Node(best_feature, best_threshold, left_child, right_child)
    
    def fit(self, x, y):
        self.root = self._build_tree(x, y)
        return self
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if (x[node.feature_idx] <= node.threshold):
            return self._traverse_tree(x,node.left)
        else:
            return self._traverse_tree(x,node.right)
    
    def predict(self,X):
        return np.array([self._traverse_tree(x,self.root) for x in X])
