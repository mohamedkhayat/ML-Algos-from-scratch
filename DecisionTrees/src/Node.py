class Node:
    def __init__(self, feature_idx = None, threshold=None, left=None, right=None, *, value=None):
        self.value = value
        self.left = left
        self.right = right
        self.feature_idx = feature_idx
        self.threshold = threshold
        
    def is_leaf_node(self):
        return (self.value is not None)