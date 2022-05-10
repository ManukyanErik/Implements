import numpy as np

class DecisionTress:
    def __init__(self, criterion='gini', max_depth=3, min_size=1):
        self.crit = criterion
        self.max_depth = max_depth
        self.min_size = min_size

    def get_split(self, x):
        b_score = 1
        b_index = None
        b_value = None
        class_values = x.shape[1]-1
        
        for index in range(class_values):
            for value in np.unique(x[:, index])[:-1]:
                l_x = x[x[:, index] <= value]
                r_x = x[x[:, index] >  value]
                score = self.criterion(l_x, r_x)
                if score < b_score:
                    b_score, b_index, b_value, b_groups = score, index, value, [l_x, r_x]

        return {'index':b_index, 'value':b_value, 'groups':b_groups}
    
    def criterion(self, l_x, r_x):
        if self.crit == 'gini':
            group_1 = []
            group_2 = []
            for label in self.class_labels:
                group_1.append(l_x[l_x[:, -1] == label].shape[0])
                group_2.append(r_x[r_x[:, -1] == label].shape[0]) 

            if sum(group_1) == 0:
                gini_score_1 = 0
            elif sum(group_2) == 0:
                gini_score_2 = 0
            else:
                gini_score_1 = (1-sum((np.array(group_1)/sum(group_1))**2)) * l_x.shape[0]/(l_x.shape[0]+r_x.shape[0])
                gini_score_2 = (1-sum((np.array(group_2)/sum(group_2))**2)) * r_x.shape[0]/(l_x.shape[0]+r_x.shape[0])
            
            return gini_score_1 + gini_score_2

        if self.crit == 'entropy':
            pass


    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)


    # Create child splits for a node or make terminal
    def split(self, node, max_depth, min_size, depth):
        left, right = node['groups']
        del(node['groups'])
        # check for a no split
        # if not left or not right:
        #     node['left'] = node['right'] = self.to_terminal(left + right)
        #     return
        # check for max depth
        if depth >= max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], max_depth, min_size, depth+1)
        # process right child
        if len(right) <= min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], max_depth, min_size, depth+1)
    
    
    # Build a decision tree
    def build_tree(self, train, max_depth, min_size):
        root = self.get_split(train)
        self.split(root, max_depth, min_size, 1)
        return root

    
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.class_labels = np.unique(np.array(self.y_train))
        x =  np.hstack((X_train, y_train.reshape(-1, 1)))

        self.stump = self.build_tree(x, self.max_depth, self.min_size)
        

    def pred(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.pred(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.pred(node['right'], row)
            else:
                return node['right']
        
    def predict(self, X_test):
        result = []
        for row in list(X_test):
            prediction = self.pred(self.stump, row)
            result.append(prediction)
        return result