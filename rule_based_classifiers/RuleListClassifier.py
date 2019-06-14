'''
Created on 19 Nov 2018

The class represents an object which is a list of rule playing as a classifier. 
The rules have to be sorted; the rule going first in doing classification. 
This classifier supports two modes for prediction: using only the first rule that the sample satisfies its LHS or using multiple rules.

@author: danhbuithi
'''

import numpy as np 
from collections import Counter
from sklearn.metrics.classification import accuracy_score

class RuleListClassifier(object):
    '''
    classdocs
    '''

    def __init__(self, rules, labels, default_class = None):
        '''
        Constructor
        '''
        self.rules = rules
        self.default_class = default_class
        self.labels = labels
        
        
    def size(self):
        return len(self.rules) + 1
        
    def myPrint(self):
        for rule_info in self.rules:
            print((rule_info['r'].serialize(), rule_info['ins'], rule_info['sup']))
        if self.default_class is not None:
            print(self.default_class)
        print('--------------------------')
        
    def _predictOne(self, x):
        for rule_info in self.rules:
            r = rule_info['r']
            if r.is_satisfied(x):
                return r.right_items[0]
            
        if self.default_class is None: return 'unknown'
        return self.default_class['r']
     
    def predict(self, data_set):
        Y = []
        for i in range(data_set.size()):
            label = self._predictOne(data_set.get_transaction(i))
            Y.append(label)
        return Y
    
    def meanLength(self):
        total = 0
        n = self.size()
        for rule_info in self.rules:
            total += len(rule_info['r'].left_items)/n
        return total
    
    def meanSupport(self):
        total = 0
        for rule_info in self.rules:
            if (rule_info['sup'] < 1e-6):
                total += 1
        return total
    
    
    def cost(self, data,is_debug = False):
        y_pred = self.predict(data)
        return 1 - accuracy_score(data.data_labels, y_pred)
    
    
    @staticmethod
    def globalDefaultClass(data):
        un_classified_classes = data.count_classes()
        default_class = max(un_classified_classes, key=un_classified_classes.get)
        return default_class, un_classified_classes[default_class]
        
    @staticmethod
    def localDefaultClass(data, indices):
        un_classified_classes = Counter([data.data_labels[i] for i in indices])
        default_class = max(un_classified_classes, key=un_classified_classes.get)
        return default_class, un_classified_classes[default_class]
    
    @staticmethod
    def getDefaultClass(cover_counter, data):
        default_class = None
        indices = np.where(cover_counter ==0)[0]
        if len(indices) > 0:
            default_class, _ = RuleListClassifier.localDefaultClass(data, indices)
        else:
            default_class, _ = RuleListClassifier.globalDefaultClass(data)
            
        return default_class
            
class SingleRuleBasedClassifier(RuleListClassifier):
    
    def __init__(self, rules, labels, default_class = None):
        '''
        Constructor
        '''
        RuleListClassifier.__init__(self, rules, labels, default_class)

class MMultiRuleBasedClassifier(RuleListClassifier):
    
    def __init__(self, rules, labels, default_class = None, coverage_thresh = 3):
        '''
        Constructor
        '''
        RuleListClassifier.__init__(self, rules, labels, default_class)
        self.coverage_threshold = coverage_thresh
    
        
    def _predictOne(self, x):
        label_dict = {key: 0 for key in self.labels}
        
        counter = 0
        for rule_info in self.rules:
            if counter >= self.coverage_threshold: break
            r = rule_info['r']
            f = rule_info['ins']
            if r.is_satisfied(x):
                label_dict[r.right_items[0]] = max(f,label_dict[r.right_items[0]])
                counter += 1
                
        if counter == 0:
            if self.default_class is None: return 'unknown'
            return self.default_class['r']
       
        p = [(value, key) for key, value in label_dict.items()]
        return max(p)[1] 
    
        
class MultiRuleBasedClassifier(RuleListClassifier):
    
    def __init__(self, rules, labels, default_class = None, coverage_thresh = 3):
        '''
        Constructor
        '''
        RuleListClassifier.__init__(self, rules, labels, default_class)
        self.coverage_threshold = coverage_thresh
        
    def _predictOne(self, x):
        label_dict = {key: 0 for key in self.labels}
        nlabels = len(self.labels)
        counter = 0
        for rule_info in self.rules:
            if counter >= self.coverage_threshold: break
            r = rule_info['r']
            f = rule_info['ins']
            if r.is_satisfied(x):
                label_dict[r.right_items[0]] += f
                for k in self.labels: 
                    if k != r.right_items[0]: label_dict[k] += (1-f)/(nlabels-1) 
                counter += 1
                
        if counter == 0:
            if self.default_class is None: return 'unknown'
            return self.default_class['r']
        #print(str(label_dict))
        c = [(value, key) for key, value in label_dict.items()]    
        return max(c)[1]
    
    def cost(self, data, weights,is_debug = False):
        
        y_pred = []
        nlabels = len(self.labels)
        
        for i in range(data.size()):
            
            counter = 0
            x = data.get_transaction(i)
                  
            label_dict = {key: 0 for key in self.labels}
            #count_dict = {key: 0 for key in self.labels}
            
            for rule_info in self.rules:
                if counter >= self.coverage_threshold: break
                r = rule_info['r']
                f = rule_info['ins']
                if r.is_satisfied(x):
                    label_dict[r.right_items[0]] += f
                    for k in self.labels: 
                        if k != r.right_items[0]: label_dict[k] += (1-f)/(nlabels-1) 
                    counter += 1
            '''
            pred_probs = {key: 0 for key in self.labels}
            for k in self.labels:
                if count_dict[k] > 0: pred_probs[k] = label_dict[k]/count_dict[k]
            '''
            if counter == 0 and self.default_class is not None:
                label_dict[self.default_class['r']] = 1.0
            
            a = label_dict[data.data_labels[i]]
            b = np.array([t for t in label_dict.values()])
            
            c = a/np.sum(b)
            #if is_debug: 
            #    print(c, data.data_labels[i], str(b))
            
            y_pred.append(c)
            
        w = np.array([weights[y] for y in data.data_labels])
        return np.sum(-np.log(np.array(y_pred) + 1e-8) * w)/np.sum(w)

        