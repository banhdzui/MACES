'''
Created on 19 Mar 2018

@author: danhbuithi
'''
import sys
import numpy as np
import time

from sklearn.metrics import f1_score

from common.CommandArgs import CommandArgs
from common.DataSet import DataSet

from rules_mining.RuleMiner import RuleMiner
from rules_mining.AssociationRule import AssociationRule

from rule_based_classifiers.InterestingnessLearner import InterestingnessLearner
    

def evaluateByF1(y_pred, y_true):
    a = f1_score(y_true, y_pred, average='micro')
    b = f1_score(y_true, y_pred, average='macro')
    return (a, b)
                 
            
    
def separateRulesAndFeatures(rule_feature_dict):
    rules = []
    X = []
    for key, value in rule_feature_dict.items():
        rules.append(key)
        X.append(value)
    return rules, np.array(X)
    

def preprocessRuleFeatureDict(rule_feature_dict):
    rules, features = separateRulesAndFeatures(rule_feature_dict)
    rule_full_list = [AssociationRule.string_2_rule(x) for x in rules]
    return rule_full_list, features, features[:, 0]
    
if __name__ == '__main__':
    config = CommandArgs({
                          'train'   : ('', 'Path of training data file'),
                          'test'   : ('', 'Path of testing data file'),
                          'class'   : (0, 'Class index'),
                          'minsup'  : (0.1, 'Minimum support'),
                          'nloop'   : (100, 'Number of loops'),
                          'lambda'  : (0.1, 'Lambda value'),
                          'beta'    : (0.01, 'Beta value')
                          })    
    
    if not config.load(sys.argv):
        print ('Argument is not correct. Please try again')
        sys.exit(2)
    
    min_conf = 0.0
    rule_format = 'spect'
        
    class_index = int(config.get_value('class'))
    train_data = DataSet()
    train_data.load(config.get_value('train'), class_index)
    
    test_data = DataSet()
    test_data.load(config.get_value('test'), class_index)
    min_sup = float(config.get_value('minsup'))
    
    nloop = int(config.get_value('nloop'))    
    labels = sorted(train_data.count_classes().keys())
    my_lambda = float(config.get_value('lambda'))
    my_beta = float(config.get_value('beta'))
    
    '''
    Generate association rules
    '''
    rule_miner = RuleMiner(rule_format, train_data.create_dataset_without_class())
    rule_miner.generate_itemsets_and_rules(min_sup, min_conf)
    rule_feature_dict = rule_miner.load_rules_features_as_dictionary()
    print('#rules ', len(rule_feature_dict))
    
    freq_itemsets_dict = rule_miner.load_freq_itemset_dictionary()
    rule_list, rule_features, rule_supports = preprocessRuleFeatureDict(rule_feature_dict)
    print('#filtered rules ', len(rule_list))
    train_args = {'data': train_data, 'rule': rule_list, 'label': labels, 'feature': rule_features, 'sup': rule_supports}
    cmar_es = InterestingnessLearner(train_args, class_weight=False, my_beta=my_beta)
    
    start = time.time()
    assoc_classifier = cmar_es.fit(my_lambda, 3, max_iters=nloop)
    end = time.time()
    print('execution time for interestingness learning: ' , end - start)
    
    
    print('Testing ....')
    print(np.unique(train_data.data_labels, return_counts = True))
    print(np.unique(test_data.data_labels, return_counts = True))
    
    
    print('Use learned measure: ')
    bnn_pred_test = assoc_classifier.predict(test_data)
    test_result = evaluateByF1(bnn_pred_test, test_data.data_labels)
    
    bnn_pred_train = assoc_classifier.predict(train_data)
    train_result = evaluateByF1(bnn_pred_train, train_data.data_labels)
            
    print("train ", train_result)
    print("test ", test_result)
    assoc_classifier.myPrint()


