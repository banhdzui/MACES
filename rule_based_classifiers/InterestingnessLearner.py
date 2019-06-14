'''
Created on 06 Apr 2018

@author: danhbuithi
'''

import numpy as np

from cma import CMAEvolutionStrategy    
from common.ActivateFunctions import sigmoid
from rule_based_classifiers.NetCBA import NetCBA
from collections import Counter
    
class InterestingnessLearner(object):
    '''
    classdocs
    '''
    def __init__(self, train_args, class_weight = False, my_beta=0.01, k = 1):
        self.my_beta = my_beta
        self.train_args = train_args
        
        self.weights = self.getClassWeights(class_weight)
        print(self.weights)
        self.K = k
        
    '''
    Compute weight for each class. It's useful for imbalance datasets.
    '''
    def getClassWeights(self, use_weight):
        weights = {label: 1 for label in self.train_args['label']}
        if use_weight == True:
            nlabels = len(self.train_args['label'])
            nsamples = self.train_args['data'].size()
            
            countings = Counter(self.train_args['data'].data_labels)
            for k, v in countings.items():
                weights[k] = nsamples/(nlabels * v)
                
        return weights 
                
    '''
    Compute the data-driven interestingness with given weight vector
    '''
    def computeInterestingness(self, w):
        b = w[0]
        wt = np.reshape(w[1:], (-1, self.K))
        
        scores = np.max(sigmoid(np.dot(self.train_args['feature'], wt) + b), axis = 1)
        
        rule_list = self.train_args['rule']
        rule_supports = self.train_args['sup']
    
        return [{'r': rule_list[i], 
                 'ins': scores[i], 
                 'sup': rule_supports[i]
                 } 
                 for i in range(len(rule_list))]
       
    '''
    Create CMAR classifier with given weight vector and database coverage
    '''
    def createClassifier(self, w, coverage):
        rule_list = self.computeInterestingness(w)
        return NetCBA().fit(self.train_args['data'], rule_list, self.train_args['label'], coverage_thresh=coverage)
    
           
    '''
    The fitness function for evolution strategy
    '''
    @staticmethod
    def cost(w, *args):
        ranker, my_lambda, coverage, is_debug = args
        return ranker.nmcost(w, my_lambda, coverage, is_debug)

    '''
    The objective function for learning. It is a trade-off between classification cost and the model size.
    '''
    def nmcost(self, w, my_lambda, coverage, is_debug):
        source_model = self.createClassifier( w, coverage)
        score1 = source_model.cost(self.train_args['data'], self.weights, is_debug)
        
        b = self.my_beta * np.linalg.norm(w)**2
        c = my_lambda * source_model.size()     
        if is_debug == True:
            print(score1, source_model.size())
        
        return score1 + b + c
            
        
    '''
    Randomize weight vector and bias. The bias is the first element of the weight vector
    '''
    def randomw0(self, nfeatures):
        
        w0 = np.random.randn(nfeatures * self.K + 1) * 0.01
        w0[0] = 0
        return w0 
           
    '''
    Training a data-driven model based on evolution strategy. 
    ''' 
    def fit(self, my_lambda, coverage=3, max_iters=50):
       
        #while(True):
        print('beta = ' ,self.my_beta)
        print('lambda = ' , my_lambda)
        
        my_args = (self, my_lambda, coverage, False)
        
        nfeatures = self.train_args['feature'].shape[1]
        w0 = self.randomw0(nfeatures)
        
        sigma_value = 0.02
        print('sigma0 ', sigma_value)
        es = CMAEvolutionStrategy(w0, sigma0 = sigma_value, inopts ={'maxiter': max_iters, 'popsize': 40})
 
        while not es.stop():
            solutions = es.ask()
            fitnesses = [InterestingnessLearner.cost(x, *my_args) for x in solutions]
            es.tell(solutions, fitnesses)
            es.disp()
            
            self.nmcost(es.result[0], my_lambda, coverage, is_debug=True)
                            
        final_model = self.createClassifier(es.result[0], coverage)
        
        print(final_model.size(), final_model.meanSupport())
        return final_model
            