from rules_mining.Helper import itemset_2_string, string_2_itemset

class AssociationRule:
    def __init__(self, left, right):
        self.left_items = left
        self.right_items = right
        self.scores = []
        
    def length(self):
        return len(self.left_items) + len(self.right_items)
     
    def rankingScore(self, index):
        return self.scores[index]
    
    def lhs_string(self):
        return itemset_2_string(self.left_items)
        
    def rhs_string(self):
        return itemset_2_string(self.right_items)
    
    def serialize(self):
        left_key = self.lhs_string()
        right_key = self.rhs_string()
        return left_key + ">" + right_key
    
    def __lt__(self, other):
        return self.length() < other.length()

    @staticmethod        
    def string_2_rule(s):
        subStrings = s.split(">")
        left = string_2_itemset(subStrings[0].strip())
        right = string_2_itemset(subStrings[1].strip())
        return AssociationRule(left, right)

    def append_score(self, rankingScore):
        self.scores.append(rankingScore)
        
    def get_itemset(self):
        itemset = []
        itemset.extend(self.left_items)
        itemset.extend(self.right_items)
        itemset.sort()
        return itemset
        
        
    def itemset_string(self):
        itemset = self.get_itemset()
        return itemset_2_string(itemset)
    
    '''
    def compute_probs(self, observations_dict):
        ntransactions = observations_dict.ntransactions + 2  
        lhs_frequency, rhs_frequency, both_frequency = observations_dict.get_frequency_tuple(self)

        pleft = lhs_frequency /ntransactions
        pright = rhs_frequency/ntransactions 
        pboth = both_frequency/ntransactions
        
        #normalized_len = len(self.left_items)/observations_dict.length_of_max_itemset
            
        #return [pboth/pleft, pleft, pright, pboth, pleft * pright, item_min, item_max, average, normalized_len]
        return [pleft, pright, pboth, pleft * pright]
            
    '''

    def compute_probs(self,observations_dict):
        ntransactions = observations_dict.ntransactions  
        feature_vector = []
        lhs_frequency, rhs_frequency, both_frequency = observations_dict.get_frequency_tuple(self)
        
        #1. P(AB)
        p_A_and_B = both_frequency/ntransactions
        #feature_vector['AB'] = p_A_and_B
        feature_vector.append(p_A_and_B)
        
        
        
        #2. P(A)
        p_A = lhs_frequency/ntransactions
        #feature_vector['A'] = p_A
        feature_vector.append(p_A)
        
        #3. P(B|A)
        p_B_if_A = 0
        if p_A_and_B > 0:
            p_B_if_A = p_A_and_B / p_A
        #feature_vector['B|A'] = p_B_if_A
        feature_vector.append(p_B_if_A)
        
        
        #4. P(B)
        p_B = rhs_frequency/ntransactions
        #feature_vector['B'] = p_B
        feature_vector.append(p_B)
        
        '''
        c = p_B - p_A_and_B
        if c == 0: feature_vector.append(c)
        else: feature_vector.append(c/(1-p_A))
        '''
        
        #5. P(~A)
        p_not_A = 1 - p_A
        #feature_vector['~A'] = p_not_A
        feature_vector.append(p_not_A)
        
        #6. P(~B)
        p_not_B = 1 - p_B
        #feature_vector['~B'] = p_not_B
        feature_vector.append(p_not_B)
        
        
        #7. P(~AB)
        p_not_A_and_B = (rhs_frequency - both_frequency)/ntransactions
        #feature_vector['~AB'] = p_not_A_and_B
        feature_vector.append(p_not_A_and_B)
        
        #8. P(A~B)
        p_A_and_not_B = (lhs_frequency - both_frequency)/ntransactions
        #feature_vector['A~B'] = p_A_and_not_B
        feature_vector.append(p_A_and_not_B)
        
        #9. P(~A~B)
        p_not_A_and_not_B = 1 - (lhs_frequency + rhs_frequency -  both_frequency)/ntransactions
        #feature_vector['~A~B'] = p_not_A_and_not_B
        feature_vector.append(p_not_A_and_not_B)
        
        
        #9. P(A)*P(B)
        feature_vector.append(p_A * p_B)
        
        '''
        #10. P(AB)*P(~A~B)
        feature_vector.append(p_A_and_B * p_not_A_and_not_B)
        
        #11. P(A~B) * P(~AB)
        feature_vector.append(p_A_and_not_B * p_not_A_and_B)
        
        #12. P(AB) * P(~B)
        feature_vector.append(p_A_and_B * p_not_B)
        #13. P(A~B) * P(B)
        feature_vector.append(p_A_and_not_B * p_B)
        '''
        '''
        c = []
        for i in range(len(feature_vector)):
            for j in range(i + 1, len(feature_vector)):
                c.append(feature_vector[i] * feature_vector[j])
        feature_vector.extend(c)
        ''' 
    
        
        
        #10. P(A|B)
        p_A_if_B = 0
        if (p_A_and_B > 0): 
            p_A_if_B = p_A_and_B / p_B
        #feature_vector['A|B'] = p_A_if_B
        feature_vector.append(p_A_if_B)
        
        #11. P(~A|~B)
        p_not_A_if_not_B = 0
        if p_not_A_and_not_B > 0:
            p_not_A_if_not_B = p_not_A_and_not_B / p_not_B
        #feature_vector['~A|~B'] = p_not_A_if_not_B
        feature_vector.append(p_not_A_if_not_B)
        
        #12. P(A|~B)
        p_A_if_not_B = 0
        if p_A_and_not_B > 0:
            p_A_if_not_B = p_A_and_not_B/p_not_B
        #feature_vector['A|~B'] = p_A_if_not_B
        feature_vector.append(p_A_if_not_B)
        
        #13. p(~A|B)
        p_not_A_if_B = 0
        if p_not_A_and_B > 0:
            p_not_A_if_B = p_not_A_and_B / p_B
        #feature_vector['~A|B'] = p_not_A_if_B
        feature_vector.append(p_not_A_if_B)
        
        #14. p(~B|A)
        p_not_B_if_A = 0
        if p_A_and_not_B > 0:
            p_not_B_if_A = p_A_and_not_B / p_A
        feature_vector.append(p_not_B_if_A)

        
        return feature_vector
    
    
    def is_redundant(self, bit_array, position, itemset, observations_dict): 
        '''
        Run out of items --> create rule and check format criterion
        '''
        if position >= len(itemset):
            items_1 = []
            items_2 = []
            for index in range(len(bit_array)):
                if bit_array[index] == True:
                    items_1.append(itemset[index])
                else:
                    items_2.append(itemset[index])
            for item in items_2:
                rule = AssociationRule(items_1, [item])
                confidence = observations_dict.get_confidence(rule)
                if confidence == 1: return True
            return False 
      
        value_domain = [True, False]
        for value in value_domain:
            bit_array[position] = value
            checker = self.is_redundant(bit_array, 
                                        position+1, 
                                        itemset, 
                                        observations_dict)
            if checker == True: return True
            bit_array[position] = True    
        return False
    
    '''
    Expand an item-set with equivalent items.
    '''
    def is_redundant_rule(self, observations_dict):
        bit_array = [True for _ in self.left_items]
        checker = self.is_redundant(bit_array, 
                                    0, 
                                    self.left_items, 
                                    observations_dict)
        if checker == True: return True
        
        bit_array =  [True for _ in self.right_items]
        return self.is_redundant(bit_array, 
                                 0, 
                                 self.right_items, 
                                 observations_dict)
        
        
    '''
    Check if an item-set is satisfied condition of the rule. 
    '''
    def is_satisfied(self, itemset):
        return set(self.left_items) < set(itemset)