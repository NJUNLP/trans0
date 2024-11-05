# -*- coding: utf-8 -*-
import numpy as np

from configs.lang_codes import LangCodes

# serving for MCTS results
class Node:
    def __init__(self, state:dict, parent=None): 
        """
        initiate a node with data with parent pointer 
        state = {data:, lang_code:, recon: }
        the lang_code for recon can be tracked by parent's lang_code
        """
        assert "data" in state, "data is required in state"
        assert "lang_code" in state, "lang_code is required in state"
        assert "recon" in state, "recon is required in state"
        self.state = state  
        # the node record for backpropagation, root parent is None
        self.parent = parent 
        # list of children
        self.children = []
        self.value = 0  # win counts for exploration selection
        self.visit = 0 # visit count, for exploration selection

    def add_child(self, state):  # create a child node and return generated node
        node = Node(state, parent=self)
        self.children.append(node)
        return node
    
    def update(self, value):  # update by the visit (explore) results
        self.visit += 1
        self.value += value 

    def get_uct_value(self):
        """
        for exploration: choosing a child with maximum uct value to expand.
        exclude newly created nodes without any visits
        """
        if self.visit == 0:
            return float('inf')  # always visit the unvisited node
        exploit = self.value/self.visit
        if self.parent is None:
            explore = 2.0* (2 * np.log(self.visit) / self.visit) ** 0.5
        else:
            explore = 2.0* (2 * np.log(self.parent.visit) / self.visit) ** 0.5
        return exploit + explore 
    
    def get_best_child(self):
        # return the child with best utc value to expand
        best_child = max(self.children, key=lambda child:child.get_uct_value())
        return best_child

class NaryTree:
    def __init__(self, state):
        self.root = Node(state)  # root is not assigned

    def select(self):  # search from root for the best leaf to expand
        current_node = self.root
        while current_node.children:
            best_child = current_node.get_best_child()
            current_node = best_child
        return current_node
    
    def get_best(self, node):  # retrieve the node with best exploit (value/visit)
        if len(node.children) ==0:
            return node
        
        max_value = node.value/node.visit
        max_node = node
        for child in node.children:
            best_child = self.get_best(child)
            temp_value = best_child.value/best_child.visit
            if temp_value >= max_value:
                max_value = temp_value
                max_node = best_child
        return max_node

    def backpropagate(self,node, value):
        while node:  # update value to that node and backpropagate to the root
            node.update(value)
            node = node.parent

    def add_child(self, parent, child_data):
        """
        expand the parent node with new child, and return the child node
        """        
        child = Node(state=child_data, parent=parent)
        parent.children.append(child)
        return child
            
    def preorder_traversal(self, node=None, value_type="utility"):  # collect all data by preorder_traversal
        if node is None:
            node = self.root
        
        results = []
        if value_type =="utility":
            value = node.value/node.visit
        elif value_type =="value":
            value = node.value
        elif value_type =="visit":
            value = node.visit
        elif value_type =="uct":
            value = node.get_uct_value()
        results.extend([(node.state["data"], value)])
        for child in node.children:
            results.extend(self.preorder_traversal(child))
        return results
    
    def postorder_traversal(self, node=None, value_type="utility"):  # for value estimation training from leaves.
        assert value_type in ["utility", "value", "visit", "uct"], "mcts traversal value type must be in utility, visit, or uct"
        if node is None:
            node = self.root
        # return type as key or 
        results = []
        for child in node.children:
            results.extend(self.postorder_traversal(child))
        if value_type =="utility":
            value = node.value/node.visit
        elif value_type =="value":
            value = node.value
        elif value_type =="visit":
            value = node.visit
        elif value_type =="uct":
            value = node.get_uct_value()
        results.extend([(node.state["data"], value)])
        return results

    def layer_traversal(self,node=None, value_type="utility"):
        assert value_type in ["utility", "value", "visit", "uct"], "mcts traversal value type must be in utility, visit, or uct"
        if node is None:
            node = self.root
        results = []
        
        q = [node]
        while q:
            item_to_read = q.pop()
            if value_type =="utility":
                value = item_to_read.value/item_to_read.visit
            elif value_type =="value":
                value = item_to_read.value
            elif value_type =="visit":
                value = item_to_read.visit
            elif value_type =="uct":
                value = item_to_read.get_uct_value()
            results.extend([(item_to_read.state["data"], value)])

            q.extend(item_to_read.children)
        return results
    