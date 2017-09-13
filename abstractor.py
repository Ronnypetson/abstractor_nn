class Node:
    def __init__(self,id_):
        self.id = id_   # Integer
        self.is_front = None   # Boolean

class Abstractor:    
    def __init__(self,parents,child,w1,w2):
        if(len(parents) != 2):
            print('There must be 2 parents.')
            return
        self.parents = parents
        self.child = child
        self.parents[0].is_front = False
        self.parents[1].is_front = False
        self.child.is_front = True

class NetGraph:
    def __init__(self,input_len,output_len):
        self.input_len = input_len
        self.output_len = output_len
        self.adj = []   # Adjacency list
        self.abs = []   # List of abstractors
        for i in range(input_len):
            self.adj.append([]) # The first input_len nodes are the input
        

