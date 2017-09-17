import numpy as np
import tensorflow as tf

class AbGraph:
    def __init__(self,input_len,output_len):
        self.id_count = input_len+output_len
        self.input_len = input_len
        self.output_len = output_len
        self.nodes = set() # Nodes identification
        self.front = set()  # Nodes that can be removed
        self.abstractors = {}   # Only one abstractor per pair of nodes
        self.X = tf.placeholder(tf.float32,shape=(None,input_len))
        self.Y = tf.placeholder(t.float32,shape=(None,output_len))
        self.weights = {}    # Weights and biases of each abstractor
        self.biases = {}
        self.activations = {}   # the output of the nodes
        for i in range(self.id_count):
            self.nodes.add(i)   # 0 to input_len-1, input_len to input_len+output_len-1
            self.front.add(i)

    def insert_ab(self,ab_in):   # ab_in must be tuple of shape (2)
        for i in ab_in:
            if i not in self.nodes:
                return False
        self.nodes.add(self.id_count)
        self.abstractors[ab_in] = self.id_count
        for i in ab_in:
            if i in self.front:
                self.front.remove(i)
        self.front.add(self.id_count)
        self.id_count += 1
        return True

    def delete_ab(self,ab_in):    # Can only delete from the front
        y = self.abstractors[ab_in]
        if ab_in in self.abstractors and y in self.front:
            del self.abstractors[ab_in]
            self.front.remove(y)
            self.nodes.remove(y)
            return True
        return False

    def build_model(self):
        # Setup parameters
        for k in self.abstractors.keys():   # Weights of abstractor k
            self.weights[k] = tf.Variable(tf.random_normal( (len(k),1) )) # mean = 0.0, sd = 1.0
            self.biases[k] = tf.Variable(0.0)
        # Define activations
        for i in range(self.input_len):
            self.activations[i] = self.X[:,i]   #
        for k in self.abstractors.keys():
            parents = tf.Variable([0.0]*len(k))    #
            for i in range(len(k)):
                parents[i] = self.activations[i]
            child = self.abstractions[k]
            self.activations[child] = tf.nn.elu(tf.matmul(parents,self.weights[k])+)

#g = AbGraph(5,3)
#print(g.insert_ab((0,1)))
#for i in g.abstractors.keys():
#    print(i,g.abstractors[i])
#print(g.delete_ab((0,1)))

