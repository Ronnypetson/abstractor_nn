import numpy as np
import tensorflow as tf
import random as rd

batch_size = 10 #
learning_rate = 0.01
class AbGraph:
    def __init__(self,input_len,output_len):
        self.id_count = input_len   #+output_len
        self.input_len = input_len
        self.output_len = output_len
        self.nodes = set() # Nodes identification
        self.front = set()  # Nodes that can be removed
        self.abstractors = {}   # Only one abstractor per pair of nodes
        self.X = tf.placeholder(tf.float32,shape=(batch_size,input_len))    # None
        self.Y = tf.placeholder(tf.float32,shape=(batch_size,output_len))   #
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
            self.biases[k] = tf.Variable(np.zeros((1,1),dtype=np.float32))   #
        self.weights[(-1,)] = tf.Variable(tf.random_normal( (len(self.front),self.output_len) ))
        self.biases[(-1,)] = tf.Variable(np.zeros((1,self.output_len),dtype=np.float32))   #
        # Define activations
        for i in range(self.input_len):
            self.activations[i] = self.X[:,i]   #
        for k in self.abstractors.keys():
            parents = []   #
            for i in range(len(k)):
                parents.append( self.activations[k[i]] )
            #self.parents = tf.Variable(tf.transpose( parents ))
            self.parents = tf.transpose( parents )
            if len(self.parents.shape) == 3: # Gambiarra
                self.parents = self.parents[0]
            #print(parents)
            child = self.abstractors[k]
            self.activations[child] = tf.nn.elu(tf.matmul(self.parents,self.weights[k].initialized_value())+self.biases[k].initialized_value())
        # Fully connected at the end (front -> output)
        front_ = [] # tf.Variable(np.zeros( (len(self.front)) ))
        for f in self.front:
            front_.append(self.activations[f])
        #self.front_ = tf.Variable(tf.transpose( front_ ))
        self.front_ = tf.transpose( front_ )
        if len(self.front_.shape) == 3:  # Gambiarra
                self.front_ = self.front_[0]
        self.output = tf.nn.elu(tf.matmul(self.front_,self.weights[(-1,)].initialized_value())+self.biases[(-1,)].initialized_value())

    def var_init(self):
        var = []
        for w in self.weights:
            var.append(self.weights[w].initialized_value())
            #tf.variables_initializer([self.weights[w]])
        for b in self.biases:
            var.append(self.biases[b].initialized_value())
            #tf.variables_initializer([self.biases[b]])
        #var.append(self.parents)
        #var.append(self.front_)
        #return tf.variables_initializer(var)

g = AbGraph(3,1)
print(g.insert_ab((0,1)))
print(g.insert_ab((1,2)))
print(g.insert_ab((3,4)))
for i in g.abstractors.keys():
    print(i,g.abstractors[i])
g.build_model()

def get_batch():
    X = np.zeros((batch_size,3))
    Y = np.zeros((batch_size,1))
    for i in range(batch_size):
        x = rd.uniform(0.0,1.0)
        y = rd.uniform(0.0,1.0)
        z = rd.uniform(0.0,1.0)
        s = x + y + z
        X[i] = [x,y,z]
        Y[i] = [s]
    return X,Y

cost = tf.losses.mean_squared_error(g.Y,g.output)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # g.var_init()
    for i in range(1000):
        X,Y = get_batch()
        loss,_ = sess.run([cost,opt],feed_dict={g.X:X,g.Y:Y})
        if i%50 == 0:
            print(loss)

#print(g.delete_ab((0,1)))

