import numpy as np
import tensorflow as tf
import random as rd
import copy

batch_size = 48 #
learning_rate = 0.001
checkpoint_dir = '/checkpoint/abstractor/'
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

    def mutate(self):   # p(add) = 1 - p(remove)
        # sample from p, add or remove abstractor and update the graph
        #dc = copy.deepcopy(self)
        return None

    def build_model(self):
        for k in self.abstractors.keys():   # Setup parameters # Weights of abstractor k
            self.weights[k] = tf.Variable(tf.random_normal( (len(k),1) )) # mean = 0.0, sd = 1.0
            self.biases[k] = tf.Variable(np.zeros((1,1),dtype=np.float32))   #
        self.weights[(-1,)] = tf.Variable(tf.random_normal( (len(self.front),self.output_len) ))
        self.biases[(-1,)] = tf.Variable(np.zeros((1,self.output_len),dtype=np.float32))   #
        for i in range(self.input_len):     # Define activations
            self.activations[i] = self.X[:,i]   #
        for k in sorted(self.abstractors.keys()):
            parents = []   #
            for i in range(len(k)):
                parents.append(self.activations[k[i]])
            parents = tf.transpose(parents)
            if len(parents.shape) == 3:
                parents = parents[0]
            child = self.abstractors[k]
            self.activations[child] = tf.nn.elu(tf.matmul(parents,self.weights[k].initialized_value())+self.biases[k].initialized_value())
        # Fully connected at the end (front -> output)
        front_ = [] # tf.Variable(np.zeros( (len(self.front)) ))
        for f in self.front:
            if len(self.activations[f].shape) == 1:
                front_.append(self.activations[f])
            else:
                front_.append(tf.transpose(self.activations[f])[0])
        front_ = tf.transpose(front_)
        if len(front_.shape) == 3:
            front_ = front_[0]
        elif len(front_.shape) == 1:
            front_ = [front_]
        self.output = tf.nn.elu(tf.matmul(front_,self.weights[(-1,)].initialized_value())+self.biases[(-1,)].initialized_value())

g = AbGraph(20,1)
#g.insert_ab((0,1))
#for i in range(19): # 0 - 19
#    g.insert_ab((i,i+1))
#for i in range(20,38):
#    g.insert_ab((i,i+1))
#for i in range(39,56):
#    g.insert_ab((i,i+1))
g.build_model()

def get_batch():
    X = np.zeros((batch_size,20))
    Y = np.zeros((batch_size,1))
    for i in range(batch_size):
        s = 0.0
        for j in range(20):
            X[i,j] = rd.uniform(0.0,1.0)
            s += j**X[i,j] + j*(X[i,j]**j)
        Y[i] = [s]
    return X,Y

cost = tf.losses.mean_squared_error(g.Y,g.output)
opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

sv = tf.train.Supervisor(logdir=checkpoint_dir,save_model_secs=60)
with sv.managed_session() as sess:
    if not sv.should_stop():
        #sess.run(tf.global_variables_initializer())
        for i in range(50000):
            X,Y = get_batch()
            loss,_ = sess.run([cost,opt],feed_dict={g.X:X,g.Y:Y})
            if i%50 == 0:
                print(loss)

