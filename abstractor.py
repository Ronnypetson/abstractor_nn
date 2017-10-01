import numpy as np
import tensorflow as tf
import random as rd
import copy
import datetime as dt
import os.path

batch_size = 48 #
learning_rate = 0.001
checkpoint_dir = '/checkpoint/abstractor/'
model_g_fn = checkpoint_dir + 'model_g.ckpt'
model_h_fn = checkpoint_dir + 'model_h.ckpt'
class AbGraph:
    def __init__(self):
        self.creation_time = dt.datetime.now()
    
    @classmethod
    def from_size(self,input_len,output_len):
        g = AbGraph()
        g.id_count = input_len   #+output_len
        g.input_len = input_len
        g.output_len = output_len
        g.nodes = set() # Nodes identification
        g.front = set()  # Nodes that can be removed
        g.abstractors = {}   # Only one abstractor per tuple of nodes
        g.X = tf.placeholder(tf.float32,shape=(None,input_len))    # None
        g.Y = tf.placeholder(tf.float32,shape=(None,output_len))   #
        g.weights = {}    # Weights and biases of each abstractor
        g.biases = {}
        g.params_named = {}
        g.activations = {}   # the output of the nodes
        for i in range(g.id_count):
            g.nodes.add(i)   # 0 to input_len-1, input_len to input_len+output_len-1
            g.front.add(i)
        return g
    
    @classmethod
    def from_graph(self,g):
        h = AbGraph.from_size(g.input_len,g.output_len)
        h.nodes = copy.deepcopy(g.nodes)
        h.front = copy.deepcopy(g.front)
        h.abstractors = copy.deepcopy(g.abstractors)
        h.build_model()
        #h.load_parameters(g)
        return h
    
    def load_parameters(self,g):
        for k in g.weights:
            tf.assign(self.weights[k],g.weights[k])
        for k in g.biases:
            tf.assign(self.biases[k],g.biases[k])
    
    def insert_ab(self,ab_in):   # ab_in must be tuple of shape (2); criar as novas ativações e parametros, atualizar a última camada
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
        for k in self.abstractors.keys():   # Setup parameters # Weights of abstractor k
            self.weights[k] = tf.Variable(tf.random_normal( (len(k),1) )) # mean = 0.0, sd = 1.0
            self.biases[k] = tf.Variable(np.zeros((1,1),dtype=np.float32))   #
            self.params_named['w_'+str(k)] = self.weights[k]
            self.params_named['b_'+str(k)] = self.biases[k]
        # Parameters of the final layer (fully-connecetd)
        self.weights[(-1,)] = tf.Variable(tf.random_normal( (len(self.front),self.output_len) ))
        self.biases[(-1,)] = tf.Variable(np.zeros((1,self.output_len),dtype=np.float32))   #
        self.params_named['w_'+str((-1,))] = self.weights[(-1,)]
        self.params_named['b_'+str((-1,))] = self.biases[(-1,)]
        for i in range(self.input_len):     # Define activations
            self.activations[i] = self.X[:,i]   #
        for k in sorted(self.abstractors.keys()):   # Must be sorted to guarantee dependency order
            parents = []   #
            for i in range(len(k)):
                parents.append(self.activations[k[i]])
            parents = tf.transpose(parents)
            if len(parents.shape) == 3: #
                parents = parents[0]
            child = self.abstractors[k]
            self.activations[child] = tf.nn.elu(tf.matmul(parents,self.weights[k])+self.biases[k])  # .initialized_value()
        # Fully connected at the end (front -> output)
        front_ = [] # tf.Variable(np.zeros( (len(self.front)) ))
        for f in sorted(self.front):
            if len(self.activations[f].shape) == 1: #
                front_.append(self.activations[f])
            else:
                front_.append(tf.transpose(self.activations[f])[0])
        front_ = tf.transpose(front_)
        if len(front_.shape) == 3:
            front_ = front_[0]
        elif len(front_.shape) == 1:
            front_ = [front_]
        self.output = tf.nn.elu(tf.matmul(front_,self.weights[(-1,)])+self.biases[(-1,)])   # .initialized_value()
    
    def init_var(self,sess):
        for k in self.weights:
            sess.run(self.weights[k].initializer)
        for k in self.biases:
            sess.run(self.biases[k].initializer)

g = AbGraph.from_size(20,1)
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
            X[i,j] = rd.uniform(1.0,2.0)
            s += j*X[i,j]**2
        Y[i] = [s]
    return X,Y

def get_cost_opt(g_):
    cost = tf.losses.mean_squared_error(g_.Y,g_.output)
    opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
    return cost, opt

cost_g, opt_g = get_cost_opt(g)
saver = tf.train.Saver(g.params_named)  # saver is associated with g.params_named variables

with tf.Session() as sess:
    if os.path.isfile(model_g_fn+'.meta'):
        saver.restore(sess,model_g_fn)
    else:
        sess.run(tf.global_variables_initializer())
    for i in range(15000):
        X,Y = get_batch()
        loss,_ = sess.run([cost_g,opt_g],feed_dict={g.X:X,g.Y:Y})
        if i%50 == 0:
            print(loss)
            if i%1000 == 0:
                saver.save(sess,model_g_fn)
    print()

with tf.Session() as sess:
    h = AbGraph.from_graph(g)
    cost_h, opt_h = get_cost_opt(h)
    h.init_var(sess)    # modifica, atualiza, inicializa, restaura
    #del h.params_named['w_(-1,)']  # restore a subset of the parameters from g
    del h.params_named['b_(-1,)']
    saver_h = tf.train.Saver(h.params_named)
    saver_h.restore(sess,model_g_fn)
    for i in range(1500):
        X,Y = get_batch()
        loss,_ = sess.run([cost_h,opt_h],feed_dict={h.X:X,h.Y:Y})
        if i%50 == 0:
            print(loss)
            if i%1000 == 0:
                saver_h.save(sess,model_h_fn)
    print()

#with tf.Session() as sess:
#    saver.restore(sess,model_g_fn)
#    for i in range(1500):
#        X,Y = get_batch()
#        loss,_ = sess.run([cost_g,opt_g],feed_dict={g.X:X,g.Y:Y})
#        if i%50 == 0:
#            print(loss)
#    print()

