import numpy as np
import tensorflow as tf
import random as rd
import copy
import datetime as dt
import os.path

batch_size = 1 #
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
        g.id_count = input_len
        g.input_len = input_len
        g.output_len = output_len
        g.nodes = set() # Nodes identification
        g.front = set()  # Nodes that can be removed
        g.abstractors = {}   # Only one abstractor per tuple of nodes
        g.X = tf.placeholder(tf.float32,shape=(None,input_len))
        g.Y = tf.placeholder(tf.float32,shape=(None,output_len))
        g.weights = {}    # Weights and biases of each abstractor
        g.biases = {}
        g.params_named = {}
        g.activations = {}   # the output of the nodes
        for i in range(g.id_count):
            g.nodes.add(i)
            g.front.add(i)
        return g
    
    @classmethod
    def from_graph(self,g):
        h = AbGraph.from_size(g.input_len,g.output_len)
        h.nodes = copy.deepcopy(g.nodes)
        h.front = copy.deepcopy(g.front)
        h.abstractors = copy.deepcopy(g.abstractors)
        return h
    
    def load_parameters(self,g):
        for k in g.weights:
            tf.assign(self.weights[k],g.weights[k])
        for k in g.biases:
            tf.assign(self.biases[k],g.biases[k])
    
    def insert_ab(self,ab_in):   # criar as novas ativacoes e parametros, atualizar a ultima camada
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
    
    def ab_name(self,k):
        s = '.'
        for i in k:
            s = s + str(i) + '.'
        return s
    
    def build_model(self):
        for k in self.abstractors.keys():   # Setup parameters # Weights of abstractor k
            if not k in self.weights:
                k_name = self.ab_name(k)
                self.weights[k] = tf.Variable(tf.random_normal( (len(k),1) ),name='w_'+ k_name)
                self.biases[k] = tf.Variable(np.zeros((1,1),dtype=np.float32),name='b_'+ k_name)
                self.params_named['w_'+k_name] = self.weights[k]
                self.params_named['b_'+k_name] = self.biases[k]
        # Parameters of the final layer (fully-connecetd)
        if not (-1,) in self.weights:
            fc_name = self.ab_name((-1,))
            self.weights[(-1,)] = tf.Variable(tf.random_normal( (len(self.front),self.output_len) ),name='w_'+fc_name)
            self.biases[(-1,)] = tf.Variable(np.zeros((1,self.output_len),dtype=np.float32),name='b_'+fc_name)
            self.params_named['w_'+fc_name] = self.weights[(-1,)]
            self.params_named['b_'+fc_name] = self.biases[(-1,)]
        for i in range(self.input_len):     # Define activations
            if not i in self.activations:
                l = []
                for j in range(batch_size):
                    x_ = self.X[j,i]
                    l.append([x_])
                self.activations[i] = l
        for k in sorted(self.abstractors.keys()):   # Must be sorted to guarantee dependency order
            child = self.abstractors[k]
            if not child in self.activations:
                parents = []
                for i in range(len(k)):
                    parents.append(self.activations[k[i]][0])  # [0]
                parents = tf.transpose(parents)
                self.activations[child] = tf.nn.relu(tf.matmul(parents,self.weights[k])+self.biases[k])
        # Fully connected at the end (front -> output)
        front_ = []
        for f in sorted(self.front):
            front_.append(self.activations[f][0])  # [0]
        front_ = tf.transpose(front_)
        #self.output = tf.nn.relu(tf.matmul(front_,self.weights[(-1,)])+self.biases[(-1,)])
        self.output = tf.matmul(front_,self.weights[(-1,)])+self.biases[(-1,)]
    
    def init_var(self,sess):
        for k in self.weights:
            sess.run(self.weights[k].initializer)
        for k in self.biases:
            sess.run(self.biases[k].initializer)

def get_batch():
    X = np.zeros((batch_size,5))
    Y = np.zeros((batch_size,1))
    for i in range(batch_size):
        s = 0.0
        for j in range(5):
            X[i,j] = rd.uniform(-1.0,1.0)
            s += X[i,j]
        #X[i,0] = rd.uniform(0.0,1.0)
        #s = X[i,0]*(1-X[i,1])+(1-X[i,0])*X[i,1]
        #s = X[i,0]+X[i,1]
        #s = X[i,0]
        #s = rd.uniform(0.0,1.0)
        Y[i] = [s]
    return X,Y

def get_cost_opt(g_):
    cost = tf.reduce_mean(tf.losses.mean_squared_error(g_.Y,g_.output)) #losses.mean_squared_error(g_.Y,g_.output)
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return cost, opt

with tf.Session() as sess:  # sess is not AbGraph
    g = AbGraph.from_size(5,1)
    g.insert_ab((0,1))
    g.insert_ab((1,2))
    g.insert_ab((2,3))
    g.insert_ab((3,4))
    g.build_model()
    cost_g, opt_g = get_cost_opt(g)
    saver = tf.train.Saver() # g.params_named;  #saver is associated with g.params_named variables
    if os.path.isfile(model_g_fn+'.meta'):
        saver.restore(sess,model_g_fn)
    else:
        sess.run(tf.global_variables_initializer())
    for i in range(36000):
        X,Y = get_batch()
        loss,_ = sess.run([cost_g,opt_g],feed_dict={g.X:X,g.Y:Y})
        if i%50 == 0:
            print(loss)
            if i%5000 == 0:
                saver.save(sess,model_g_fn)
    print()

tf.reset_default_graph()
with tf.Session() as sess:
    #
    h = AbGraph.from_graph(g)   # same structure as g, loads the operations declared in g
    # modify h's topology
    h.delete_ab((0,1))
    h.build_model()
    #del h.params_named['w_'+h.ab_name((1,2))]
    #del h.params_named['b_'+h.ab_name((1,2))]
    del h.params_named['w_.-1.']
    del h.params_named['b_.-1.']
    cost_h, opt_h = get_cost_opt(h)
    sess.run(tf.global_variables_initializer())
    #saver_h = tf.train.Saver(h.params_named)  # g.params_named
    #saver_h.restore(sess,model_g_fn)    # restore from g
    #
    for i in range(11000):
        X,Y = get_batch()
        loss,_ = sess.run([cost_h,opt_h],feed_dict={h.X:X,h.Y:Y})
        if i%50 == 0:
            print(loss)
            #if i%5000 == 0:
            #    saver_h.save(sess,model_h_fn)
    print()

