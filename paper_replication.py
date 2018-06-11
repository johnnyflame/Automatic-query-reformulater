import atire
import numpy as np
import tensorflow as tf
import pickle as pkl
import tensorflow.contrib.slim as slim
import re
from itertools import islice


import gensim

"""
"First system"

A prototype of the model described in the paper "Task-Oriented Query Reformulation with Reinforcement Learning", 
Nogueira and Cho 2017


Author: Johnny Flame Lee 

"""
DUMMYMODE = False




CREATE_DICTIONARY = True
#TOPIC_FILE = "/Users/johnnyflame/atire/evaluation/topics.51-100.txt"

TOPIC_FILE = "/Users/johnnyflame/atire/evaluation/topics.51-61"
ASSESSMENT_FILE = "/Users/johnnyflame/atire/evaluation/WSJ.qrels"

CONTEXT_WINDOW = 4

# Total number of terms to go into the second network
CANDIDATE_AND_CONTEXT_LENGTH = CONTEXT_WINDOW * 2 + 1


WORD_VECTOR_DIMENSIONS = 300

# maximum number of terms in q0
MAX_SEQUENCE_LENGTH = 15

WORD_EMBEDDING_PATH = "wsj-collection-vectors"

METRIC = " -mMAP"



PADDING = np.ones(WORD_VECTOR_DIMENSIONS)


# HYPERPARAMETERS:
atire.init("atire -a " + ASSESSMENT_FILE + METRIC)



def load_lookup_table(file):
    """Return a dictionary of term-vector pairs"""
    return pkl.load(open(file, "rb"))

def read_topic_file(topic_file_path,topic_list):
    """Read a TREC topic file and parse it into a dictionary"""
    f = open(topic_file_path,'r')
    for line in f:
        topic_id = line.split()[0]
        original_query = " ".join(line.split()[1::])
        topic_list[topic_id] = original_query




def retrieve_document_terms(query):
    """
    :param query: A query to pass to the search engine
    :return: a dictionary of term-Word2Vec embedding pairs, in the order the terms appear in the collection.
    """
    tokens = []
    results = atire.lookup(-1, query)

    for result in results:
        tokens.append(atire.get_ordered_tokens(result))

    return tokens


class GenerateNetwork:

    def __init__(self,number_of_terms):

        self.session = tf.Session()
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        self.query_input = tf.placeholder(tf.float32,[None,number_of_terms,WORD_VECTOR_DIMENSIONS],name="query_input")
        self.candidate_and_context_input = tf.placeholder(tf.float32,[None,CANDIDATE_AND_CONTEXT_LENGTH,WORD_VECTOR_DIMENSIONS]
                                                          ,name="candidate_vectors")


        # Reshaping the query so it becomes Rank 4, the order is [batch_size, width,height, channel]
        self.reshaped_query_input = tf.reshape(self.query_input,[self.batch_size,number_of_terms,WORD_VECTOR_DIMENSIONS,1])

        self.reshaped_candidate_and_context = tf.reshape(self.candidate_and_context_input,
                                                         [self.batch_size,CANDIDATE_AND_CONTEXT_LENGTH,WORD_VECTOR_DIMENSIONS,1])

        # Add 2 convolutional layers with ReLu activation
        self.query_conv1 = slim.conv2d(
            self.reshaped_query_input, num_outputs=256,
            kernel_size=[3,WORD_VECTOR_DIMENSIONS], stride=[1,1], padding='VALID'
        )


        # Second convolution layer
        self.query_conv2 = slim.conv2d(
            self.query_conv1, num_outputs=256,
            kernel_size=[3,1], stride=[1,1], padding='VALID'
        )

        # Not super confident about these parameters, may need revisit
        self.query_pooled = tf.nn.max_pool(
            self.query_conv2,
            ksize=[1,11,1,1],
            strides=[1, 1, 1,1],
            padding='VALID',
            name="pool")


        self.candidates_conv1 =  slim.conv2d(
            self.reshaped_candidate_and_context, num_outputs=256,
            kernel_size=[5,WORD_VECTOR_DIMENSIONS], stride=[1,1], padding='VALID'
        )

        self.candidates_conv2 =  slim.conv2d(
            self.candidates_conv1, num_outputs=256,
            kernel_size=[3,1], stride=[1,1], padding='VALID'
        )

        self.candidates_pooled = tf.nn.max_pool(
            self.candidates_conv2,
            ksize=[1, 3, 1, 1],
            strides=[1, 1, 1, 1],
            padding='VALID',
            name="pool")




        self.pooled_vectors_concatenated = tf.concat((self.query_pooled, self.candidates_pooled), axis=3)






        self.policy_fc1 = tf.contrib.layers.fully_connected(tf.reshape(self.pooled_vectors_concatenated,[self.batch_size,512]),
                                                            num_outputs=256,activation_fn=tf.nn.tanh,
                                          weights_initializer=tf.contrib.layers.xavier_initializer(),
                                       biases_initializer=tf.constant_initializer(0.01))



     #   self.aprob = slim.fully_connected(self.policy_fc1,1,biases_initializer=None,activation_fn=tf.nn.sigmoid)





        # outputsthe probabilty of selecting term
        self.action_probability = tf.contrib.layers.fully_connected(self.policy_fc1,num_outputs=1,activation_fn=tf.nn.sigmoid,
                                                                    weights_initializer=tf.contrib.layers.xavier_initializer(),
                                                                    biases_initializer=None)







        self.mean_context_vector = tf.reduce_mean(self.candidates_pooled,axis=0)
        self.mean_context_and_query_concatenated = tf.concat((tf.reduce_mean(self.query_pooled,axis=0),self.mean_context_vector),axis=2)

        #
        # self.value_fc1 = tf.contrib.layers.fully_connected(tf.reshape(self.mean_context_and_query_concatenated,[1,512]),
        #                                                     num_outputs=256,activation_fn=tf.nn.tanh,
        #                                   weights_initializer=tf.contrib.layers.xavier_initializer(),
        #                                biases_initializer=tf.constant_initializer(0.01))
        #
        # self.value_prediction = tf.contrib.layers.fully_connected(self.value_fc1,num_outputs=1,activation_fn=tf.nn.sigmoid,
        #                                                             weights_initializer=tf.contrib.layers.xavier_initializer(),
        #                                                             biases_initializer=None)




        # Using the same training parameters from the paper
        self.optimizer = tf.train.AdamOptimizer(learning_rate=10e-4,beta1=0.9,beta2=0.999,epsilon=10e-8)
        # Update the parameters according to the computed gradient.



        self.action_holder = tf.placeholder(shape=None,dtype=tf.int32)



        self.reward = tf.placeholder(shape=[1],dtype=tf.float32,name="target")

        # self.reward_prediction = tf.placeholder(shape=[1],dtype=tf.float32)



        #
        # #TODO: ask about this line
        # Nicked this line, unsure why it's there
        # self.responsible_weight = tf.slice(self.output,self.action_holder,[0])
        self.policy_loss = self.reward * tf.reduce_sum(-tf.log(self.action_probability))



        # self.policy_loss = tf.nn.l2_loss(self.reward - tf.reduce_sum(-tf.log(self.responsible_weight), axis=0))

        # self.value_loss = 0.1 * tf.nn.l2_loss(self.reward - self.value_prediction)
        # self.policy_loss = -tf.log(self.picked_action_prob) * self.target
        # self.value_loss = 0.1* tf.squared_difference(self.reward,self.value_prediction)
        #
        # self.train_value_network = self.optimizer.minimize(self.value_loss)
        self.train_policy_network = self.optimizer.minimize(self.policy_loss)

        self.tvar = tf.trainable_variables()




        init = tf.global_variables_initializer()
        self.session.run(init)

    def get_query_pooled(self,query,batchsize):
        feed_dict = {self.query_input:query,self.batch_size:batchsize}
        return self.session.run(self.query_pooled,feed_dict=feed_dict)
    def candidate_pooled(self,candidate,batch_size):
        feed_dict = {self.candidate_and_context_input:candidate, self.batch_size:batch_size}
        return self.session.run(self.candidates_pooled,feed_dict=feed_dict)



    def get_action_prob(self, query, candidates,batch_size=1):
        feed_dict = {self.query_input: query, self.candidate_and_context_input: candidates,self.batch_size:batch_size}
        return self.session.run([self.action_probability], feed_dict=feed_dict)

    # def get_value_prediction(self,query_input,candidate_input,batch_size=1):
    #     feed_dict = {self.query_input:query_input, self.candidate_and_context_input:candidate_input,self.batch_size:batch_size}
    #     return self.session.run(self.value_prediction,feed_dict=feed_dict)



    def policy_update(self,state,query_feed_in,reward,batch_size):
        feed_dict = {self.candidate_and_context_input:state,
                     self.query_input:query_feed_in,
                     self.reward:reward
                    ,self.batch_size:batch_size}
        loss, tvar,_ = self.session.run([self.policy_loss,self.tvar, self.train_policy_network], feed_dict=feed_dict)
        return loss,tvar

    # def value_update(self,state,query_feed_in,reward,batch_size):
    #     feed_dict = {self.reward: reward,
    #                  self.query_input: query_feed_in,
    #                  self.candidate_and_context_input:state,
    #                  self.batch_size:batch_size}
    #     loss,_ = self.session.run([self.value_loss,self.train_value_network],feed_dict=feed_dict)
    #     return loss


def lookup_term_vectors(terms):
    """
    Looks up the wordembedding for a set of terms,
    and returns a numpy array version of the vectors to be used in the model.

    :param query: the query to search for in the lookup table
    :return: a numpy ndarray, each entry corresponding to a term in the set.
    """

    query = []
    word_vector = []




    # query_terms.append([x for x in term.split(" ")])

    query.append(re.sub(r'\W+', " ", terms).lower())


    for terms in query:
        for word in terms.split(" "):
            # TODO: Retrain Word2Vec and remove this line
            if word not in word_embedding.wv.vocab:
                print (word)
                word_vector.append(np.zeros(shape=WORD_VECTOR_DIMENSIONS))
            else:
                word_vector.append(word_embedding.wv[word])


    if len(word_vector) < MAX_SEQUENCE_LENGTH:
        diff = MAX_SEQUENCE_LENGTH - len(word_vector)

        for i in range(0,diff):
            word_vector.append(PADDING)

    return query,np.array(word_vector)



def window(seq, n=3):
    "Returns a sliding window (of width n) over data from the iterable"
    "   s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...                   "
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result


if __name__ == "__main__":

    word_embedding = gensim.models.Word2Vec.load(WORD_EMBEDDING_PATH)


    if not DUMMYMODE:
        # Query lookup table, a dictionary with the format "topic-ID:original query" as key:value
        querys_table = {}
        read_topic_file(TOPIC_FILE,querys_table)

        network = GenerateNetwork(number_of_terms=MAX_SEQUENCE_LENGTH)



        for i in range(0,50):
            for topicID in querys_table:

                current_query = querys_table[topicID]
                reformulated_query = []
                candidate_pooled_vector_accumulated = []

                query_feed = []
                states = []
                candidate_terms = []

                aprob_record = []

                actions_taken = []


                # This is the input to the left half of the neural network
                current_query, query_vectors = lookup_term_vectors(current_query)

                #queries_vectors = lookup_term_vectors("gobbody-gook dah baqfuafk vnvi")


                #TODO: Need to concatenate the original query as well here. Although it could be considered trivial, because it's reasonable to assume that the original query terms are going to be in the first set of documents anyway.
                # Get a list of all the words in the top 10 documents
                terms_in_results = retrieve_document_terms(" ".join(current_query))

                terms_in_results = terms_in_results[:1]


                # Each document in the top 10 results list
                for doc in terms_in_results:
                    # For each term in one of the documents
                    for i in range(0,len(doc)):
                        candidate_and_context = []
                        candidate_terms.append(doc[i])

                        # This represents the state
                        candidate_and_context_vectors = []

                        # pad on the left
                        if i < CONTEXT_WINDOW:
                            diff = CONTEXT_WINDOW - i
                            candidate_and_context = doc[0:i + CONTEXT_WINDOW + 1]
                            for term in candidate_and_context:
                                candidate_and_context_vectors.append(word_embedding.wv[term])


                            for j in range(0,diff):
                                candidate_and_context.insert(0,"$PADDING$")
                                candidate_and_context_vectors.insert(0,PADDING)
                        # pad on the right---
                        elif (len(doc) - (i+1)) < CONTEXT_WINDOW:
                            # TODO: A known issue here: '' seperation at the end of each document is counted as a valid term, this may require fixing if it causes a problem in query reformulation.

                            diff = CONTEXT_WINDOW - (len(doc) - (i+1))
                            candidate_and_context = doc[i-CONTEXT_WINDOW:len(doc)]
                            for term in candidate_and_context:
                                candidate_and_context_vectors.append(word_embedding.wv[term])

                            for j in range(0, diff):
                                candidate_and_context.append("$PADDING$")
                                candidate_and_context_vectors.append(PADDING)
                        # No padding, sliding window in normal range
                        else:
                            candidate_and_context = doc[i-CONTEXT_WINDOW:i + CONTEXT_WINDOW + 1]
                            for term in candidate_and_context:
                                candidate_and_context_vectors.append(word_embedding.wv[term])


                        # Per-step action begins here


                        # Here we prepare the inputs to the neural network
                        states.append(candidate_and_context_vectors)
                        query_feed.append(query_vectors)






                query_feed = np.array(query_feed)
                states = np.array(states)
                batch_size = len(states)


                a_prob = network.get_action_prob(query_feed,states, batch_size)
                # predicted_reward = network.get_value_prediction(query_feed, states, batch_size)





                for i in range(0, batch_size):

                    probability = a_prob[0][i][0]

                    if probability > 0.5:
                        reformulated_query.append(candidate_terms[i])
                        actions_taken.append(1)
                    else:
                        actions_taken.append(0)


                reward = atire.lookup(int(topicID), " ".join(reformulated_query))


                # Feed the mean context vector in to the network to predict value




                """
                At this point we are ready to evaluate the reformulated query and retrieve a reward from ATIRE.
                
                Things we need here:
                
                1. A list of words representing the reformulated query
                2. Decide on a reward metric: precision@k or recall@k
                3. The original queryID
                4. Path to the evaluation file 
                
                """


                print("reward: " + str(reward[0]))
                policy_loss,tvar = network.policy_update(states,query_feed,reward,batch_size)
                # value_loss = network.value_update(states,query_feed,reward,batch_size)




                # mean_policy_loss/=len(states)

                print("policy loss: ", policy_loss)
                # print("value loss: ", value_loss)




            """
            Save the weights here
            
            Can I visialize the graph to ensure my implementation is correct? 
            """







    #
    # # dummy inputs to the network
    # else:
    #
    #     dummy_query = "hello world"
    #     dummy_candidate_terms = "hello world this is a dummy result something something"
    #     dummy_query_input_vectors = []
    #     dummy_candidate_terms_vector= []
    #
    #
    #     candidate_pooled_vector_accumulated = []
    #
    #
    #
    #
    #     for word in dummy_query.split(" "):
    #         dummy_query_input_vectors.append(word_embedding.wv[word])
    #
    #     for word in dummy_candidate_terms.split(" "):
    #         dummy_candidate_terms_vector.append(word_embedding.wv[word])
    #
    #
    #
    #     dummy_query_input_vectors = np.array(dummy_query_input_vectors)
    #     dummy_candidate_terms_vector = np.array(dummy_candidate_terms_vector)
    #
    #     # padding
    #     if len(dummy_query_input_vectors) < MAX_SEQUENCE_LENGTH:
    #         diff = MAX_SEQUENCE_LENGTH - len(dummy_query_input_vectors)
    #         dummy_query_input_vectors = np.pad(dummy_query_input_vectors,[(0,diff),(0,0)],mode='constant',constant_values=0)
    #
    #
    #
    #
    #
    #
    #     network = GenerateNetwork(number_of_terms=MAX_SEQUENCE_LENGTH)
    #
    #     max_pooled_candidate_vector = network.get_pooled_candidate_vector(dummy_candidate_terms_vector)
    #
    #     candidate_pooled_vector_accumulated.append(np.squeeze(max_pooled_candidate_vector))
    #
    #
    #
    #
    #
    #
    #     # dummy_input_vectors = np.array(
    #     #     [
    #     #         [1,0,0,0,1,1],
    #     #         [2,0,0,0,1,1],
    #     #         [3,0,0,0,1,1],
    #     #         [1, 0, 0, 0, 1,1],
    #     #         [2, 0, 0, 0, 1,1],
    #     #         [3, 0, 0, 0, 1,1]
    #     #
    #     #     ]
    #     # )
    #
    #
    #
    #     pooled_vector = network.get_pooled_query_vector(dummy_query_input_vectors)
    #     #
    #
    #     print (pooled_vector)
    #
    #
    #
    #
    #
    #
    #
    #
    #







