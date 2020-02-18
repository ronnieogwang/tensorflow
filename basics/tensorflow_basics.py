#https://www.analyticsvidhya.com/blog/2016/10/an-introduction-to-implementing-neural-networks-using-tensorflow/
'''
    Build a computational graph; this can be any mathematical operation TensorFlow supports.
    Initialize variables; to compile the variables defined previously
    Create session; this is where the magic starts!
    Run graph in session; the compiled graph is passed to the session, which starts its execution.
    Close session; shutdown the session
    
when using scikit-learn library we follow three steps i.e
a. create object of class
b. fit data to object
c. predict
'''

# import tensorflow
import tensorflow as tf

# build computational graph
a = tf.placeholder(tf.int16)
b = tf.placeholder(tf.int16)

addition = tf.add(a, b)

# initialize variables
init = tf.initialize_all_variables()

# create session and run the graph
with tf.Session() as sess:
    sess.run(init)
    print( "Addition: %i" % sess.run(addition, feed_dict={a: 2, b: 3}))

# close session
sess.close()

'''tensorboard'''
#inside python shell
x = tf.placeholder(tf.float32, name='x')
y = tf.placeholder(tf.float32, name='y')
z = tf.add(x, y, name='sum')

session = tf.Session()
summary_writer = tf.summary.FileWriter('/tmp/1', session.graph)

#teriminal
tensorboard --logdir=/tmp/

#browser
http://localhost:6006/

