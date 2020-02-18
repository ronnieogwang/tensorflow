import tensorflow as tf

#import the data with label onehotencoded
from tensorflow.examples.tutorials.mnist import input_data 
mnist_data = input_data.read_data_sets('MNIST_data', one_hot=True)

#building a perceptron. This is a single layer network
#variables for input layer
input_size = 784 
no_classes = 10 
batch_size = 100 
total_batches = 200 #epochs
#none means can be of any size
x_input = tf.placeholder(tf.float32, shape=[None, input_size]) 
y_input = tf.placeholder(tf.float32, shape=[None, no_classes])

#variables for fully connected layer
weights = tf.Variable(tf.random_normal([input_size, no_classes])) 
bias = tf.Variable(tf.random_normal([no_classes]))

#weighted sum produces logits
logits = tf.matmul(x_input, weights) + bias

softmax_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                        labels=y_input, logits=logits) 
loss_operation = tf.reduce_mean(softmax_cross_entropy) 
optimiser = tf.train.GradientDescentOptimizer(learning_rate=0.5).minimize(loss_operation) 

#Traiing the model with data
#initialize variables
session = tf.Session() 
session.run(tf.global_variables_initializer())

#run the batches
for batch_no in range(total_batches):    
    mnist_batch = mnist_data.train.next_batch(batch_size)   
    _, loss_value = session.run([optimiser, loss_operation], feed_dict={x_input: mnist_batch[0], y_input: mnist_batch[1]})    
    print(loss_value)

#make predictions
predictions = tf.argmax(logits, 1) 
correct_predictions = tf.equal(predictions, tf.argmax(y_input, 1)) 
accuracy_operation = tf.reduce_mean(tf.cast(correct_predictions, tf.float32)) 
test_images, test_labels = mnist_data.test.images, mnist_data.test.labels 
accuracy_value = session.run(accuracy_operation, feed_dict={x_input: test_images, y_input: test_labels }) 
print('Accuracy : ', accuracy_value) 
session.close()
    
