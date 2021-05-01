import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


hello = tf.constant('Hello Tensorflow')

sess = tf.Session()

print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)

print(node1, node2)
print(node3)

sess = tf.Session()
print(sess.run([node1, node2]))
print(sess.run(node3))


a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a+b
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict = {a:[1,3],b:[2,4]}))


x_train = [1,2,3]
y_train = [1,2,3]

w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')

hypothesis = x_train * w + b

cost = tf.reduce_mean(tf.square(hypothesis - y_train))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(w), sess.run(b))
        
############################

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

for step in range (2001):
    cost_val, w_val, b_val, _ = sess.run(([cost, w, b, train]), feed_dict = {X : [1,2,3], Y:[1,2,3]})
    if step %20 == 0:
        print( step, cost_val, w_val, b_val)


#import tensorflow as tf

w = tf.Variable(tf.random_normal([1]), name = 'weight')
w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

hopythesis = X * w + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict = {X: [1,2,3], Y:[1,2,3]})
    if step%20==0:
        print(step, cost_val, w_val, b_val)

############

w = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name = 'bias')
X = tf.placeholder(tf.float32, shape = [None])
Y = tf.placeholder(tf.float32, shape = [None])


hypothesis = X * w + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = optimizer.minimize(cost)

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for step in range(2001):
    cost_val, w_val, b_val, _ = sess.run([cost, w, b, train], feed_dict = {X:[1,2,3,4,5], Y:[2.1,3.1,4.1,5.1,6.1]})
    if step % 20 == 0:
        print(step, cost_val, w_val, b_val)




print(sess.run(hypothesis, feed_dict = {X:[5]}))
print(sess.run(hypothesis, feed_dict = {X:[2.5]}))
print(sess.run(hypothesis, feed_dict = {X:[1.5, 3.5]}))