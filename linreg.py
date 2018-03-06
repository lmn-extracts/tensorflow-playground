import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import utils
import tensorflow as tf
import matplotlib.pyplot as plt

data, nsamples = utils.read_birth_life_data('birth_life_2010.txt')

X = tf.placeholder(tf.float32, name='x')
Y = tf.placeholder(tf.float32, name='y')

weight = tf.get_variable('weight', [], initializer = tf.constant_initializer(0.0))
bias = tf.get_variable('bias', [], initializer = tf.constant_initializer(0.0))

pred = weight * X + bias

loss = tf.square(Y - pred)

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(loss)

writer = tf.summary.FileWriter('.graphs/lin_reg', tf.get_default_graph())

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(data.shape)
	for i in range(100):
		total_loss = 0.0
		for x,y in data:
			_, l = sess.run([optimizer,loss], feed_dict = {X:x,Y:y})
			total_loss += l
		print('Epoch {}: {}'.format(i,total_loss))

		w_out, b_out = sess.run([weight,bias])
		writer.close()

plt.plot(data[:,0], data[:,1],'bo', label='Actual Data')
plt.plot(data[:,0], data[:,0] * w_out + b_out, label = 'Fit')
plt.title('Linear Regression')
plt.xlabel('Birth Rate')
plt.ylabel('Life Expectancy')
plt.legend()
plt.show()