
import tensorflow as tf


def conv2d(input_, output_dim, k_w,MAX_LEN,
		   k_h=2, d_h=1, d_w=1, stddev=0.02,
		   name="conv2d"):
	with tf.variable_scope(name):
		w = tf.get_variable('w', [k_h, k_w, 1, output_dim],   #[2,100,1,100]
							initializer=tf.truncated_normal_initializer(stddev=stddev))
		conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='VALID')

		pooled_T = tf.nn.max_pool(conv,ksize=[1,MAX_LEN - 2 +1,1,1],strides=[1,1,1,1],padding="VALID",name="pool")
		T_flat = tf.squeeze(pooled_T)
		return T_flat



def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
	shape = input_.get_shape().as_list()

	with tf.variable_scope(scope or "Linear"):
		matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
								 tf.random_normal_initializer(seed=333,stddev=stddev))
		bias = tf.get_variable("bias", [output_size],
			initializer=tf.constant_initializer(bias_start))
		if with_w:
			return tf.matmul(input_, matrix) + bias, matrix, bias
		else:
			return tf.matmul(input_, matrix) + bias
