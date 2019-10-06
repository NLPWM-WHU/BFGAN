import tensorflow as tf
from   hotel.ga import *
import numpy as np

class GAN:

    def __init__(self, options):
        self.options = options

    def build_model(self):
        bf_size = self.options['bf_dim']
        t_real_bf = tf.placeholder('float32', [self.options['batch_size'],bf_size], name = 'real_bf')
        t_real_rating = tf.placeholder('float32', [self.options['batch_size'], self.options['emb_dim']],
                                     name='real_rating_input')
        t_wrong_rating = tf.placeholder('float32', [self.options['batch_size'], self.options['emb_dim']],
                                     name='wrong_rating_input')


        t_z = tf.placeholder('float32', [self.options['batch_size'], self.options['z_dim']])



        with tf.variable_scope(tf.get_variable_scope()):

            #随机采样文本后投入生成器 生成了行为特征fake_bf和对应匹配的文本t_text_conv UT
            fake_bf,t_text_conv,z_3 = self.generator(1,t_z, t_real_rating)

            tf.summary.histogram("fake_bf", fake_bf)
            tf.summary.histogram("real_bf", t_real_bf)


            t_text_conv_sigmoid=tf.nn.sigmoid(t_text_conv)

            disc_real_bf, disc_real_bf_logits   = self.discriminator(6,t_real_bf, t_real_rating)
            disc_wrong_bf, disc_wrong_bf_logits   = self.discriminator(6,t_real_bf, t_wrong_rating,reuse=True)
            disc_fake_bf, disc_fake_bf_logits   = self.discriminator(6,fake_bf, t_text_conv,reuse=True )


        #J(D(FB+MT),1)
        g_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_bf_logits, labels=tf.ones_like(disc_fake_bf)))

        #J(FB,TB)
        g_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_bf, labels=t_real_bf))


        g_loss = g_loss1+g_loss2

        d_loss1 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_real_bf_logits, labels=tf.ones_like(disc_real_bf)))
        d_loss2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_wrong_bf_logits, labels=tf.zeros_like(disc_wrong_bf)))
        d_loss3 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=disc_fake_bf_logits, labels=tf.zeros_like(disc_fake_bf)))
        d_loss = d_loss1 + (d_loss2 + d_loss3)/2


        t_vars = tf.trainable_variables()
        d_vars = [var for var in t_vars if 'd_' in var.name]
        g_vars = [var for var in t_vars if 'g_' in var.name]

        tf.summary.scalar("d_loss", d_loss)
        tf.summary.scalar("g_loss", g_loss)

        input_tensors = {
            't_real_bf' : t_real_bf,
            't_z' : t_z,
            't_wrong_rating':t_wrong_rating,
            't_real_rating':t_real_rating,

        }

        variables = {
            'd_vars' : d_vars,
            'g_vars' : g_vars
        }

        loss = {
            'g_loss' : g_loss,
            'd_loss' : d_loss,

        }

        outputs = {
            'generator' : fake_bf,
            'generator2' : t_text_conv,
            'generator3': tf.concat([1*fake_bf,t_text_conv],1),
            'generator4': tf.concat([1*fake_bf,t_text_conv_sigmoid],1),
            'z_3':z_3
        }

        checks = {
            'g_loss1':g_loss1,
            'g_loss2':g_loss2,
            'd_loss1': d_loss1,
            'd_loss2': d_loss2,
            'd_loss3' : d_loss3,
            'disc_real_bf_logits' : disc_real_bf_logits,
            'disc_wrong_bf_logits' : disc_wrong_bf_logits,
            'disc_fake_bf_logits' : disc_fake_bf_logits,

        }

        return input_tensors, variables, loss, outputs, checks

    # GENERATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def generator(self,l,t_z,t_text):
        #拼接上ratio额外信息 为101维度

        z_concat = tf.concat([t_text],1)
      #  z_concat=tf.nn.sigmoid(t_text)
        z_1 = z_concat
        #三层非线性全连接网络 最后为维度为6的生成行为特征
        z_2 = tf.nn.tanh(linear(z_1, self.options['gf2_dim'], 'g_h2_lin'))
        z_3 = tf.nn.tanh(linear(z_2, self.options['gf2_dim']/2, 'g_h3_lin'))
        z_5 = tf.nn.sigmoid(linear(z_3, self.options['gf3_dim']/l, 'g_h5_lin'))# behavior featur

        return z_5,t_text,z_3


    # DISCRIMINATOR IMPLEMENTATION based on : https://github.com/carpedm20/DCGAN-tensorflow/blob/master/model.py
    def discriminator(self,length,bf, t_text_embedding,reuse=False):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        t_text = tf.nn.sigmoid(linear(t_text_embedding, length, scope='d_h0_lin'))
        h_concat = tf.concat([bf, t_text],1,name='h_concat')#拼接层
        h1 = tf.nn.tanh(linear(h_concat, self.options['df1_dim'], scope='d_h1_lin')) #二层
        h2 = tf.nn.tanh(linear(h1, self.options['df1_dim']/2, scope='d_h2_lin'))  #三层
        #h3 = tf.nn.tanh(ops.linear(h2, self.options['df1_dim'] / 4, scope='d_h3_lin'))
        h4 = linear(h2, self.options['df2_dim'], scope='d_h4_lin') #分类层
        return tf.nn.sigmoid(h4), h4



