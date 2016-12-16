from codelog.tensorflow.tictactoe import py23
import tensorflow as tf
import numpy as np

ROTATE = [[0 for _ in py23.range_(9)] for _ in py23.range_(9)]

for i in py23.range_(3):
    for j in py23.range_(3):
        ROTATE[3*i+j][3*j+2-i] = 1

TRANSPOSE = [[0 for _ in py23.range_(9)] for _ in py23.range_(9)]

for i in py23.range_(3):
    for j in py23.range_(3):
        TRANSPOSE[3*i+j][3*j+i] = 1

def dot(mat_list):
    ret = mat_list[0]
    for t in mat_list[1:]:
        ret = np.dot(ret,t)
    return ret

def mcd(a,b):
    return tf.matmul(a,tf.constant(dot(b),tf.float32))

if __name__ == '__main__':
    
    print(ROTATE)
    print(TRANSPOSE)
    
    i0 = tf.constant([[1,2,3],[4,5,6],[7,8,9]],dtype=tf.float32,shape=[3,3])
    i1 = tf.reshape(i0, [1,9])

    sess = tf.Session()

    print(sess.run(i1))
    print(sess.run(tf.matmul(i1,tf.constant(dot([ROTATE]),dtype=tf.float32))))
    print(sess.run(tf.matmul(i1,tf.constant(dot([ROTATE,ROTATE]),dtype=tf.float32))))
    print(sess.run(tf.matmul(i1,tf.constant(dot([ROTATE,ROTATE,ROTATE]),dtype=tf.float32))))
    print(sess.run(tf.matmul(i1,tf.constant(dot([ROTATE,ROTATE,ROTATE,ROTATE]),dtype=tf.float32))))
    print(sess.run(tf.matmul(i1,tf.constant(TRANSPOSE,dtype=tf.float32))))
