import tensorflow as tf
import codelog.tensorflow.tictactoe.Logic as tttl
import codelog.tensorflow.tictactoe.Console as tttc

WIN_SCORE = 1
LOSE_SCORE = -1
DRAW_SCORE = -0.1
ACT_SCORE = -0.01
WRONG_ACT_SCORE = -1
BATCH_SIZE = 1000
BETA = 15.0/16.0

def getQdn():
    dataIn = tf.placeholder(tf.float32, [None,3,3])
    
    mid = dataIn
    mid = tf.reshape(mid, [9])
    mid = tf.matmul(mid,tf.zeros([9,10]))
    mid = mid + tf.zeros[10]
    mid = tf.nn.relu(mid)
    mid = tf.matmul(mid,tf.zeros([10,9]))
    mid = mid + tf.zeros[9]
    
    dataOut = mid
    
    return dataIn,dataOut

def status2In(status):
    return [
        [
            1 if c == status.actor else
            0 if c == None else
            -1
            for c in cl
        ] for cl in status.cell
    ]

if __name__ == '__main__':

    qdnIn, qdnOut = getQdn()
    nextBestScore
    actionOut = tf.argmax(qdnOut,1)
    scoreMax = tf.reduce_max(qdnOut,1)

    trainIn = qdnIn
    trainOut = tf.placeholder(tf.float32, [None,9])

    mask = tf.placeholder(tf.float32, [None,9])
    mid = mask
    mid = qdnOut * mid
    mid = trainOut - mid
    mid = tf.abs(mid)
    mid = tf.reduce_sum(mid, reduction_indices=[1])
    mid = tf.reduce_mean(mid)
    train_step = tf.train.AdamOptimizer(1e-4).minimize(mid)
    
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    stateInList = []
    maskList = []
    scoreList = []

    lastStateIn = None
    lastMask = None
    logic = tttl.Logic()
    while True:
        status = logic.getStatus()
        tttc.printStatus(status)

        print("========")

        nextAction = None
        lastScore = None
        
        if status.actor != None:
            dataIn = status2In(status)
            nextAction, nextBestScore = sess.run([qdnOut,scoreMax], feed_dict={qdnIn: [dataIn]})
            lastScore = ACT_SCORE + BETA * nextBestScore
        elif status.winner :
            
        
        
            if lastStateIn != None:
                stateInList.append(lastStateIn)
