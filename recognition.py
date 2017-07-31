# encoding:UTF-8

import tensorflow as tf
import sys, os
 
# 加载图像分类标签
labels = []
for label in tf.gfile.GFile("output_labels.txt"):
    labels.append(label.rstrip())
 
# 加载Graph
with tf.gfile.FastGFile("output_graph.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    tf.import_graph_def(graph_def, name='')
 
with tf.Session() as sess:
    softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')

    rights = 0.0
    for image_file in sys.argv[1:]:
        print("===========================\n%s\n===========================" % os.path.basename(image_file))
        nam = os.path.basename(image_file).split('-')[0]
 
        # 读取图像
        image = tf.gfile.FastGFile(image_file, 'rb').read()
        predict = sess.run(softmax_tensor, {'DecodeJpeg/contents:0': image})
     
        # 根据分类概率进行排序
        top = predict[0].argsort()[-len(predict[0]):][::-1]
        if labels[top[0]] == nam:
            rights += 1.0
        for index in top:
            human_string = labels[index]
            score = predict[0][index]
            print(human_string, score)

    size = len(sys.argv)-1
    print("===========================\nresult: rights(%d) * 100.0 / size(%d) = %.2f%%" % (rights, size, rights * 100.0 / float(size)))
