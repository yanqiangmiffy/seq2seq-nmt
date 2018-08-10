from utils import load_data
from keras.models import load_model
import numpy as np

# 参数设置
file_path='data/cmn.txt'
num_samples = 10000

# 加载数据
input_texts,target_dict,target_dict_reverse, \
output_length,input_feature_length,output_feature_length,\
encoder_input,decoder_input,decoder_output=load_data(file_path,num_samples)

def predict_chinese(source,encoder_inference, decoder_inference, n_steps, features):
    #先通过推理encoder获得预测输入序列的隐状态
    state = encoder_inference.predict(source)
    #第一个字符'\t',为起始标志
    predict_seq = np.zeros((1,1,features))
    predict_seq[0,0,target_dict['\t']] = 1

    output = ''
    #开始对encoder获得的隐状态进行推理
    #每次循环用上次预测的字符作为输入来预测下一次的字符，直到预测出了终止符
    for i in range(n_steps):#n_steps为句子最大长度
        #给decoder输入上一个时刻的h,c隐状态，以及上一次的预测字符predict_seq
        yhat,h,c = decoder_inference.predict([predict_seq]+state)
        #注意，这里的yhat为Dense之后输出的结果，因此与h不同
        char_index = np.argmax(yhat[0,-1,:])
        char = target_dict_reverse[char_index]
        output += char
        state = [h,c]#本次状态做为下一次的初始状态继续传递
        predict_seq = np.zeros((1,1,features))
        predict_seq[0,0,char_index] = 1
        if char == '\n':#预测到了终止符则停下来
            break
    return output

encoder_infer=load_model('result/encoder_infer.h5')
decoder_infer=load_model('result/decoder_infer.h5')

for i in range(1000,1100):
    test = encoder_input[i:i+1,:,:]#i:i+1保持数组是三维
    out = predict_chinese(test,encoder_infer,decoder_infer,output_length,output_feature_length)
    #print(input_texts[i],'\n---\n',target_texts[i],'\n---\n',out)
    print(input_texts[i])
    print(out)