from model import Seq2Seq
from utils import load_data
from keras.utils import plot_model
import os
os.environ["PATH"] += os.pathsep + 'E:/Program Files (x86)/Graphviz2.38/bin'

# 参数设置
file_path='data/cmn.txt'
n_units = 256
batch_size = 64
epoch = 200
num_samples = 10000

# 加载数据
input_texts,target_texts,target_dict,target_dict_reverse,\
    output_length,input_feature_length,output_feature_length,\
    encoder_input,decoder_input,decoder_output=load_data(file_path,num_samples)

seq2seq=Seq2Seq(input_feature_length,output_feature_length,n_units)
model_train,encoder_infer,decoder_infer=seq2seq.create_model()


# 查看模型结构
plot_model(to_file='assets/model.png',model=model_train,show_shapes=True)
plot_model(to_file='assets/encoder.png',model=encoder_infer,show_shapes=True)
plot_model(to_file='assets/decoder.png',model=decoder_infer,show_shapes=True)

model_train.compile(optimizer='rmsprop', loss='categorical_crossentropy')
encoder_infer.compile(optimizer='rmsprop')
decoder_infer.compile(optimizer='rmsprop')

print(model_train.summary())
print(encoder_infer.summary())
print(decoder_infer.summary())

# 模型训练
model_train.fit([encoder_input,decoder_input],decoder_output,batch_size=batch_size,epochs=epoch,validation_split=0.2)

model_train.save("result/model_train.h5")
encoder_infer.save("result/encoder_infer.h5")
decoder_infer.save("result/decoder_infer.h5")
