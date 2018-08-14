from keras.layers import Input,LSTM,Dense
from keras.models import Model


class Seq2Seq(object):

    def __init__(self,n_input,n_output,n_units):
        self.n_input=n_input
        self.n_output=n_output
        self.n_units=n_units

    def create_model(self):
        # 训练阶段
        # encoder
        encoder_input = Input(shape=(None, self.n_input))
        # encoder输入维度n_input为每个时间步的输入xt的维度，这里是用来one-hot的英文字符数
        encoder = LSTM(self.n_units, return_state=True)
        # n_units为LSTM单元中每个门的神经元的个数，return_state设为True时才会返回最后时刻的状态h,c
        _, encoder_h, encoder_c = encoder(encoder_input)
        encoder_state = [encoder_h, encoder_c]
        # 保留下来encoder的末状态作为decoder的初始状态

        # decoder
        decoder_input = Input(shape=(None, self.n_output))
        # decoder的输入维度为中文字符数
        decoder = LSTM(self.n_units, return_sequences=True, return_state=True)
        # 训练模型时需要decoder的输出序列来与结果对比优化，故return_sequences也要设为True
        decoder_output, _, _ = decoder(decoder_input, initial_state=encoder_state)
        # 在训练阶段只需要用到decoder的输出序列，不需要用最终状态h.c
        decoder_dense = Dense(self.n_output, activation='softmax')
        decoder_output = decoder_dense(decoder_output)
        # 输出序列经过全连接层得到结果

        # 生成的训练模型
        model = Model([encoder_input, decoder_input], decoder_output)
        # 第一个参数为训练模型的输入，包含了encoder和decoder的输入，第二个参数为模型的输出，包含了decoder的输出

        # 推理阶段，用于预测过程
        # 推断模型—encoder
        encoder_infer = Model(encoder_input, encoder_state)

        # 推断模型-decoder
        decoder_state_input_h = Input(shape=(self.n_units,))
        decoder_state_input_c = Input(shape=(self.n_units,))
        decoder_state_input = [decoder_state_input_h, decoder_state_input_c]  # 上个时刻的状态h,c

        decoder_infer_output, decoder_infer_state_h, decoder_infer_state_c = decoder(decoder_input,
                                                                                     initial_state=decoder_state_input)
        decoder_infer_state = [decoder_infer_state_h, decoder_infer_state_c]  # 当前时刻得到的状态
        decoder_infer_output = decoder_dense(decoder_infer_output)  # 当前时刻的输出
        decoder_infer = Model([decoder_input] + decoder_state_input, [decoder_infer_output] + decoder_infer_state)

        return model, encoder_infer, decoder_infer


