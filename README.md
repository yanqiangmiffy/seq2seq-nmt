<div align=center><img width="968" height="450" src="https://github.com/yanqiangmiffy/seq2seq-nmt/blob/master/assets/test.png"/></div>
# seq2seq-nmt

基于Keras实现seq2seq，进行英文到中文的翻译

![](https://github.com/yanqiangmiffy/seq2seq-nmt/blob/master/assets/seq2seq.png)


## 模型结构
![](https://github.com/yanqiangmiffy/seq2seq-nmt/blob/master/assets/model.png)

## 推理模型结构
- encoder

![](https://github.com/yanqiangmiffy/seq2seq-nmt/blob/master/assets/encoder.png)

- decoder

![](https://github.com/yanqiangmiffy/seq2seq-nmt/blob/master/assets/decoder.png)

## 测试结果
```text
Is it all there?
全都在那裡嗎？

Is it too salty?
还有多余的盐吗？

Is she Japanese?
她是日本人嗎？

Is this a river?
這是一條河嗎?

Isn't that mine?
那是我的吗？

It is up to you.
由你來決定。
```
```text
python predict.py --eng_sent "It's a nice day."

今天天氣很好。
```

## 参考：

https://github.com/pjgao/seq2seq_keras/blob/master/seq2seq_keras.ipynb

https://github.com/keras-team/keras/blob/master/examples/lstm_seq2seq.py

https://blog.csdn.net/PIPIXIU/article/details/81016974