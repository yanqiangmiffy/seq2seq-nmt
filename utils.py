import pandas as pd
import numpy as np


def load_data(filepath,num_samples=10000):
    data=pd.read_table(filepath,header=None).iloc[:num_samples,:,]
    data.columns=['inputs','targets']
    data['targets']=data['targets'].apply(lambda x:'\t'+x+'\n')

    input_texts=data['inputs'].values.tolist()
    target_texts=data['targets'].values.tolist()

    input_characters=sorted(list(set(data.inputs.unique().sum())))
    targets_characters=sorted(list(set(data.targets.unique().sum())))
    # print(targets_characters)

    input_length=max([len(i) for i in input_texts])
    output_length=max([len(i) for i in target_texts])
    input_feature_length=len(input_characters)
    output_feature_length=len(targets_characters)

    encoder_input=np.zeros((num_samples,input_length,input_feature_length))
    decoder_input=np.zeros((num_samples,output_length,output_feature_length))
    decoder_output=np.zeros((num_samples,output_length,output_feature_length))

    input_dict={char:index for index,char in enumerate(input_characters)}
    input_dict_reverse={index:char for index,char in enumerate(input_characters)}

    target_dict={char:index for index,char in enumerate(targets_characters)}
    target_dict_reverse={index:char for index,char in enumerate(targets_characters)}

    for seq_index,seq in enumerate(input_texts):
        for char_index,char in enumerate(seq):
            encoder_input[seq_index,char_index,input_dict[char]]=1

    for seq_index,seq in enumerate(target_texts):
        for char_index,char in enumerate(seq):
            decoder_input[seq_index,char_index,target_dict[char]]=1.0
            if char_index>0:
                decoder_output[seq_index,char_index-1,target_dict[char]]=1.0

    # print(' '.join([input_dict_reverse[np.argmax(i)] for i in encoder_input[0] if max(i)!=0]))
    # print(' '.join([target_dict_reverse[np.argmax(i)] for i in decoder_output[0] if max(i)!=0]))
    # print(' '.join([target_dict[np.argmax(i)] for i in decoder_input[0] if max(i)!=0]))
    data_dict=dict()
    data_dict['input_texts']=input_texts
    data_dict['target_texts']=target_texts
    data_dict['input_dict']=input_dict
    data_dict['input_dict_reverse']=input_dict_reverse
    data_dict['target_dict']=target_dict
    data_dict['target_dict_reverse']=target_dict_reverse
    data_dict['input_length']=input_length
    data_dict['output_length']=output_length
    data_dict['input_feature_length']=input_feature_length
    data_dict['output_feature_length']=output_feature_length
    data_dict['encoder_input']=encoder_input
    data_dict['decoder_input']=decoder_input
    data_dict['decoder_output']=decoder_output
    return input_texts,target_texts,input_dict,target_dict,target_dict_reverse,output_length,\
           input_feature_length,output_feature_length,encoder_input,decoder_input,decoder_output

if __name__ == '__main__':
    data_path = 'data/cmn.txt'
    load_data(data_path)
