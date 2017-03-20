# Bi-GRU Seq2seq Text Summarization

This is an implementation of sequence-to-sequence model using a bidirectional GRU encoder and a GRU decoder. This project aims to help people start working on **Abstractive Short Text Summarization** immediately. And hopefully, it may also work on machine translation tasks. 

##Dataset
Please check [harvardnlp/sent-summary](https://github.com/harvardnlp/sent-summary).

##Usage
Please download the dataset and put all `.txt` into `data/`. 

```python3 script/train.py``` can reproduce the experiments shown below. 

By doing so, it will train 200k batches first. Then do generation on `[giga, duc2003, duc2004]` with beam_size in `[1, 10]` respectively every 20k batches. It will terminate at 300k batches. Also, the model will be saved every 20k batches. 

```python3 script/test.py``` will automatically use the most updated model to do generation. 

For advanced users, ```python3 src/summarization.py -h``` can print help. Please check the code for details. 

##Implementation Details

###Attention Mechanism
The attention mechanism follows [Bahdanau et. al](https://arxiv.org/abs/1409.0473).

We follow the implementation in tf.contrib.seq2seq. We refine the softmax function in attention so that paddings always get 0. 

###Beam Search
For simplicity and flexibility, we implement the beam search algorithm in python while leave the network part in tensorflow. In testing, we consider batch\_size as beam\_size. The tensorflow graph will generate only 1 word, then some python code will create a new batch according to the result. By iteratively doing so, beam search result is generated. 

Check `step_beam(...)` in `bigru_model.py` for details.

##Results


##Requirement
- Python3
- Tensorflow r1.0

##TODO
- Improve automatic scripts by parameterizing magic numbers. 
- Some tricks caused by new tensorflow seq2seq framework. 
