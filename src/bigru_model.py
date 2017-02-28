import random

import numpy as np
import tensorflow as tf

import data_util
import decoder_util

emb_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
fc_layer = tf.contrib.layers.fully_connected

class BiGRUModel(object):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 buckets,
                 state_size,
                 num_layers,
                 embedding_size,
                 max_gradient,
                 batch_size,
                 learning_rate,
                 use_lstm=False,
                 num_samples=10000,
                 forward_only=False,
                 feed_previous=False,
                 dtype=tf.float32):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.buckets = buckets
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.encoder_inputs = tf.placeholder(
            tf.int32, shape=[self.batch_size, None])
        self.decoder_inputs = tf.placeholder(
            tf.int32, shape=[self.batch_size, None])
        self.decoder_targets = tf.placeholder(
            tf.int32, shape=[self.batch_size, None])
        self.encoder_len = tf.placeholder(tf.int32, shape=[self.batch_size])
        self.decoder_len = tf.placeholder(tf.int32, shape=[self.batch_size])

        single_cell = tf.contrib.rnn.GRUCell(state_size)
        if use_lstm:
            single_cell = tf.contrib.rnn.BasicLSTMCell(state_size)
        cell = single_cell
        if num_layers > 1:
            cell = tf.contrib.rnn.MultiRNNCell([single_cell] * num_layers)
        if not forward_only:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=0.50)

        with tf.variable_scope("seq2seq", dtype=dtype):
            with tf.variable_scope("encoder"):

                encoder_emb = tf.get_variable(
                    "embedding", [source_vocab_size, embedding_size],
                    initializer=emb_init)

                encoder_inputs_emb = tf.nn.embedding_lookup(
                    encoder_emb, self.encoder_inputs)

                encoder_outputs, encoder_states = \
                    tf.nn.bidirectional_dynamic_rnn(
                        cell, cell, encoder_inputs_emb,
                        sequence_length=self.encoder_len, dtype=dtype)

            with tf.variable_scope("init_state"):
                init_state = fc_layer(
                    tf.concat(encoder_states, 1), state_size)

            with tf.variable_scope("decoder"):
                #TODO encoder and decoder state size must fit for attention
                att_states = fc_layer(
                    tf.concat(encoder_outputs, 2), state_size)

                att_keys, att_values, att_scfn, att_cofn = \
                    decoder_util.prepare_attention(
                        att_states, "bahdanau", state_size)

                decoder_emb = tf.get_variable(
                    "embedding", [target_vocab_size, embedding_size],
                    initializer=emb_init)

                if not forward_only:
                    decoder_fn = decoder_util.attention_decoder_fn_train(
                        init_state, att_keys, att_values, att_scfn, att_cofn)

                    decoder_inputs_emb = tf.nn.embedding_lookup(
                        decoder_emb, self.decoder_inputs)

                    outputs, final_state, final_context_state = \
                        tf.contrib.seq2seq.dynamic_rnn_decoder(
                            cell, decoder_fn, inputs=decoder_inputs_emb,
                            sequence_length=self.decoder_len)

                    with tf.variable_scope("proj") as scope:
                        outputs_logits = fc_layer(
                            outputs, target_vocab_size, scope=scope)

                    self.outputs = outputs_logits

                    weights = tf.sequence_mask(
                        self.decoder_len, dtype=tf.float32)

                    loss_t = tf.contrib.seq2seq.sequence_loss(
                        outputs_logits, self.decoder_targets, weights,
                        average_across_timesteps=False,
                        average_across_batch=True)
                    self.loss = tf.reduce_sum(loss_t)

                    params = tf.trainable_variables()
                    opt = tf.train.AdadeltaOptimizer(
                        self.learning_rate, epsilon=1e-6)
                    gradients = tf.gradients(self.loss, params)
                    clipped_gradients, norm = \
                        tf.clip_by_global_norm(gradients, max_gradient)
                    self.updates = opt.apply_gradients(
                        zip(clipped_gradients, params),
                        global_step=self.global_step)

                else:
                    self.loss = tf.constant(0)
                    with tf.variable_scope("proj") as scope:
                        output_fn = lambda x: fc_layer(
                            x, target_vocab_size, scope=scope)

                    decoder_fn = \
                        decoder_util.attention_decoder_fn_inference(
                            output_fn,
                            init_state, att_keys, att_values,
                            att_scfn, att_cofn, decoder_emb,
                            data_util.ID_GO, data_util.ID_EOS,
                            20, target_vocab_size)  # TODO maxlength 20

                    outputs, final_state, final_context_state = \
                        tf.contrib.seq2seq.dynamic_rnn_decoder(
                            cell, decoder_fn, inputs=None,
                            sequence_length=None)
                    self.outputs = outputs

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)

    def step(self,
             session,
             encoder_inputs,
             decoder_inputs,
             encoder_len,
             decoder_len,
             forward_only):

        input_feed = {}
        input_feed[self.encoder_inputs.name] = encoder_inputs
        input_feed[self.decoder_inputs.name] = decoder_inputs[:, :-1]
        input_feed[self.decoder_targets.name] = decoder_inputs[:, 1:]
        input_feed[self.encoder_len] = encoder_len
        input_feed[self.decoder_len] = decoder_len

        if forward_only:
            output_feed = [self.loss, self.outputs]
        else:
            output_feed = [self.loss, self.updates]

        outputs = session.run(output_feed, input_feed)
        return outputs


    def add_pad(self, data, fixlen):
        data = map(lambda x: x + [data_util.ID_PAD] * (fixlen - len(x)), data)
        data = list(data)
        return np.asarray(data)

    def get_batch(self, data, bucket_id):
        encoder_inputs, decoder_inputs = [], []
        encoder_len, decoder_len = [], []

        # Get a random batch of encoder and decoder inputs from data,
        # and add GO to decoder.
        for _ in range(self.batch_size):
            encoder_input, decoder_input = random.choice(data[bucket_id])

            encoder_inputs.append(encoder_input)
            encoder_len.append(len(encoder_input))

            decoder_inputs.append(decoder_input)
            decoder_len.append(len(decoder_input))

        batch_enc_len = max(encoder_len)
        batch_dec_len = max(decoder_len)

        encoder_inputs = self.add_pad(encoder_inputs, batch_enc_len)
        decoder_inputs = self.add_pad(decoder_inputs, batch_dec_len)
        encoder_len = np.asarray(encoder_len)
        # decoder_input has both <GO> and <EOS>
        # len(decoder_input)-1 is number of steps in the decoder.
        decoder_len = np.asarray(decoder_len) - 1

        return encoder_inputs, decoder_inputs, encoder_len, decoder_len
