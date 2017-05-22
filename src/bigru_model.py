import random

import numpy as np
import tensorflow as tf

import data_util

emb_init = tf.truncated_normal_initializer(mean=0.0, stddev=0.01)
fc_layer = tf.contrib.layers.fully_connected

def seq2seq(encoder_inputs_emb,
            decoder_inputs_emb,
            encoder_len,
            decoder_len,
            prev_att,
            beam_tok,
            decoder_emb,
            target_vocab_size,
            batch_size,
            state_size,
            embedding_size,
            fw_cell,
            bw_cell,
            dc_cell,
            forward_only,
            scope,
            reuse,
            dtype):
    with tf.variable_scope(scope or "seq2seq", reuse=reuse):
        with tf.variable_scope("encoder"):

            encoder_outputs, encoder_states = \
                tf.nn.bidirectional_dynamic_rnn(
                    fw_cell, bw_cell, encoder_inputs_emb,
                    sequence_length=encoder_len, dtype=dtype)

        with tf.variable_scope("init_state"):
            init_state = fc_layer(
                tf.concat(encoder_states, 1), state_size,
                activation_fn=None)
            # the shape of bidirectional_dynamic_rnn is weird
            # None for batch_size
            init_state.set_shape([batch_size, state_size])
            att_states = tf.concat(encoder_outputs, 2)
            att_states.set_shape([batch_size, None, state_size*2])

        with tf.variable_scope("attention"):
            attention = tf.contrib.seq2seq.BahdanauAttention(
                state_size, att_states, encoder_len)
            decoder_cell = tf.contrib.seq2seq.DynamicAttentionWrapper(
                dc_cell, attention, state_size * 2)
            wrapper_state = tf.contrib.seq2seq.DynamicAttentionWrapperState(
                init_state, prev_att)

        with tf.variable_scope("decoder"):

            decoder_cell = tf.contrib.rnn.OutputProjectionWrapper(
                decoder_cell, target_vocab_size)

            if not forward_only:
                helper = tf.contrib.seq2seq.TrainingHelper(
                    decoder_inputs_emb, decoder_len)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, wrapper_state)

                outputs, final_state = \
                    tf.contrib.seq2seq.dynamic_decode(decoder)

                outputs_logits = outputs[0]
                return outputs_logits

            else:
                st_toks = tf.convert_to_tensor(
                    [data_util.ID_GO]*batch_size, dtype=tf.int32)

                helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                    decoder_emb, st_toks, data_util.ID_EOS)

                decoder = tf.contrib.seq2seq.BasicDecoder(
                    decoder_cell, helper, wrapper_state)

                # outputs, final_state = \
                #     tf.contrib.seq2seq.dynamic_decode(decoder)
                #
                # self.outputs = outputs[0]

                # single step decode for beam search
                # fake variable scope "decoder" to simulate dynamic_decode
                with tf.variable_scope("decoder"):
                    beam_emb = tf.nn.embedding_lookup(
                        decoder_emb, beam_tok)
                    beam_outputs, beam_nxt_state, _, _ = \
                        decoder.step(0, beam_emb, wrapper_state)
                    beam_logsoftmax = \
                        tf.nn.log_softmax(beam_outputs[0])
                return init_state, att_states, \
                    beam_logsoftmax, beam_nxt_state



class BiGRUModel(object):

    def __init__(self,
                 source_vocab_size,
                 target_vocab_size,
                 state_size,
                 num_layers,
                 embedding_size,
                 max_gradient,
                 batch_size,
                 learning_rate,
                 forward_only=False,
                 dtype=tf.float32):

        self.source_vocab_size = source_vocab_size
        self.target_vocab_size = target_vocab_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.state_size = state_size

        self.encoder_inputs = tf.placeholder(
            tf.int32, shape=[self.batch_size, None], name="encoder_inputs")
        self.decoder_inputs = tf.placeholder(
            tf.int32, shape=[self.batch_size, None], name="decoder_inputs")
        self.decoder_targets = tf.placeholder(
            tf.int32, shape=[self.batch_size, None], name="decoder_targets")
        self.encoder_len = tf.placeholder(
            tf.int32, shape=[self.batch_size], name="encoder_len")
        self.decoder_len = tf.placeholder(
            tf.int32, shape=[self.batch_size], name="decoder_len")
        self.beam_tok = tf.placeholder(
            tf.int32, shape=[self.batch_size], name="beam_tok")
        self.prev_att = tf.placeholder(
            tf.float32, shape=[self.batch_size, state_size * 2],
            name="prev_att")
        self.source_vocab_freq = tf.placeholder(
            tf.float32, shape=[source_vocab_size], name="source_freq")
        self.target_vocab_freq = tf.placeholder(
            tf.float32, shape=[target_vocab_size], name="target_freq")

        encoder_fw_cell = tf.contrib.rnn.GRUCell(state_size)
        encoder_bw_cell = tf.contrib.rnn.GRUCell(state_size)
        decoder_cell = tf.contrib.rnn.GRUCell(state_size)

        if not forward_only:
            encoder_fw_cell = tf.contrib.rnn.DropoutWrapper(
                encoder_fw_cell, output_keep_prob=0.50)
            encoder_bw_cell = tf.contrib.rnn.DropoutWrapper(
                encoder_bw_cell, output_keep_prob=0.50)
            decoder_cell = tf.contrib.rnn.DropoutWrapper(
                decoder_cell, output_keep_prob=0.50)

        with tf.variable_scope("embeddings"):
            encoder_emb = tf.get_variable(
                "enc_embedding", [source_vocab_size, embedding_size],
                initializer=emb_init)

            decoder_emb = tf.get_variable(
                "dec_embedding", [target_vocab_size, embedding_size],
                initializer=emb_init)

            encoder_inputs_emb = tf.nn.embedding_lookup(
                encoder_emb, self.encoder_inputs)

            decoder_inputs_emb = tf.nn.embedding_lookup(
                decoder_emb, self.decoder_inputs)

        if not forward_only:
            outputs_logits = seq2seq(
                encoder_inputs_emb, decoder_inputs_emb,
                self.encoder_len, self.decoder_len,
                self.prev_att, None, None, target_vocab_size,
                batch_size, state_size, embedding_size,
                encoder_fw_cell, encoder_bw_cell, decoder_cell,
                forward_only=False, scope='seq2seq', reuse=False, dtype=dtype)

            weights = tf.sequence_mask(
                self.decoder_len, dtype=tf.float32)

            loss_t = tf.contrib.seq2seq.sequence_loss(
                outputs_logits, self.decoder_targets, weights,
                average_across_timesteps=False,
                average_across_batch=False)
            self.loss = tf.reduce_sum(loss_t) / self.batch_size

            params = tf.trainable_variables()
            opt = tf.train.AdadeltaOptimizer(
                self.learning_rate, epsilon=1e-6)
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = \
                tf.clip_by_global_norm(gradients, max_gradient)
            self.updates = opt.apply_gradients(
                zip(clipped_gradients, params),
                global_step=self.global_step)

            tf.summary.scalar('loss', self.loss)

            def normalize_embedding(emb, freq):
                e = tf.reduce_sum(tf.reshape(freq, [-1, 1]) * emb, axis=0)
                v = tf.reduce_sum(
                    tf.reshape(freq, [-1, 1]) * (emb - e) * (emb - e),
                    axis=0)
                return tf.assign(emb, (emb - e) / tf.sqrt(v + 1e-6))
            self.update_enc_emb = normalize_embedding(
                encoder_emb, self.source_vocab_freq)
            self.update_dec_emb = normalize_embedding(
                decoder_emb, self.target_vocab_freq)
        else:
            self.loss = tf.constant(0)
            self.init_state, self.att_states, \
                self.beam_logsoftmax, self.beam_nxt_state = \
                seq2seq(encoder_inputs_emb, None,
                self.encoder_len, None,
                self.prev_att, self.beam_tok, decoder_emb, target_vocab_size,
                batch_size, state_size, embedding_size,
                encoder_fw_cell, encoder_bw_cell, decoder_cell,
                forward_only=True, scope='seq2seq', reuse=False, dtype=dtype)


        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=0)
        self.summary_merge = tf.summary.merge_all()

    def norm_emb(self,
                 session,
                 source_vocab_freq,
                 target_vocab_freq):
        input_feed = {}
        input_feed[self.source_vocab_freq] = source_vocab_freq
        input_feed[self.target_vocab_freq] = target_vocab_freq
        output_feed = [self.update_dec_emb, self.update_dec_emb]
        session.run(output_feed, input_feed)

    def step(self,
             session,
             encoder_inputs,
             decoder_inputs,
             encoder_len,
             decoder_len,
             forward_only,
             summary_writer=None):

        # dim fit is important for sequence_mask
        # TODO better way to use sequence_mask
        if encoder_inputs.shape[1] != max(encoder_len):
            raise ValueError("encoder_inputs and encoder_len does not fit")
        if not forward_only and \
            decoder_inputs.shape[1] != max(decoder_len) + 1:
            raise ValueError("decoder_inputs and decoder_len does not fit")
        input_feed = {}
        input_feed[self.encoder_inputs] = encoder_inputs
        input_feed[self.decoder_inputs] = decoder_inputs[:, :-1]
        input_feed[self.decoder_targets] = decoder_inputs[:, 1:]
        input_feed[self.encoder_len] = encoder_len
        input_feed[self.decoder_len] = decoder_len
        input_feed[self.prev_att] = np.zeros(
            [self.batch_size, 2 * self.state_size])

        if forward_only:
            output_feed = [self.loss, self.outputs]
        else:
            output_feed = [self.loss, self.updates]

        if summary_writer:
            output_feed += [self.summary_merge, self.global_step]

        outputs = session.run(output_feed, input_feed)

        if summary_writer:
            summary_writer.add_summary(outputs[2], outputs[3])
        return outputs[:2]

    def step_beam(self,
                  session,
                  encoder_inputs,
                  encoder_len,
                  max_len=10,
                  geneos=True):

        beam_size = self.batch_size

        if encoder_inputs.shape[0] == 1:
            encoder_inputs = np.repeat(encoder_inputs, beam_size, axis=0)
            encoder_len = np.repeat(encoder_len, beam_size, axis=0)

        if encoder_inputs.shape[1] != max(encoder_len):
            raise ValueError("encoder_inputs and encoder_len does not fit")
        #generate attention_states
        input_feed = {}
        input_feed[self.encoder_inputs] = encoder_inputs
        input_feed[self.encoder_len] = encoder_len
        output_feed = [self.att_states, self.init_state]
        outputs = session.run(output_feed, input_feed)

        att_states = outputs[0]
        prev_state = outputs[1]
        prev_tok = np.ones([beam_size], dtype="int32") * data_util.ID_GO
        prev_att = np.zeros([self.batch_size, 2 * self.state_size])

        input_feed = {}
        input_feed[self.att_states] = att_states
        input_feed[self.encoder_len] = encoder_len

        ret = [[]] * beam_size
        neos = np.ones([beam_size], dtype="bool")

        score = np.ones([beam_size], dtype="float32") * (-1e8)
        score[0] = 0

        beam_att = np.zeros(
            [self.batch_size, self.state_size*2], dtype="float32")

        for i in range(max_len):
            input_feed[self.init_state] = prev_state
            input_feed[self.beam_tok] = prev_tok
            input_feed[self.prev_att] = beam_att
            output_feed = [self.beam_nxt_state[1],
                           self.beam_logsoftmax,
                           self.beam_nxt_state[0]]

            outputs = session.run(output_feed, input_feed)

            beam_att = outputs[0]
            tok_logsoftmax = np.asarray(outputs[1])
            tok_logsoftmax = tok_logsoftmax.reshape(
                [beam_size, self.target_vocab_size])
            if not geneos:
                tok_logsoftmax[:, data_util.ID_EOS] = -1e8

            tok_argsort = np.argsort(tok_logsoftmax, axis=1)[:, -beam_size:]
            tmp_arg0 = np.arange(beam_size).reshape([beam_size, 1])
            tok_argsort_score = tok_logsoftmax[tmp_arg0, tok_argsort]
            tok_argsort_score *= neos.reshape([beam_size, 1])
            tok_argsort_score += score.reshape([beam_size, 1])
            all_arg = np.argsort(tok_argsort_score.flatten())[-beam_size:]
            arg0 = all_arg // beam_size #previous id in batch
            arg1 = all_arg % beam_size
            prev_tok = tok_argsort[arg0, arg1] #current word
            prev_state = outputs[2][arg0]
            score = tok_argsort_score[arg0, arg1]

            neos = neos[arg0] & (prev_tok != data_util.ID_EOS)

            ret_t = []
            for j in range(beam_size):
                ret_t.append(ret[arg0[j]] + [prev_tok[j]])

            ret = ret_t
        return ret[-1]



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
