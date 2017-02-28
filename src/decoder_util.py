import tensorflow as tf

from tensorflow.contrib.layers.python.layers import layers
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope
from tensorflow.python.util import nest


def attention_decoder_fn_train(encoder_state,
                               attention_keys,
                               attention_values,
                               attention_score_fn,
                               attention_construct_fn,
                               name=None):

    with ops.name_scope(name, "attention_decoder_fn_train", [
        encoder_state, attention_keys, attention_values, attention_score_fn,
        attention_construct_fn]):
        pass

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with ops.name_scope(
                name, "attention_decoder_fn_train",
                [time, cell_state, cell_input, cell_output, context_state]):
            if cell_state is None:  # first call, return encoder_state
                cell_state = encoder_state

                # init attention
                attention = _init_attention(encoder_state)
            else:
                # construct attention
                attention = attention_construct_fn(cell_output, attention_keys,
                                                   attention_values)
                cell_output = attention

            # combine cell_input and attention
            next_input = array_ops.concat([cell_input, attention], 1)

            return (None, cell_state, next_input, cell_output, context_state)

    return decoder_fn


def attention_decoder_fn_inference(output_fn,
                                   encoder_state,
                                   attention_keys,
                                   attention_values,
                                   attention_score_fn,
                                   attention_construct_fn,
                                   embeddings,
                                   start_of_sequence_id,
                                   end_of_sequence_id,
                                   maximum_length,
                                   num_decoder_symbols,
                                   dtype=dtypes.int32,
                                   name=None):
    with ops.name_scope(name, "attention_decoder_fn_inference", [
        output_fn, encoder_state, attention_keys, attention_values,
        attention_score_fn, attention_construct_fn, embeddings,
        start_of_sequence_id, end_of_sequence_id, maximum_length,
        num_decoder_symbols, dtype
    ]):
        start_of_sequence_id = ops.convert_to_tensor(
            start_of_sequence_id, dtype)
        end_of_sequence_id = ops.convert_to_tensor(end_of_sequence_id, dtype)
        maximum_length = ops.convert_to_tensor(maximum_length, dtype)
        num_decoder_symbols = ops.convert_to_tensor(num_decoder_symbols, dtype)
        encoder_info = nest.flatten(encoder_state)[0]
        batch_size = encoder_info.get_shape()[0].value
        if output_fn is None:
            output_fn = lambda x: x
        if batch_size is None:
            batch_size = array_ops.shape(encoder_info)[0]

    def decoder_fn(time, cell_state, cell_input, cell_output, context_state):
        with ops.name_scope(
                name, "attention_decoder_fn_inference",
                [time, cell_state, cell_input, cell_output, context_state]):
            if cell_input is not None:
                raise ValueError("Expected cell_input to be None, but saw: %s" %
                                 cell_input)
            if cell_output is None:
                # invariant that this is time == 0
                next_input_id = array_ops.ones(
                    [batch_size, ], dtype=dtype) * (start_of_sequence_id)
                done = array_ops.zeros([batch_size, ], dtype=dtypes.bool)
                cell_state = encoder_state
                cell_output = array_ops.zeros(
                    [num_decoder_symbols], dtype=dtypes.float32)
                cell_input = array_ops.gather(embeddings, next_input_id)

                # init attention
                attention = _init_attention(encoder_state)
            else:
                # construct attention
                attention = attention_construct_fn(
                    cell_output, attention_keys, attention_values)
                cell_output = attention

                # argmax decoder
                cell_output = output_fn(cell_output)  # logits
                next_input_id = math_ops.cast(
                    math_ops.argmax(cell_output, 1), dtype=dtype)
                done = math_ops.equal(next_input_id, end_of_sequence_id)
                cell_input = array_ops.gather(embeddings, next_input_id)

            # combine cell_input and attention
            next_input = array_ops.concat([cell_input, attention], 1)

            # if time > maxlen, return all true vector
            done = control_flow_ops.cond(
                math_ops.greater(time, maximum_length),
                lambda: array_ops.ones([batch_size, ], dtype=dtypes.bool),
                lambda: done)
            return (done, cell_state, next_input, cell_output, context_state)

    return decoder_fn


## Helper functions ##
def prepare_attention(attention_states,
                      attention_option,
                      num_units,
                      reuse=False):
    # Prepare attention keys / values from attention_states
    with variable_scope.variable_scope("attention_keys", reuse=reuse) as scope:
        attention_keys = layers.linear(
            attention_states, num_units, biases_initializer=None, scope=scope)
    attention_values = attention_states

    # Attention score function
    attention_score_fn = _create_attention_score_fn(
        "attention_score", num_units, attention_option, reuse)

    # Attention construction function
    attention_construct_fn = _create_attention_construct_fn(
        "attention_construct", num_units, attention_score_fn, reuse)

    return (attention_keys, attention_values, attention_score_fn,
            attention_construct_fn)


def _init_attention(encoder_state):
    # Multi- vs single-layer
    # TODO(thangluong): is this the best way to check?
    if isinstance(encoder_state, tuple):
        top_state = encoder_state[-1]
    else:
        top_state = encoder_state

    # LSTM vs GRU
    if isinstance(top_state, core_rnn_cell_impl.LSTMStateTuple):
        attn = array_ops.zeros_like(top_state.h)
    else:
        attn = array_ops.zeros_like(top_state)

    return attn


def _create_attention_construct_fn(name, num_units, attention_score_fn, reuse):
    with variable_scope.variable_scope(name, reuse=reuse) as scope:

        def construct_fn(attention_query, attention_keys, attention_values):
            context = attention_score_fn(attention_query, attention_keys,
                                         attention_values)
            concat_input = array_ops.concat([attention_query, context], 1)
            attention = layers.linear(
                concat_input, num_units, biases_initializer=None, scope=scope)
            return attention

        return construct_fn


# keys: [batch_size, attention_length, attn_size]
# query: [batch_size, 1, attn_size]
# return weights [batch_size, attention_length]
@function.Defun(func_name="attn_add_fun", noinline=True)
def _attn_add_fun(v, keys, query):
    return math_ops.reduce_sum(v * math_ops.tanh(keys + query), [2])


@function.Defun(func_name="attn_mul_fun", noinline=True)
def _attn_mul_fun(keys, query):
    return math_ops.reduce_sum(keys * query, [2])


def _create_attention_score_fn(name,
                               num_units,
                               attention_option,
                               reuse,
                               dtype=dtypes.float32):
    with variable_scope.variable_scope(name, reuse=reuse):
        if attention_option == "bahdanau":
            query_w = variable_scope.get_variable(
                "attnW", [num_units, num_units], dtype=dtype)
            score_v = variable_scope.get_variable(
                "attnV", [num_units], dtype=dtype)

        def attention_score_fn(query, keys, values):
            if attention_option == "bahdanau":
                # transform query
                query = math_ops.matmul(query, query_w)

                # reshape query: [batch_size, 1, num_units]
                query = array_ops.reshape(query, [-1, 1, num_units])

                # attn_fun
                scores = _attn_add_fun(score_v, keys, query)
            elif attention_option == "luong":
                # reshape query: [batch_size, 1, num_units]
                query = array_ops.reshape(query, [-1, 1, num_units])

                # attn_fun
                scores = _attn_mul_fun(keys, query)
            else:
                raise ValueError("Unknown attention option %s!" %
                                 attention_option)

            # Compute alignment weights
            #   scores: [batch_size, length]
            #   alignments: [batch_size, length]
            # TODO(thangluong): not normalize over padding positions.
            alignments = nn_ops.softmax(scores)

            # Now calculate the attention-weighted vector.
            alignments = array_ops.expand_dims(alignments, 2)
            context_vector = math_ops.reduce_sum(alignments * values, [1])
            context_vector.set_shape([None, num_units])

            return context_vector

        return attention_score_fn
