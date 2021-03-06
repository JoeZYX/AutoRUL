from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.recurrent import _standardize_args
from tensorflow.python.keras.layers.recurrent import DropoutRNNCellMixin
from tensorflow.python.keras.layers.recurrent import RNN
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.util.tf_export import keras_export
import math		
import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Flatten,TimeDistributed,LSTM,BatchNormalization,Reshape,Conv2D


class ConvRNN2D(RNN):

  def __init__(self,
               cell,
               return_sequences=False,
               return_state=False,
               go_backwards=False,
               stateful=False,
               unroll=False,
               **kwargs):

    if unroll:
      raise TypeError('Unrolling isn\'t possible with '
                      'convolutional RNNs.')
    if isinstance(cell, (list, tuple)):
      # The StackedConvRNN2DCells isn't implemented yet.
      raise TypeError('It is not possible at the moment to'
                      'stack convolutional cells.')
    super(ConvRNN2D, self).__init__(cell,
                                    return_sequences,
                                    return_state,
                                    go_backwards,
                                    stateful,
                                    unroll,
                                    **kwargs)
    self.input_spec = [InputSpec(ndim=5)]
    self.states = None
    self._num_constants = None

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    cell = self.cell
    if cell.data_format == 'channels_first':
      rows = input_shape[3]
      cols = input_shape[4]
    elif cell.data_format == 'channels_last':
      rows = input_shape[2]
      cols = input_shape[3]
    rows = conv_utils.conv_output_length(rows,
                                         cell.kernel_size[0][0],
                                         padding=cell.padding,
                                         stride=cell.strides[0],
                                         dilation=cell.dilation_rate[0])
    cols = conv_utils.conv_output_length(cols,
                                         cell.kernel_size[0][1],
                                         padding=cell.padding,
                                         stride=cell.strides[1],
                                         dilation=cell.dilation_rate[1])

    if cell.data_format == 'channels_first':
      output_shape = input_shape[:2] + (cell.filters[-1], rows, cols)
    elif cell.data_format == 'channels_last':
      output_shape = input_shape[:2] + (rows, cols, cell.filters[-1])

    if not self.return_sequences:
      output_shape = output_shape[:1] + output_shape[2:]

    if self.return_state:
      output_shape = [output_shape]
      if cell.data_format == 'channels_first':
        output_shape += [(input_shape[0], cell.filters[-1], rows, cols)
                         for _ in range(2)]
      elif cell.data_format == 'channels_last':
        output_shape += [(input_shape[0], rows, cols, cell.filters[-1])
                         for _ in range(2)]
    return output_shape

  @tf_utils.shape_type_conversion
  def build(self, input_shape):

    # Note input_shape will be list of shapes of initial states and
    # constants if these are passed in __call__.
    if self._num_constants is not None:
      constants_shape = input_shape[-self._num_constants:]  # pylint: disable=E1130
    else:
      constants_shape = None

    if isinstance(input_shape, list):
      input_shape = input_shape[0]

    batch_size = input_shape[0] if self.stateful else None
    self.input_spec[0] = InputSpec(shape=(batch_size, None) + input_shape[2:5])

    # allow cell (if layer) to build before we set or validate state_spec
    if isinstance(self.cell, Layer):
      step_input_shape = (input_shape[0],) + input_shape[2:]
      if constants_shape is not None:
        self.cell.build([step_input_shape] + constants_shape)
      else:
        self.cell.build(step_input_shape)

    # set or validate state_spec
    if hasattr(self.cell.state_size, '__len__'):
      state_size = list(self.cell.state_size)
    else:
      state_size = [self.cell.state_size]

    if self.state_spec is not None:
      # initial_state was passed in call, check compatibility
      if self.cell.data_format == 'channels_first':
        ch_dim = 1
      elif self.cell.data_format == 'channels_last':
        ch_dim = 3
      if [spec.shape[ch_dim] for spec in self.state_spec] != state_size:
        raise ValueError(
            'An initial_state was passed that is not compatible with '
            '`cell.state_size`. Received `state_spec`={}; '
            'However `cell.state_size` is '
            '{}'.format([spec.shape for spec in self.state_spec],
                        self.cell.state_size))
    else:
      if self.cell.data_format == 'channels_first':
        self.state_spec = [InputSpec(shape=(None, dim, None, None))
                           for dim in state_size]
      elif self.cell.data_format == 'channels_last':
        self.state_spec = [InputSpec(shape=(None, None, None, dim))
                           for dim in state_size]
    if self.stateful:
      self.reset_states()
    self.built = True

  def get_initial_state(self, inputs):

    # (samples, timesteps, rows, cols, filters)
    initial_state = K.zeros_like(inputs)
    # (samples, rows, cols, filters)
    initial_state = K.sum(initial_state, axis=1)
    shape = list(self.cell.kernel_shape)
    shape[-1] = self.cell.filters
    initial_state = self.cell.input_conv(initial_state,
                                         array_ops.zeros(tuple(shape)),
                                         padding=self.cell.padding)


    if hasattr(self.cell.state_size, '__len__'):
      return [initial_state for _ in self.cell.state_size]
    else:
      return [initial_state]

  def __call__(self, inputs, initial_state=None, constants=None, **kwargs):

    inputs, initial_state, constants = _standardize_args(
        inputs, initial_state, constants, self._num_constants)

    if initial_state is None and constants is None:

      return super(ConvRNN2D, self).__call__(inputs, **kwargs)

    # If any of `initial_state` or `constants` are specified and are Keras
    # tensors, then add them to the inputs and temporarily modify the
    # input_spec to include them.

    additional_inputs = []
    additional_specs = []
    if initial_state is not None:
      kwargs['initial_state'] = initial_state
      additional_inputs += initial_state
      self.state_spec = []
      for state in initial_state:
        shape = K.int_shape(state)
        self.state_spec.append(InputSpec(shape=shape))

      additional_specs += self.state_spec
    if constants is not None:
      kwargs['constants'] = constants
      additional_inputs += constants
      self.constants_spec = [InputSpec(shape=K.int_shape(constant))
                             for constant in constants]
      self._num_constants = len(constants)
      additional_specs += self.constants_spec
    # at this point additional_inputs cannot be empty
    for tensor in additional_inputs:
      if K.is_keras_tensor(tensor) != K.is_keras_tensor(additional_inputs[0]):
        raise ValueError('The initial state or constants of an RNN'
                         ' layer cannot be specified with a mix of'
                         ' Keras tensors and non-Keras tensors')

    if K.is_keras_tensor(additional_inputs[0]):
      # Compute the full input spec, including state and constants
      full_input = [inputs] + additional_inputs
      full_input_spec = self.input_spec + additional_specs
      # Perform the call with temporarily replaced input_spec
      original_input_spec = self.input_spec
      self.input_spec = full_input_spec
      output = super(ConvRNN2D, self).__call__(full_input, **kwargs)
      self.input_spec = original_input_spec
      return output
    else:
      return super(ConvRNN2D, self).__call__(inputs, **kwargs)

  def call(self,
           inputs,
           mask=None,
           training=None,
           initial_state=None,
           constants=None):
    # note that the .build() method of subclasses MUST define
    # self.input_spec and self.state_spec with complete input shapes.

    if isinstance(inputs, list):
      inputs = inputs[0]
    if initial_state is not None:
      pass
    elif self.stateful:
      initial_state = self.states
    else:
      initial_state = self.get_initial_state(inputs)

    if isinstance(mask, list):
      mask = mask[0]

    if len(initial_state) != len(self.states):
      raise ValueError('Layer has ' + str(len(self.states)) +
                       ' states but was passed ' +
                       str(len(initial_state)) +
                       ' initial states.')
    timesteps = K.int_shape(inputs)[1]

    kwargs = {}
    if generic_utils.has_arg(self.cell.call, 'training'):
      kwargs['training'] = training

    if constants:
      if not generic_utils.has_arg(self.cell.call, 'constants'):
        raise ValueError('RNN cell does not support constants')

      def step(inputs, states):
        constants = states[-self._num_constants:]
        states = states[:-self._num_constants]
        return self.cell.call(inputs, states, constants=constants,
                              **kwargs)
    else:
      def step(inputs, states):
        return self.cell.call(inputs, states, **kwargs)

    last_output, outputs, states = K.rnn(step,
                                         inputs,
                                         initial_state,
                                         constants=constants,
                                         go_backwards=self.go_backwards,
                                         mask=mask,
                                         input_length=timesteps)
    if self.stateful:
      updates = []
      for i in range(len(states)):
        updates.append(K.update(self.states[i], states[i]))
      self.add_update(updates)

    if self.return_sequences:
      output = outputs
    else:
      output = last_output

    if self.return_state:
      if not isinstance(states, (list, tuple)):
        states = [states]
      else:
        states = list(states)
      return [output] + states
    else:
      return output

  def reset_states(self, states=None):

    if not self.stateful:
      raise AttributeError('Layer must be stateful.')
    input_shape = self.input_spec[0].shape
    state_shape = self.compute_output_shape(input_shape)
    if self.return_state:
      state_shape = state_shape[0]
    if self.return_sequences:
      state_shape = state_shape[:1].concatenate(state_shape[2:])
    if None in state_shape:
      raise ValueError('If a RNN is stateful, it needs to know '
                       'its batch size. Specify the batch size '
                       'of your input tensors: \n'
                       '- If using a Sequential model, '
                       'specify the batch size by passing '
                       'a `batch_input_shape` '
                       'argument to your first layer.\n'
                       '- If using the functional API, specify '
                       'the time dimension by passing a '
                       '`batch_shape` argument to your Input layer.\n'
                       'The same thing goes for the number of rows and '
                       'columns.')

    # helper function
    def get_tuple_shape(nb_channels):
      result = list(state_shape)
      if self.cell.data_format == 'channels_first':
        result[1] = nb_channels
      elif self.cell.data_format == 'channels_last':
        result[3] = nb_channels
      else:
        raise KeyError
      return tuple(result)

    # initialize state if None
    if self.states[0] is None:
      if hasattr(self.cell.state_size, '__len__'):
        self.states = [K.zeros(get_tuple_shape(dim))
                       for dim in self.cell.state_size]
      else:
        self.states = [K.zeros(get_tuple_shape(self.cell.state_size))]
    elif states is None:
      if hasattr(self.cell.state_size, '__len__'):
        for state, dim in zip(self.states, self.cell.state_size):
          K.set_value(state, np.zeros(get_tuple_shape(dim)))
      else:
        K.set_value(self.states[0],
                    np.zeros(get_tuple_shape(self.cell.state_size)))
    else:
      if not isinstance(states, (list, tuple)):
        states = [states]
      if len(states) != len(self.states):
        raise ValueError('Layer ' + self.name + ' expects ' +
                         str(len(self.states)) + ' states, ' +
                         'but it received ' + str(len(states)) +
                         ' state values. Input received: ' + str(states))
      for index, (value, state) in enumerate(zip(states, self.states)):
        if hasattr(self.cell.state_size, '__len__'):
          dim = self.cell.state_size[index]
        else:
          dim = self.cell.state_size
        if value.shape != get_tuple_shape(dim):
          raise ValueError('State ' + str(index) +
                           ' is incompatible with layer ' +
                           self.name + ': expected shape=' +
                           str(get_tuple_shape(dim)) +
                           ', found shape=' + str(value.shape))
        # TODO(anjalisridhar): consider batch calls to `set_value`.
        K.set_value(state, value)

class ConvLSTM1DCell(DropoutRNNCellMixin, Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 input_channel, 
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 conv_activation='hard_sigmoid',
                 convolutional_type = "early",
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        """
        filters : A list , Specifies the number of filters in each layer, e.g. [10,10]
        kernel_size : A List , Same length as filters， Window size for 1D convolution e.g. [3,3]
        input_channel: int , Number of multiple time series e.g 28 sensors -->  28
        recurrent_activation : A str List, Specifies the tupe of activation functions
        
        
        """

        super(ConvLSTM1DCell, self).__init__(**kwargs)

        self.out_feature_number = input_channel
        self.convolutional_type = convolutional_type
		
        self.number_of_layer = len(filters)  
        

        self.filters = filters
        self.conv_layer_number = len(filters)
        
        self.kernel_size = []
        for index, size in enumerate(kernel_size): 
            if self.convolutional_type[index] == "hybrid":
                self.kernel_size.append(conv_utils.normalize_tuple((size,1), 2, 'kernel_size'))
            if self.convolutional_type[index] == "early":
                self.kernel_size.append(conv_utils.normalize_tuple((size,input_channel), 2, 'kernel_size'))
                self.out_feature_number = 1
                input_channel = 1

        
        self.recurrent_activation = []        
        for acti in recurrent_activation:
            self.recurrent_activation.append(activations.get(acti))

        self.conv_activation = []        
        for acti in conv_activation:
            self.conv_activation.append(activations.get(acti))
            
        self.state_size = (self.filters[-1], self.filters[-1])
        
        # =============   Each layer has the same parameter   ======================
        self.strides = conv_utils.normalize_tuple(strides, 2, 'strides')                # (1,1)
        self.padding = conv_utils.normalize_padding(padding)                            # valid
        self.data_format = conv_utils.normalize_data_format(data_format)                # None --- -1
        self.dilation_rate = conv_utils.normalize_tuple(dilation_rate, 2,'dilation_rate')
        self.activation = activations.get(activation)                                   # tanh default
        self.use_bias = use_bias                                                        # True
        self.kernel_initializer = initializers.get(kernel_initializer)                  # glorot_uniform
        self.recurrent_initializer = initializers.get(recurrent_initializer)            # orthogonal
        self.bias_initializer = initializers.get(bias_initializer)                      # zeros
        self.unit_forget_bias = unit_forget_bias                                        # True

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))


    def build(self, input_shape):
        """
        According to the input  and the structural parameters defined in the initialization, 
        the required weights are constructed
        """


        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
        if input_shape[channel_axis] is None:
            raise ValueError('The channel dimension of the inputs '
                             'should be defined. Found `None`.')
            
        input_dim = input_shape[channel_axis]  
        state_len=input_shape[1] 
        for i in self.kernel_size:
            if (i[0]%2==0):
                state_len=state_len-2*int(i[0]/2)+1
            else:
                state_len=state_len-2*int(i[0]/2)
			

        self.kernel = []
        self.recurrent_kernel = []
        self.kernel_shape = []
        self.recurrent_kernel_shape = []
        
        for index in range(len(self.filters)):
            if index==0:
                kernel_shape = self.kernel_size[index] + (input_dim, self.filters[index] * 4)
            else:
                kernel_shape = self.kernel_size[index] + (self.filters[index-1], self.filters[index] * 4)

            self.kernel_shape.append(kernel_shape)

            
            self.kernel.append(self.add_weight(shape=kernel_shape,
                                               initializer=self.kernel_initializer,
                                               name='kernel_{}'.format(index),
                                               regularizer=self.kernel_regularizer,
                                               constraint=self.kernel_constraint))
            

												 
        for index in range(len(self.filters)):

            h_state_kernel = min(self.kernel_size[index][0],   int(math.ceil(state_len/self.conv_layer_number)))
            if index == self.number_of_layer-1:
                recurrent_kernel_shape = (h_state_kernel,self.out_feature_number) + (self.filters[-1], self.filters[-1] * 5)
            else:
                recurrent_kernel_shape = (h_state_kernel,self.out_feature_number) + (self.filters[-1], self.filters[-1] * 4)
                ###################################################################################### 

            self.recurrent_kernel_shape.append(recurrent_kernel_shape)
            
            
            self.recurrent_kernel.append(self.add_weight(shape=recurrent_kernel_shape,
                                                         initializer=self.recurrent_initializer,
                                                         name='recurrent_kernel_{}'.format(index),
                                                         regularizer=self.recurrent_regularizer,
                                                         constraint=self.recurrent_constraint))
                
	        

        if self.use_bias:
            self.bias = []
            if self.unit_forget_bias:
                self._index=0
                def bias_initializer(_, *args, **kwargs):
                    return K.concatenate([self.bias_initializer((self.filters[self._index],), *args, **kwargs),
                                          initializers.Ones()((self.filters[self._index],), *args, **kwargs),
                                          self.bias_initializer((self.filters[self._index] * 2,), *args, **kwargs),
                                         ])
            else:
                bias_initializer = self.bias_initializer
                
            for index in range(self.number_of_layer):
                self._index = index
                self.bias.append(self.add_weight(shape=(self.filters[index] * 4,),
                                                 name='bias_{}'.format(index),
                                                 initializer=bias_initializer,
                                                 regularizer=self.bias_regularizer,
                                                 constraint=self.bias_constraint))
        else:
            self.bias = [None]*self.number_of_layer
     
            
        self.built = True
                                 
                                 
    def call(self, inputs, states, training=None):
        """
        inputs: shape is [batch , window_size, number_of_sensor, 1]
        """

        h_state = states[0]                                                                 # previous memory state
 
        c_state = states[1]                                                                 # previous carry state
        #c_shape = c_tm1.get_shape().as_list()                                             # [BATCH, conv_rest, 1, LAST_FILTER] 
                                                                                          # 
        # dropout matrices for input units
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        # dropout matrices for recurrent units
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(h_state, training, count=4)

        if 0 < self.dropout < 1.:
            x_i = inputs * dp_mask[0]
            x_f = inputs * dp_mask[1]
            x_c = inputs * dp_mask[2]
            x_o = inputs * dp_mask[3]
        else:
            x_i = inputs
            x_f = inputs
            x_c = inputs
            x_o = inputs

        if 0 < self.recurrent_dropout < 1.:
            h_i = h_state * rec_dp_mask[0]
            h_f = h_state * rec_dp_mask[1]
            h_c = h_state * rec_dp_mask[2]
            h_o = h_state * rec_dp_mask[3]
        else:
            h_i = h_state
            h_f = h_state
            h_c = h_state
            h_o = h_state

        for index in range(self.number_of_layer):
            # weights for inputs in FOUR GATES
            (kernel_i, kernel_f, kernel_c, kernel_o) = array_ops.split(self.kernel[index], 4, axis=3)
            # weights for hidden states in FOUR GATES
            if index == self.number_of_layer-1:
                (recurrent_kernel_i,recurrent_kernel_f,
                 recurrent_kernel_c,recurrent_kernel_o, recurrent_kernel_c_1) = array_ops.split(self.recurrent_kernel[index], 5, axis=3)			
            else:
                (recurrent_kernel_i,recurrent_kernel_f,
                 recurrent_kernel_c,recurrent_kernel_o) = array_ops.split(self.recurrent_kernel[index], 4, axis=3)
            #######################################################################################
            # weights for BIAS in FOUR GATES
            if self.use_bias:
                bias_i, bias_f, bias_c, bias_o = array_ops.split(self.bias[index], 4)
            else:
                bias_i, bias_f, bias_c, bias_o = None, None, None, None  
            x_i = self.input_conv(x_i, kernel_i, bias_i, padding=self.padding)
            x_f = self.input_conv(x_f, kernel_f, bias_f, padding=self.padding)
            x_c = self.input_conv(x_c, kernel_c, bias_c, padding=self.padding)
            x_o = self.input_conv(x_o, kernel_o, bias_o, padding=self.padding)
            
            h_i = self.recurrent_conv(h_i, recurrent_kernel_i)
            h_f = self.recurrent_conv(h_f, recurrent_kernel_f)
            h_c = self.recurrent_conv(h_c, recurrent_kernel_c)
            h_o = self.recurrent_conv(h_o, recurrent_kernel_o)
            
            if index == self.number_of_layer-1:
                #######################################################################################
                c_c = self.recurrent_conv(c_state, recurrent_kernel_c_1)

                i = self.recurrent_activation[index](x_i + h_i)
                f = self.recurrent_activation[index](x_f + h_f)
                o = self.recurrent_activation[index](x_o + h_o)

                c = f * c_c + i * self.activation(x_c + h_c)
                h = o * self.activation(c)
            else:

                x_i = self.conv_activation[index](x_i)
                x_f = self.conv_activation[index](x_f)
                x_c = self.conv_activation[index](x_c)
                x_o = self.conv_activation[index](x_o)
                
                h_i = self.recurrent_activation[index](h_i)
                h_f = self.recurrent_activation[index](h_f)
                h_c = self.recurrent_activation[index](h_c)
                h_o = self.recurrent_activation[index](h_o)
                
            if index==1:
                self.data_format = "channels_last"
        
        self.data_format = None
        return h, [h, c]
    

    def input_conv(self, x, w, b=None, padding='valid'):
        conv_out = K.conv2d(x, w, strides=self.strides,
                            padding=padding,
                            data_format=self.data_format,
                            dilation_rate=self.dilation_rate)
        if b is not None:
            conv_out = K.bias_add(conv_out, b,
                                  data_format=self.data_format)
        return conv_out



    def recurrent_conv(self, x, w):
        conv_out = K.conv2d(x, w, strides=(1, 1),
                            padding='same',
                            data_format=self.data_format)
        return conv_out



    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation[0]),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ConvLSTM1DCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
		
@keras_export('keras.layers.ConvLSTM2D')
class ECLSTM1D(ConvRNN2D):
    def __init__(self,
                 filters = [10,10],
                 kernel_size = [3,3],
                 input_channel = None,  # this ensure the 1D 
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 activation='tanh',
                 recurrent_activation=['linear','hard_sigmoid'],
                 conv_activation = ['hard_sigmoid','hard_sigmoid'],
                 convolutional_type = ["early","early"],
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 return_sequences=False,
                 go_backwards=False,
                 stateful=False,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        """
        How many layers? For each Layer:
            1. number of filter
            2. window size of each filter
            3. activation function for each layer
        
        """

        cell = ConvLSTM1DCell(filters=filters,
                              kernel_size=kernel_size,
                              input_channel=input_channel,
                              strides=strides,
                              padding=padding,
                              data_format=data_format,
                              dilation_rate=dilation_rate,
                              activation=activation,
                              recurrent_activation=recurrent_activation,
                              conv_activation = conv_activation,
                              convolutional_type=convolutional_type,
                              use_bias=use_bias,
                              kernel_initializer=kernel_initializer,
                              recurrent_initializer=recurrent_initializer,
                              bias_initializer=bias_initializer,
                              unit_forget_bias=unit_forget_bias,
                              kernel_regularizer=kernel_regularizer,
                              recurrent_regularizer=recurrent_regularizer,
                              bias_regularizer=bias_regularizer,
                              kernel_constraint=kernel_constraint,
                              recurrent_constraint=recurrent_constraint,
                              bias_constraint=bias_constraint,
                              dropout=dropout,
                              recurrent_dropout=recurrent_dropout,
                              dtype=kwargs.get('dtype'))
        
        super(ECLSTM1D, self).__init__(cell,
                                           return_sequences=return_sequences,
                                           go_backwards=go_backwards,
                                           stateful=stateful,
                                           **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):

        self.cell.reset_dropout_mask()#-------------------------------------------------
        self.cell.reset_recurrent_dropout_mask()#-------------------------------------------------
        return super(ECLSTM1D, self).call(inputs,
                                            mask=mask,
                                            training=training,
                                            initial_state=initial_state)
    
    def get_initial_state(self, inputs):
        """
        INPUTS: shape = (Batch, Time_steps, window_size, numberofSensors, 1 or filters )
        This function will be called after build()
        """

        initial_state = K.zeros_like(inputs)
        # Because of all zeros, sum to delete the timestpes dimension ---->(Batch, Time_steps, window_size, numberofSensors, 1 or filters )
        initial_state = K.sum(initial_state, axis=1)
        # Through the convlution to inference the size of hidden state

        for index, k_shape in enumerate(self.cell.kernel_shape):
            shape = list(k_shape)

            shape[-1] = self.cell.filters[index]

            initial_state = self.cell.input_conv(initial_state,
                                                   array_ops.zeros(tuple(shape)),
                                                   padding=self.cell.padding)

        #self.len_state = initial_state.shape[1]
        if hasattr(self.cell.state_size, '__len__'):
            return [initial_state for _ in self.cell.state_size]
        else:
            return [initial_state]        

    @property
    def filters(self):
        return self.cell.filters

    @property
    def kernel_size(self):
        return self.cell.kernel_size

    @property
    def strides(self):
        return self.cell.strides

    @property
    def padding(self):
        return self.cell.padding

    @property
    def data_format(self):
        return self.cell.data_format

    @property
    def dilation_rate(self):
        return self.cell.dilation_rate

    @property
    def activation(self):
        return self.cell.activation

    @property
    def recurrent_activation(self):
        return self.cell.recurrent_activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def unit_forget_bias(self):
        return self.cell.unit_forget_bias

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'filters': self.filters,
                  'kernel_size': self.kernel_size,
                  'strides': self.strides,
                  'padding': self.padding,
                  'data_format': self.data_format,
                  'dilation_rate': self.dilation_rate,
                  'activation': activations.serialize(self.activation),
                  'recurrent_activation': activations.serialize(
                      self.recurrent_activation[0]),
                  'use_bias': self.use_bias,
                  'kernel_initializer': initializers.serialize(
                      self.kernel_initializer),
                  'recurrent_initializer': initializers.serialize(
                      self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'unit_forget_bias': self.unit_forget_bias,
                  'kernel_regularizer': regularizers.serialize(
                      self.kernel_regularizer),
                  'recurrent_regularizer': regularizers.serialize(
                      self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer': regularizers.serialize(
                      self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(
                      self.kernel_constraint),
                  'recurrent_constraint': constraints.serialize(
                      self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(ECLSTM1D, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        return cls(**config)



def check_the_config_valid(para, window_size,feature):
    initial_state = np.zeros((1,window_size,feature,1))
    initial_state = tf.cast(initial_state, 'float32')
    initial_state = K.zeros_like(initial_state)

    channel = 1
    try:
        for i in range(para["preprocessing_layers"]):

            shape = (para["pre_kernel_width"], 1, channel,para["pre_number_filters"])
            channel = para["pre_number_filters"]
            initial_state = K.conv2d(initial_state, array_ops.zeros(tuple(shape)), (para["pre_strides"],1))#,dilation_rate=(para["pre_dilation_rate"],1))


        for i in range(1,4):
            assert len(para["eclstm_{}_recurrent_activation".format(i)]) == len(para["eclstm_{}_conv_activation".format(i)]) == \
                   len(para["eclstm_{}_number_filters".format(i)]) == len(para["eclstm_{}_kernel_width".format(i)])== \
                   len(para["eclstm_{}_fusion".format(i)]), "Archtecture Parameters of {} layer should be in same length".format(i)

            for j in range(len(para["eclstm_{}_recurrent_activation".format(i)])):

                if para["eclstm_{}_recurrent_activation".format(i)][0] is None:
                    break
                if para["eclstm_{}_fusion".format(i)][j] == "early":
                    shape = (para["eclstm_{}_kernel_width".format(i)][j], feature, channel,para["eclstm_{}_number_filters".format(i)][j] )
                    feature = 1
                    channel =     para["eclstm_{}_number_filters".format(i)][j]
                else :
                    shape = (para["eclstm_{}_kernel_width".format(i)][j], 1, channel,para["eclstm_{}_number_filters".format(i)][j] )
                    channel =     para["eclstm_{}_number_filters".format(i)][j]
                initial_state = K.conv2d(initial_state, array_ops.zeros(tuple(shape)), (para["eclstm_{}_strides".format(i)],1))
        print("valid Configuration!")
        return True
    except:
        print("Invalid Configuration! Try smaller strides or kernel size or greater window size!")
        return False
		
		



def build_the_model(para, seq, window, feature):
    model =Sequential()
    if para["preprocessing_layers"]>=1:
        for i in range(para["preprocessing_layers"]):
            model.add(TimeDistributed(Conv2D(para["pre_number_filters"], (para["pre_kernel_width"], 1), 
                                          strides=(para["pre_strides"],1),
                                          padding="valid", activation=para["pre_activation"])))
    return_sequences = True
    
    for i in range(1,5):
        if para["eclstm_{}_recurrent_activation".format(i)][0] is None:
            break
        if i==4:
            return_sequences = False
        elif para["eclstm_{}_recurrent_activation".format(i+1)][0] is None:
            return_sequences = False


        model.add(ECLSTM1D(filters=para["eclstm_{}_number_filters".format(i)], 
                           kernel_size=para["eclstm_{}_kernel_width".format(i)],
                           recurrent_activation=para["eclstm_{}_recurrent_activation".format(i)],
                           conv_activation = para["eclstm_{}_conv_activation".format(i)],
                           convolutional_type = para["eclstm_{}_fusion".format(i)],
                           strides = (para["eclstm_{}_strides".format(i)],1),
                           input_channel= feature,
                           return_sequences=return_sequences))



        model.add(BatchNormalization())

        if "early" in para["eclstm_{}_fusion".format(i)]:
            feature = 1

    model.add(Flatten())   
    #model.add(tf.compat.v2.keras.layers.Dropout(0.4))
    for i in range(1,5):
        if para["prediction_{}_filters".format(i)]==0:
            break
        model.add(Dense(units = para["prediction_{}_filters".format(i)], 
                        activation = para["prediction_{}_activation".format(i)]))

    model.add(Dense(units = 1, activation = "linear"))

    model.compile(loss='mse', optimizer='Adam')
            
    return model