import tensorflow as tf
from tensorboard_plugin_pinnboard.metadata import PLUGIN_NAME
from pinn.networks import PiNet, BPNN


def trace_layer(layer):
  if hasattr(layer, '_traced_io') and layer._traced_io:
    return
  # Wrap the call function to record
  # the input/output tensors
  # this works only in eager mode
  old_call = layer.call
  def new_call(tensors):
    layer._input = tensors
    output = old_call(tensors)
    layer._output = output
    return output
  layer.call = new_call
  layer._traced_io = True

def recursive_trace(target):
  from pinn.io.base import flatten_nested
  if issubclass(type(target), tf.keras.layers.Layer):
    trace_layer(target)
    for k, attr in target.__dict__.items():
      if not k.startswith('_') and k!='layers':
        for item in flatten_nested(attr):
          recursive_trace(item)

def bpnn2summary(bpnn, **kwargs):
  tensors = {
    'elems': bpnn.preprocess._output['elems'],
    'coord': bpnn.preprocess._output['coord'],
    'ind_1': bpnn.preprocess._output['ind_1'],
    'ind_2': bpnn.preprocess._output['ind_2'],
    'diff': bpnn.preprocess._output['diff'],
    'node_p_g0_c0': [],
  }
  ff_layers = bpnn.feed_forward.ff_layers
  max_depth = max([len(ff_layer) for ff_layer in ff_layaers.values()])

  for i, (k, ff_layer) in enumerate(ff_layers.items()):
    tensors[f'node_p_g{i}_c1'] = ff_layer._input
    indices = tf.where(tf.equal(tensors['elems'], k))[:,0]
    for j, dense in ff_layer:
      hidden = dense._output
      hidden = tf.math.unsorted_segment_sum(
        hidden, indices, tf.shape(tensors['ind_1'])[0])
      tensors[f'node_p_g{i}_c{j+2}'] = hidden
      tensors[f'weight_g{i}_c{j+1}_g{i}_c{j+2}'] = dense.kernel
    tensors[f'weight_g{i}_c{j+2}_g0_c{max_depth+2}'] = []
  tensors[f'node_g0_c{max_depth+2}'] = bpnn.feed_forward._ouput
  return tensors


def pinet2summary(pinet, use_pi_gradient=True, **kwargs):
  tensors = {
    'elems': pinet.preprocess._output['elems'],
    'coord': pinet.preprocess._output['coord'],
    'ind_1': pinet.preprocess._output['ind_1'],
    'ind_2': pinet.preprocess._output['ind_2'],
    'diff': pinet.preprocess._output['diff'],
    'node_p_g0_c0': [],
    'node_p_g0_c1': pinet.gc_blocks[0]._input[1]
  }
  prev_node = 'g0_c1'
  count = 1
  n_basis = pinet.basis_fn._output.shape[-1]

  for i, gc_block in enumerate(pinet.gc_blocks):
    # PP layers
    for j, dense in enumerate(gc_block.pp_layer.dense_layers):
      count += 1
      this_node = f'g0_c{count}'
      tensors[f'node_p_{this_node}'] = dense._output
      tensors[f'weight_{prev_node}_{this_node}'] = dense.kernel
      prev_node = this_node
    count += 1
    prev_p_node = prev_node

    # PI layer
    if use_pi_gradient:
      # Go through the PI layer once more to get the
      # dI/dP_i dI/dP_j gradients for visualization
      pi_inp = gc_block.pi_layer._input
      with tf.GradientTape(persistent=True) as gtape:
        gtape.watch(pi_inp[1])
        pi_out = gc_block.pi_layer(pi_inp)
        pi_out = [pi_out[:,j] for j in range(pi_out.shape[-1])]
        dense_inp = gc_block.pi_layer.ff_layer.dense_layers[0]._input
      pi_kernel = tf.stack(
        [tf.reduce_mean(gtape.gradient(pi_out_slice, dense_inp), axis=0)
          for pi_out_slice in pi_out], axis = -1)
      n_in =  int(pi_kernel.shape[0])
      n_out = int(pi_kernel.shape[1])
    else: #just use the kernels from the dense layer
      pi_kernel = gc_block.pi_layer.ff_layer.dense_layers[0].kernel
      for dense in gc_block.pi_layer.ff_layer.dense_layers[1:]:
        pi_kernel = tf.matmul(pi_kernel, dense.kernel)
      n_in =  int(pi_kernel.shape[0])
      n_out = int(pi_kernel.shape[1])//n_basis
      pi_kernel = tf.reshape(pi_kernel, [n_in, n_out, n_basis])
      pi_kernel = tf.norm(pi_kernel, axis=-1)
    # either way we get a [n_prop*2, n_inter] kernel, now we slice them
    pi_kernel_1 = tf.slice(pi_kernel, [0,0], [n_in//2,n_out])
    pi_kernel_2 = tf.slice(pi_kernel, [n_in//2, 0], [n_in//2,n_out])
    this_node = f'g0_c{count}'
    tensors[f'node_i_{this_node}'] = gc_block.pi_layer._output
    tensors[f'weight_{prev_node}_{this_node}_1'] = pi_kernel_1
    tensors[f'weight_{prev_node}_{this_node}_2'] = pi_kernel_2
    count = count +1
    prev_node = this_node

    # II layer
    for dense in gc_block.ii_layer.dense_layers:
      this_node = f'g0_c{count}'
      tensors[f'node_i_{this_node}'] = dense._output
      tensors[f'weight_{prev_node}_{this_node}'] = dense.kernel
      prev_node = this_node
      count += 1
    tensors[f'node_p_{this_node}'] = gc_block.ip_layer._output

    # ResUpdate block
    res_update = pinet.res_update[i]
    this_node = f'g0_c{count}'
    tensors[f'weight_{prev_p_node}_{this_node}_2'] = []
    if isinstance(res_update.transform, tf.keras.layers.Dense):
      tensors[f'weight_{prev_node}_{this_node}_1']= res_update.transform.kernel
    else:
      tensors[f'weight_{prev_node}_{this_node}_1']= []
    tensors['node_p_{}'.format(this_node)] = res_update._output
    prev_node = this_node

    # Output block
    out_group = 0 if len(pinet.out_layers) == 1 else 1
    out_count = count
    prev_out_node = prev_node
    out_layer = pinet.out_layers[i]

    for dense in out_layer.ff_layer.dense_layers:
      out_count += 1
      this_node = f'g{out_group}_c{out_count}'
      tensors[f'node_p_{this_node}'] = dense._output
      tensors[f'weight_{prev_out_node}_{this_node}'] = dense.kernel
      prev_out_node = this_node
    out_count += 1
    this_node = f'g{out_group}_c{out_count}'
    tensors[f'node_p_{this_node}'] = out_layer.out_units._output
    tensors[f'weight_{prev_out_node}_{this_node}'] = out_layer.out_units.kernel
    if i==0:
      last_out_node = this_node
    else:
      tensors[f'weight_{last_out_node}_{this_node}']=[]
  return tensors

def pinnboard_tensors(params):
  if params['network'] != 'pinet':
    raise "Only PiNet supported for now."
  mapping = {
    'elems': 'elems', 'coord': 'coord',
    'ind_1': 'ind_1', 'ind_2': 'ind_2',
    'node_p_g1_c0': 'embed', 'diff': 'diff',
    'node_p_g2_c0':None
  }
  depth = params['network_params']['depth']  
  pp_nodes = len(params['network_params']['pp_nodes'])
  ii_nodes = len(params['network_params']['ii_nodes'])
  pi_nodes = len(params['network_params']['pi_nodes'])
  en_nodes = len(params['network_params']['en_nodes'])
  n_basis = params['network_params']['n_basis']
  count = 0

  prev_node = 'g1_c0'
  for i in range(depth):
    if i>0:
      for j in range(pp_nodes):
        this_node = 'g1_c{}'.format(count+1+j)
        mapping['node_p_{}'.format(this_node)]\
          = 'pp-{}/dense-{}/Tanh'.format(i, j)
        mapping['weight_{}_{}'.format(prev_node, this_node)]\
          ='pp-{}/dense-{}/kernel'.format(i, j)
        prev_node = this_node
      count = count + pp_nodes
    # We only show one interaction for the Pi operation
    # Some transformation is required
    pi_kernel = tf.get_default_graph().get_tensor_by_name('pi-{}/fc_layer/dense-0/kernel:0'.format(i))
    n_in =  int(pi_kernel.shape[0])
    n_out = int(pi_kernel.shape[1])

    if pi_nodes == 1:
      n_out = n_out//n_basis      
      pi_kernel = tf.reshape(pi_kernel, [n_in, n_out, n_basis])
      pi_kernel = tf.norm(pi_kernel, axis=-1)

    pi_kernel_1 = tf.slice(pi_kernel, [0,0], [n_in//2,n_out], name='pinn/pi_{}_1'.format(i))
    pi_kernel_2 = tf.slice(pi_kernel, [n_in//2, 0], [n_in//2,n_out], name='pinn/pi_{}_2'.format(i))
    this_node = 'g1_c{}'.format(count+1)
    mapping['node_i_{}'.format(this_node)] = 'pi-{}/Sum'.format(i)
    mapping['weight_{}_{}_1'.format(prev_node, this_node)] = 'pinn/pi_{}_1'.format(i)
    mapping['weight_{}_{}_2'.format(prev_node, this_node)] = 'pinn/pi_{}_2'.format(i)
    count = count +1

    prev_node = this_node
    for j in range(ii_nodes):
      this_node = 'g1_c{}'.format(count+1+j)
      mapping['node_i_{}'.format(this_node)]\
        = 'ii-{}/dense-{}/Tanh'.format(i,j)
      mapping['weight_{}_{}'.format(prev_node, this_node)]\
        ='ii-{}/dense-{}/kernel'.format(i, j)
      prev_node = this_node        
    
    # For summation after II nodes - updates to the properties
    # is shown in the interaction block
    count += ii_nodes
    this_node = 'g1_c{}'.format(count)
    mapping['node_p_{}'.format(this_node)]\
        = 'ip_{}/UnsortedSegmentSum'.format(i)

    # ResNet block
    prev_node = 'g1_c{}'.format(count-1-ii_nodes)
    count = count + 1
    res_node = 'g1_c{}'.format(count)
    mapping['weight_{}_{}_2'.format(this_node, res_node)]=None    
    if (i!=0 or params['network_params']['pp_nodes'] == [] or
        len(params['network_params']['atom_types']) ==
        params['network_params']['pp_nodes'][0]):
      mapping['weight_{}_{}_1'.format(prev_node, res_node)]=None
    else:
      mapping['weight_{}_{}_1'.format(prev_node, res_node)]='dense/kernel'
    mapping['node_p_{}'.format(res_node)] = 'prop_{}'.format(i)
    prev_node = res_node

    # Output layer
    en_group = 1 if depth==1 else 2
    for j in range(en_nodes):
      this_node = 'g{}_c{}'.format(en_group, count+j+1)
      mapping['node_p_{}'.format(this_node)]\
        = 'en_{}/dense-{}/Tanh'.format(i,j)
      mapping['weight_{}_{}'.format(prev_node, this_node)]\
          ='en_{}/dense-{}/kernel'.format(i, j)
      prev_node = this_node
      
    this_node = 'g{}_c{}'.format(en_group, count+en_nodes+1)
    mapping['node_p_{}'.format(this_node)]\
      = 'en_{}/E_OUT/MatMul'.format(i,j)
    mapping['weight_{}_{}'.format(prev_node, this_node)]\
      ='en_{}/E_OUT/kernel'.format(i, j)
    if i>0:
      mapping['weight_{}_{}'.format(prev_en, this_node)]=None
    # The next PP layer starts from the res node
    prev_node = res_node
    prev_en = this_node

  tensors = {
    k: tf.get_default_graph().get_tensor_by_name(v+':0')
    if v is not None else [[]]
    for k, v in mapping.items()}
  return tensors

def bpnn2summary(bpnn, **kwargs):
  tensors = {
    'elems': bpnn.preprocess._output['elems'],
    'coord': bpnn.preprocess._output['coord'],
    'ind_1': bpnn.preprocess._output['ind_1'],
    'ind_2': bpnn.preprocess._output['ind_2'],
    'diff': bpnn.preprocess._output['diff'],
    'node_p_g0_c0': [],
  }
  ff_layers = bpnn.feed_forward.ff_layers
  max_depth = max([len(ff_layer) for ff_layer in ff_layers.values()])

  for i, (k, ff_layer) in enumerate(ff_layers.items()):
    indices = tf.where(tf.equal(tensors['elems'], k))[:,0]
    tensors[f'node_p_g{i}_c1'] = tf.math.unsorted_segment_sum(
        ff_layer[0]._input, indices, tf.shape(tensors['ind_1'])[0])
    for j, dense in enumerate(ff_layer):
      hidden = dense._output
      hidden = tf.math.unsorted_segment_sum(
        hidden, indices, tf.shape(tensors['ind_1'])[0])
      tensors[f'node_p_g{i}_c{j+2}'] = hidden
      tensors[f'weight_g{i}_c{j+1}_g{i}_c{j+2}'] = dense.kernel
    tensors[f'weight_g{i}_c{j+2}_g0_c{max_depth+2}'] = []
  tensors[f'node_p_g0_c{max_depth+2}'] = bpnn.feed_forward._output
  return tensors

def write_pinnboard_summary(model, sample, step):
  """This works in eager mode so far..."""
  from tensorflow.python.eager import context
  recursive_trace(model)
  with context.eager_mode():
    model(sample)
  if isinstance(model, PiNet):
    summary = pinet2summary(model)
  elif isinstance(model, BPNN):
    summary = bpnn2summary(model)
  else:
    raise NotImplementedError("This model seems unsupported by PiNNBoard")

  metadata = tf.compat.v1.SummaryMetadata(
          summary_description='',
          plugin_data=tf.compat.v1.SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME))

  for k,v in summary.items():
    with tf.summary.experimental.summary_scope(f'pinnboard.{k}', "", [v, step]) as (tag, _):
      tf.summary.write(tag, v, step=step, metadata=metadata)


class PiNNBoardCallback(tf.keras.callbacks.Callback):
    def __init__(self, logdir, sample, freq=10):
      self.writer = tf.summary.create_file_writer(logdir)
      self.freq = freq
      self.step = 0
      if issubclass(type(sample), tf.data.Dataset):
        self.sample = next(iter(sample))
      else:
        self.sample = sample

    def _write_summary(self, step):
      with self.writer.as_default():
        write_pinnboard_summary(self.model, self.sample, step)

    def on_batch_end(self, batch, logs=None):
      if self.step%self.freq == 0:
        self._write_summary(self.step)
      self.step += 1
