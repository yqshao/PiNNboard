import tensorflow as tf
from tensorboard_plugin_pinnboard.metadata import PLUGIN_NAME

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

  
def pinnboard_summary(params,
       display_name=None,
       description=None,
       collections=None):
  """Create a TensorFlow summary op to greet the given guest.

  Arguments:
    params: parameter dictionary of the model
    display_name: If set, will be used as the display name
      in TensorBoard. Defaults to `name`.
    description: A longform readable description of the summary data.
      Markdown is supported.
    collections: Which TensorFlow graph collections to add the summary
      op to. Defaults to `['summaries']`. Can usually be ignored.
  """
  tensors = pinnboard_tensors(params)
  summary_metadata = {k: tf.SummaryMetadata(
      summary_description=description,
      plugin_data=tf.SummaryMetadata.PluginData(
      plugin_name=PLUGIN_NAME)) for k in tensors.keys()}
  
  summary_op = {
    k:tf.summary.tensor_summary(
      'pinnboard.'+k, val,
      summary_metadata=summary_metadata[k], collections=collections)
    for k, val in tensors.items()}
  # Return disctionary of configured summary operations
  return summary_op
