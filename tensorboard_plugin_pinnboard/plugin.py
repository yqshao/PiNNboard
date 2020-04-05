from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os
import tensorflow as tf
import numpy as np
import six
from tensorboard.backend.http_util import Respond
from tensorboard.plugins import base_plugin
from tensorboard.util import tensor_util
import werkzeug
from werkzeug import wrappers

from tensorboard_plugin_pinnboard import metadata


class PiNNboard(base_plugin.TBPlugin):
  plugin_name = metadata.PLUGIN_NAME

  def __init__(self, context):
    self._multiplexer = context.multiplexer

  def is_active(self):
    return bool(self._multiplexer.PluginRunToTagToContent(metadata.PLUGIN_NAME))

  def get_plugin_apps(self):
    return {
        "/index.js": self._serve_js,
        "/core.js": self._core_js,
        "/pinnboard.html": self._serve_html,      
        "/runs": self._serve_runs,
        "/data": self._serve_data,
    }

  def frontend_metadata(self):
    return base_plugin.FrontendMetadata(es_module_path="/index.js")

  @wrappers.Request.application
  def _serve_js(self, request):
    del request  # unused
    filepath = os.path.join(os.path.dirname(__file__), "static", "index.js")
    with open(filepath) as infile:
      contents = infile.read()
    return werkzeug.Response(contents, content_type="application/javascript")

  @wrappers.Request.application
  def _core_js(self, request):
    del request  # unused
    filepath = os.path.join(os.path.dirname(__file__), "static", "pinnboard-core.js")
    with open(filepath) as infile:
      contents = infile.read()
    return werkzeug.Response(contents, content_type="application/javascript")  

  @wrappers.Request.application
  def _serve_html(self, request):
    del request  # unused
    filepath = os.path.join(os.path.dirname(__file__), "static", "pinnboard.html")
    with open(filepath) as infile:
      contents = infile.read()
    return werkzeug.Response(contents, content_type="text/html")

  @wrappers.Request.application
  def _serve_runs(self, request):
    mapping = self._multiplexer.PluginRunToTagToContent(metadata.PLUGIN_NAME)
    result = {run: {} for run in self._multiplexer.Runs()}
    for (run, tag_to_content) in six.iteritems(mapping):
      tensor_events = self._multiplexer.Tensors(run, 'pinnboard.ind_1')
      ind_1 = tf.make_ndarray(tensor_events[0].tensor_proto)
      result[run]['n_events'] = len(tensor_events)
      result[run]['n_sample'] = int(np.max(ind_1) + 1)
    contents = json.dumps(result, sort_keys=True)
    return Respond(request, contents, "application/json")

  @wrappers.Request.application
  def _serve_data(self, request):
    run = request.args.get("run")
    event = int(request.args.get("event"))
    sample = int(request.args.get("sample"))
    
    if run is None or event is None or sample is None:
      raise werkzeug.exceptions.BadRequest("Must specify run and tag")

    this_run = self._multiplexer.PluginRunToTagToContent(
      metadata.PLUGIN_NAME)[run]

    data = {
      tag.split('.')[1]: tf.make_ndarray(
        self._multiplexer.Tensors(run, tag)[event].tensor_proto)
      for tag in this_run.keys()
    }

    this_ind_1 = np.equal(data['ind_1'][:,0], sample)
    this_ind_2 = this_ind_1[data['ind_2'][:,0]]

    for key in data.keys():
      if data[key].shape[0] == 0:
        continue
      if key in ['elems', 'coord'] or key.startswith('node_p'):
        data[key] = data[key][this_ind_1]
      if key in ['diff', 'ind_2'] or key.startswith('node_i'):
        data[key] = data[key][this_ind_2]
      if key.startswith('node'):
        data[key] = np.nan_to_num(data[key]/np.abs(data[key]).max())
        data[key] = data[key].T
      if key.startswith('weight'):
        if data[key].shape[1]!=0:
          data[key] = data[key]/np.abs(data[key]).max()
      
    data['ind_2'] -= data['ind_2'].min()
    data['coord'] -= (data['coord'].min(axis=0)+data['coord'].max(axis=0))/2
    del data['ind_1']
    data = {k: v.round(2).astype(str).astype(float).tolist()
            if k not in ['ind_1', 'ind_2', 'elems'] else v.tolist()
            for k,v in data.items()}

    contents = json.dumps(data)
    return Respond(request, contents, "application/json")
