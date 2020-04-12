/*
This is the main app for the PiNNboard visualization

The code reads in formatted tensors defined in PiNN's convention. And
renders a visualization of the atomic neurl network.

By Yunqi Shao (yunqi.shao@kemi.uu.se)
*/

/**
 *  Initialize the app
 */
function init() {
  // Draw the colormap
  const x = d3.scaleLinear().domain([-1, 1]).range([0, 179]);
  const xAxis = d3.axisBottom()
      .scale(x)
      .tickValues([-1.0, -0.5, 0.0, 0.5, 1.0])
      .tickFormat(d3.format('.1f'));
  d3.select('#colormap g.core').append('g')
      .attr('class', 'xaxis')
      .attr('transform', 'translate(0,10)')
      .call(xAxis);
  // Default values
  const defaultData = {
    control: 0,
    runs: ['Not found'],
    event: 0,
    events: [[]],
    n_events: 0,
    sample: 0,
    n_sample: 0,
    prev: Promise.resolve(),
    this_run: null,
  };
  app.data = defaultData;
  // Get list of runts
  app.getRuns();
}

/**
 * Get all runs
 */
function getRuns() {
  app.runs = null;
}

/**
 * Retrieve data from a run, redraw the dom elements and setup the scenes
 */
function selectRun() {
  app.data = requestData(run);
  drawFrames();
  app.update();
}

/**
 * Update event or sample
 */
function update() {
  const atoms = app.atoms[app.event][app.sample];
  if (app.allMeshes[atoms] != app.meshes) {
    app.meshes.forEach((mesh)=>(mesh.visible = false));
    app.meshes = app.allMeshes[atoms];
    app.meshes.forEach((mesh)=>(mesh.visible = true));
  }
  if (app.eventNow != app.event) {
    drawLinks();
  }
  colorNodes();
}


const app = new Vue({
  el: '#sidebar',
  vuetify: new Vuetify({theme: {dark: true}}),
  data: defaultData,
  methods: {
    init: ()=>init(),
    update: ()=> update(),
    getRuns: () => getRuns(),
    selectRun: () => selectRun(),
  },
  watch: {
    sample: app.prev=app.prev.then(app.update),
    event: app.prev=app.prev.then(app.update),
  },
});

app.init();
app.refresh();
animate();
