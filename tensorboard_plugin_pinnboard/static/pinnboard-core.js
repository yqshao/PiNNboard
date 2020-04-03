/*
This is the main app for the PiNNboard visualization

The code reads in formatted tensors defined in PiNN's convention. And
renders a visualization of the atomic neurl network.

By Yunqi Shao (yunqi.shao@kemi.uu.se)
*/
var app = new Vue({
  el: '#sidebar',
  vuetify: new Vuetify({theme:{dark: true}}),
  data: {
    control: 0,
    runs: ["Not found"],
    event: 0,
    events: {},
    n_events: 100,
    sample: 0,
    n_sample: 10,
    prev: Promise.resolve(),
    this_run: "Select run"
  },
  methods: {
    refresh: function() {getRuns(this)},
    selectRun: function (run) {
      this.this_run = run;
      this.n_sample = this.runinfo[run].n_sample-1;
      this.n_events = this.runinfo[run].n_events-1;
      this.event = 0;
      this.events = {};
      this.sample = 0;
      this.getData();
    },
    getData: function() {
      getData(this);
    },
    async update (){
      if (this.events[this.event]){
        const evt = this.event;
        this.prev = this.prev.then(()=>{this.slowRedraw(this.events[evt])})
      }else{
        const url = `./data?run=${this.this_run}&event=${this.event}&sample=${this.sample}`;
        const evt = this.event;
        response = fetch(url).then(res => res.text());
        this.prev = Promise.all([response, this.prev])
          .then(response => {this.events[evt] = response[0], this.slowRedraw(response[0])})
      }
    },
    async redraw (response){
      data = JSON.parse(await response);
      atoms_tmp = {coord: data.coord, elems: data.elems,
                   diff: data.diff, ind_2: data.ind_2}
      tensors = data;
      tmp = readTensors(tensors);
      layers = tmp[0];
      weights = tmp[1];
      if (JSON.stringify(atoms_tmp) !== JSON.stringify(atoms))
      {atoms=atoms_tmp;
       drawLayers(layers, frames, atoms)}
      // Set the colors
      weights.map((weight, i) => setLinkColors(weight.val, links[i]));
      layers.map(layer => setLayerColor(layer, frames))
    }
  },
    watch:{
      sample: function(){this.events={}; this.slowUpdate();},
      event: function(){this.slowUpdate()}      
    },
    created: function () {
      this.slowUpdate = _.throttle(this.update, 50);
      this.slowRedraw = _.throttle(this.redraw, 50)
    },
})


const radi = [0.5, 0.24, 0.28, 0.364, 0.306, 0.384, 0.34, 0.31, 0.304, 0.294, 0.308, 0.454, 0.346, 0.368, 0.42, 0.36, 0.36, 0.35, 0.376, 0.55];
const jmol = ["#ff0000", "#ffffff", "#d9ffff", "#cc80ff", "#c2ff00", "#ffb5b5", "#909090", "#3050f8", "#ff0d0d", "#90e050", "#b3e3f5", "#ab5cf2", "#8aff00", "#bfa6a6", "#f0c8a0", "#ff8000", "#ffff30", "#1ff01f", "#80d1e3", "#8f40d4", "#3dff00", "#e6e6e6", "#bfc2c7"];
const myscale = d3.scaleLinear().domain([-1, 0, 1])
  .range(["#f59322", "#ffffff", "#0877bd"])
  .clamp(true);

var scenes = [];
var canvas = document.getElementById("pinn-canvas");
var content = document.getElementById("pinn-content");

var renderer = new THREE.WebGLRenderer({
  canvas: canvas,
  antialias: true,
  alpha: true,
});
var svgcanvas;
var links;
var frames;
var camera = new THREE.OrthographicCamera(-4, 4, 4, -4, 1, 1000);
camera.position.z = 8;

// Draw the colormap
let x = d3.scaleLinear().domain([-1, 1]).range([0, 179]);
let xAxis = d3.axisBottom()
    .scale(x)
    .tickValues([-1.0, -0.5, 0.0, 0.5, 1.0])
    .tickFormat(d3.format(".1f"));
d3.select("#colormap g.core").append("g")
    .attr("class", "xaxis")
    .attr("transform", "translate(0,10)")
    .call(xAxis);

app.refresh()


function getData(app){
  const Http = new XMLHttpRequest();
  const url = `./data?run=${app.this_run}&event=${app.event}&sample=${app.sample}`;  
  Http.open("GET", url);
  Http.send();
  Http.onreadystatechange = (e) => {
    if (Http.readyState == 4 && Http.status == 200) {
      data = JSON.parse(Http.responseText);
      atoms = {coord: data.coord, elems: data.elems,
	       diff: data.diff, ind_2: data.ind_2}
      tensors = data;
      initialize(atoms,tensors);
      scenes.forEach(scene => {
        var controls = new THREE.OrthographicTrackballControls(
        		scene.userData.camera, scene.userData.element);
  			controls.zoomSpeed = 0.01;
  			controls.noPan = true;
        scene.userData.controls = controls;
  })
      animate();
    }}
}


function getRuns(app){
  const Http = new XMLHttpRequest();
  const url = './runs';  
  Http.open("GET", url);
  Http.send();
  Http.onreadystatechange = (e) => {
    if (Http.readyState == 4 && Http.status == 200) {
      app.runinfo = JSON.parse(Http.responseText);
      app.runs = Object.keys(app.runinfo);
      if (app.runs) {app.selectRun(app.runs[0])};}}
}

function readTensors(tensors) {
  const isNode = key => key.startsWith('node');
  const isWeight = key => key.startsWith('weight');

  var keys = Object.keys(tensors);
  var nodes = keys.filter(isNode).map(k =>
    ({
      type: k.split('_')[1],
      g: k.split('_')[2].slice(1),
      c: k.split('_')[3].slice(1),
      val: tensors[k]
    })
  );
  var weights = keys.filter(isWeight).map(k =>
    ({
      from: {
        g: k.split('_')[1].slice(1),
        c: k.split('_')[2].slice(1)
      },
      to: {
        g: k.split('_')[3].slice(1),
        c: k.split('_')[4].slice(1)
      },
      val: tensors[k]
    })
  );

  return [nodes, weights]
}


/*
 The frames drawer needs to be aware of the overall topology of the
 network. Frames and links only need to be redrawn when a new model
 is loaded.
*/

function drawFrames(layers) {
  var max_g = Math.max(...layers.map(layer => layer.g)) + 1;
  var max_c = Math.max(...layers.map(layer => layer.c)) + 1;

  var canvas_w = (max_c * 150 + 300).toString()+"px";
  var g_h = Array(max_g).fill(0);
  layers.forEach((layer) => g_h[layer.g] = Math.max(g_h[layer.g],layer.val.length));
  var canvas_h = g_h.map(x => 20+x*106).reduce((a,b) => a+b, 0) + 100;  

  ["app", "pinn-canvas", "pinn-links", "pinn-content"].forEach(
    dom => {
      document.getElementById(dom).style.width = canvas_w;
      document.getElementById(dom).style.height = canvas_h;      
    }
  )  

  while (content.firstChild) {
    content.removeChild(content.firstChild);
  }

  d3.selectAll('#pinn-links *').remove();
  svgcanvas = d3.select("#pinn-links");
  scenes.forEach(scene  => clearThree(scene));
  frames = [];
  for (g = 0; g < max_g; g++) {
      group = document.createElement("div");
      group.clasName="pinn-group"
    content.appendChild(group);
    this_g = [];
    this_g.dom = group;
    for (c = 0; c < max_c; c++) {
      column = document.createElement("div");
      column.className = "pinn-layer";
      group.appendChild(column);
      this_c = [];
      this_c.dom = column;
      this_c.inputs = 0;
      this_g.push(this_c)
    }
    frames.push(this_g)
  }
  layers.forEach(function(layer) {
    if (frames[layer.g][layer.c].dom.hasChildNodes()) {
      return
    }
    const n_node = Math.max(layer.val.length, 1);
    for (i=0; i<n_node; i++) {
      node = document.createElement("div");
      node.className = "pinn-node";
      frames[layer.g][layer.c].dom.appendChild(node);
      var scene = new THREE.Scene();
      scene.userData.camera = camera;
      scene.userData.element = node;
      frames[layer.g][layer.c].push({
        dom: node,
        scene: scene
      });
      scenes.push(scene);
    };
  });

  return frames
}


/*
  Atoms should be redrawn when the atoms data is updated.
*/

function drawLayers(layers, frames, atoms) {
  var tmp = makeAtoms(atoms);
  atoms = tmp[0];
  bonds = tmp[1];
  outline = tmp[2];

  // Draw atoms
  layers.forEach(layer =>
    frames[layer.g][layer.c].forEach(
      frame => drawAtoms(frame, atoms, outline)))

  // Draw bonds
  layers.forEach(function(layer) {
    if (layer.type == 'i') {
      frames[layer.g][layer.c].forEach(
        frame => drawBonds(frame, bonds))
    }
  })
}


function clearThree(obj) {
  while (obj.children.length > 0) {
    clearThree(obj.children[0])
    obj.remove(obj.children[0])}
  if (obj.geometry) obj.geometry.dispose()
  if (obj.material) obj.material.dispose()
}


function drawAtoms(frame, atoms, outline) {
  clearThree(frame.scene);
  frame.atoms = atoms.clone();
  frame.atoms.traverse((node) => {
    if (node.isMesh) {
      node.material = node.material.clone()
    }
  });
  frame.outline = outline.clone();
  frame.scene.add(frame.atoms);
  frame.scene.add(frame.outline);
}

function drawBonds(frame, bonds) {
  frame.bonds = bonds.clone();
  frame.bonds.traverse((node) => {
    if (node.isMesh) {
      node.material = node.material.clone()
    }
  });
  frame.scene.add(frame.bonds);
}

function makeAtom(elem, coord, group, out_group) {
  var geo = new THREE.SphereGeometry(radi[elem] * 1.2, 12, 12);
  var mat = new THREE.MeshBasicMaterial({
    color: jmol[elem]
  });
  var ball = new THREE.Mesh(geo, mat);
  var out_geo = new THREE.SphereGeometry(radi[elem] * 1.2 + 0.1, 12, 12);
  var out_mat = new THREE.MeshBasicMaterial({
    color: 0x000000,
    side: THREE.BackSide
  });
  var out_ball = new THREE.Mesh(out_geo, out_mat);
  ball.position.set(...coord);
  out_ball.position.set(...coord);
  group.add(ball);
  out_group.add(out_ball);
}

function makeBond(coord, diff, group) {
  var coord = new THREE.Vector3(...coord);
  var diff = new THREE.Vector3(...diff);
  var HALF_PI = Math.PI * 0.5;
  var distance = diff.length();
  var position = coord.add(diff.divideScalar(4));
  var geo = new THREE.CylinderGeometry(0.15, 0.15, distance / 2, 6, 1, false);
  var mat = new THREE.MeshBasicMaterial();
  var orientation = new THREE.Matrix4();
  var offsetRotation = new THREE.Matrix4();
  var offsetPosition = new THREE.Matrix4();
  orientation.lookAt(new THREE.Vector3(0, 0, 0), diff, new THREE.Vector3(0, 1, 0));
  offsetRotation.makeRotationX(HALF_PI);
  orientation.multiply(offsetRotation);
  geo.applyMatrix(orientation)
  var bond = new THREE.Mesh(geo, mat);
  bond.position.set(position.x, position.y, position.z);
  group.add(bond);
}

function makeAtoms(atoms) {
  var atom_group = new THREE.Group();
  var bond_group = new THREE.Group();
  var out_group = new THREE.Group();
  // Returns two groups, atoms and bonds
  atoms.elems.map((elem, i) => makeAtom(elem, atoms.coord[i], atom_group, out_group))
  atoms.ind_2.map((ind, i) => makeBond(atoms.coord[ind[0]], atoms.diff[i],
				       bond_group))

  atom_group.matrixAutoUpdate = false;
  bond_group.matrixAutoUpdate = false;
  out_group.matrixAutoUpdate = false;
  return [atom_group, bond_group, out_group];
}

/*
  Setting colors, they are called when the tensors are updated.
*/

function setLayerColor(layer) {
  if (layer.type == 'p') {
    frames[layer.g][layer.c].map(
      (node, i) => setAtomColors(node.atoms, layer.val[i]))
  } else {
    frames[layer.g][layer.c].map(
      (node, i) => setBondColors(node.bonds, layer.val[i]))
  }
}

function setAtomColors(obj, val) {
  if (!val){return};
  obj.children.forEach(
    function(child, idx) {
      child.material.color.set(myscale(val[idx]));
      child.colorNeedUpdate = true;
    })
}

function setBondColors(obj, val) {
  obj.children.forEach(
    function(child, idx) {
    	if (Math.abs(val[idx])<0.3){
      	child.visible=false;
      	return;
      }
      else{
      	child.visible=true;
      }
      child.material.color.set(myscale(val[idx]));
      child.colorNeedUpdate = true;
    })
}

function render() {
  updateSize();
  canvas.style.transform = `translateY(${window.scrollY}px)`;
  renderer.setClearColor(0xffffff, 0);
  renderer.setScissorTest(false);
  renderer.clear();
  renderer.setClearColor(0xffffff);
  renderer.setScissorTest(true);
  scenes.forEach(
    function(scene, i) {
      var element = scene.userData.element;
      var rect = element.getBoundingClientRect();

      if (rect.bottom < 0 || rect.top > renderer.domElement.clientHeight ||
        rect.right < 0 || rect.left > renderer.domElement.clientWidth) {
        return
      }

      var width = rect.right - rect.left - 4;
      var height = rect.bottom - rect.top - 4;
      var left = rect.left - canvas.getBoundingClientRect().left + 3;
      var bottom = canvas.getBoundingClientRect().bottom - rect.bottom + 2;
      var camera = scene.userData.camera;
      renderer.setViewport(left, bottom, width, height);
      renderer.setScissor(left, bottom, width, height);
      scene.userData.controls.update();
      renderer.render(scene, camera)
    });
}


function updateSize() {
  var width = canvas.clientWidth;
  var height = canvas.clientHeight;
  renderer.setSize(width, height, false);
  if (canvas.width !== width || canvas.height !== height) {
    renderer.setSize(width, height, false);
  }
}


function setLinkColors(weights, links) {
  wdim1 = weights.length;
  if (!wdim1) {
    links.forEach(
            (link) => link.attr("stroke",'#999999'))
  } else {
    wdim2 = weights[0].length;
    weights.forEach((wi, i) =>
      wi.forEach((wij, j) => links[i][j].attr("stroke", myscale(wij))))
  }
}

function drawLinks(weights, froms, tos) {
  fromdim = froms.length;
  todim = tos.length;
  wdim1 = weights.val.length;

  len = 106 / (tos.inputs + 1);
  offset = (weights.order) * len;
  if (!wdim1) {
    links = froms.map(
      (from, i) => drawLink(froms[i], tos[i], offset, false))
  } else {
    wdim2 = weights.val[0].length;
    links = weights.val.map((wi, i) =>
      wi.map((wij, j) => drawLink(froms[i], tos[j],
        offset + (i / (wdim1 - 1.) - 0.5) / 5 * len)))
  }
  return links
}

function drawLink(from, to, offset = 0, dashed = true) {
  box1 = from.dom.getBoundingClientRect();
  box2 = to.dom.getBoundingClientRect();
  canvasleft = canvas.getBoundingClientRect().left + 3;

  var data = {
    source: {
      x: box1.left + 103 - canvasleft,
      y: box1.bottom - 53
    },
    target: {
      x: box2.left - canvasleft + 3,
      y: box2.bottom - 106 + offset
    }
  }
  var link = d3.linkHorizontal().x(d => d.x).y(d => d.y);
  var line = svgcanvas.append("path").attr("fill", "none")
    .attr("class", "links")
    .attr("d", link(data)).attr('stroke-width', '2px');
  if (dashed) {
    line.style("stroke-dasharray", ("10, 2"))
  }
  return line
}

function initialize(atoms, tenors) {
  tmp = readTensors(tensors);
  layers = tmp[0];
  weights = tmp[1];
  // Draw the Frames and Links
  frames = drawFrames(layers);
  weights.forEach(weight => {
    weight.order = frames[weight.to.g][weight.to.c].inputs += 1
  });
  links = weights.map(weight => drawLinks(weight,
    frames[weight.from.g][weight.from.c], frames[weight.to.g][weight.to.c]));

  // Draw the Atoms and bonds
  drawLayers(layers, frames, atoms);
  // Set the colors
  weights.map((weight, i) => setLinkColors(weight.val, links[i]));
  layers.map(layer => setLayerColor(layer, frames));
}

function animate() {
  render();
  setTimeout(() => requestAnimationFrame(animate), 1000 / 60);
}


