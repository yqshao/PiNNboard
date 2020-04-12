/**
* Rendering module of PiNNboard
*/

const radi = [0.5, 0.24, 0.28, 0.364, 0.306, 0.384, 0.34, 0.31, 0.304, 0.294, 0.308, 0.454, 0.346, 0.368, 0.42, 0.36, 0.36, 0.35, 0.376, 0.55];
const jmol = ['#ff0000', '#ffffff', '#d9ffff', '#cc80ff', '#c2ff00', '#ffb5b5', '#909090', '#3050f8', '#ff0d0d', '#90e050', '#b3e3f5', '#ab5cf2', '#8aff00', '#bfa6a6', '#f0c8a0', '#ff8000', '#ffff30', '#1ff01f', '#80d1e3', '#8f40d4', '#3dff00', '#e6e6e6', '#bfc2c7'];

/**
 * Draw links in the visible region
 * Trigger a debounced total redraw
 * @param {any} app
 */
function drawLinks() {
  const slowDrawAllLinks = _.debounce(drawAllLinks);
  app.links.filter((link)=>link.visible).forEach(renderLink);
  slowDrawAllLinks(app);
}

/**
 * Draw all the links on the big canvas
 * @param {any} app
 */
function drawAllLinks() {
  app.links.forEach(renderLink);
}


/**
 * Setting up the frames and three.js models
 * @param {any} app
 */
function drawFrames() {
  app.allMeshes = null;
}


/**
 * Build a merged geometry for outlin, atoms, and bonds
 * https://threejsfundamentals.org/threejs/lessons/threejs-optimize-lots-of-objects.html
 * @return {any} outline, atoms, bonds
 */
function makeModels() {
  return [outline, atoms, bonds];
}

/**
 * Set the colors of the geometries
 *
 */
function colorNodes() {
  app.nodes.forEach((i, node) => {
    app.meshes[i].setAttribute('color', node[app.event][app.sample]);
  });
}

/**
 * Render scenes in the visiable region
 */
function renderScenes() {
  app.nodes.filter((node)=>node.visible)
      .forEach(renderNode);
}


/**
 * Update the visible state of the scenes and links
 */
function updateVisible() {
  app.nodes.forEach((node)=>{
    node.visible = false;
  });
  app.links.forEach((link)=>{
    link.visible = link.from.visible || link.to.visible;
  });
}


/**
 * Animate the app
 */
function animate() {
  app.scenes.forEach(
      function(scene, i) {
        scene.userData.controls.update();
      });
  if (cam_changed()) {
    render();
    oldCam = camera.position.clone();
    oldCam.zoom = camera.zoom;
  }
  requestAnimationFrame(animate);
}
