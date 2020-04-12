/**
 * Parse the input tensors
 * @param {any} tensors
 *
 *
 */
const myscale = d3.scaleLinear().domain([-1, 0, 1])
    .range(['#f59322', '#ffffff', '#0877bd'])
    .clamp(true);

/**
 * Read the tenors from JSON response
 * compute the colors
 * @param {any} tensors
 * @return {any} [nodes, weights, atoms]
 */
function readTensors(tensors) {
  return [nodes, weights, atoms];
}
