export async function render() {
  const iframe = document.createElement('iframe');
  iframe.src = './pinnboard.html';
  Object.assign(iframe.style, {
    border: 0,
    height: '100%',
    width: '100%',
  });
  document.body.appendChild(iframe);
}
