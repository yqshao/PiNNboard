export async function render() {
  const style = document.createElement('style');
  style.innerText = `
html,
body,
iframe {
  border: 0;
  height: 100%;
  margin: 0;
  width: 100%;
}`;
  document.head.appendChild(style);

  const iframe = document.createElement('iframe');
  iframe.src = './pinnboard.html';
  document.body.appendChild(iframe);
}
