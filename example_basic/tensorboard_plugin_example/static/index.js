export function render() {
    document.body.innerHTML = `
      <h1>My IFrame Plugin</h1>
      <p>Hello from an embedded HTML string!</p>
      <button id="helloButton">Click Me</button>
    `;

    const button = document.getElementById("helloButton");
    button.onclick = () => alert("Hello!");
  }

// Base URL to your plugin’s API (in TensorBoard, plugin routes are prefixed with `data/plugin/PLUGIN_NAME/`)
const BASE_URL = 'data/plugin/Traning_Control_Plugin';

