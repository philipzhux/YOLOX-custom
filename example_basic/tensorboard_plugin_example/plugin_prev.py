import logging, os, six, json
from tensorboard.plugins import base_plugin
from tensorboard.util import tensor_util
from werkzeug import wrappers

import werkzeug

logger = logging.getLogger(__name__)

# class metadata:
#     PLUGIN_NAME = "Traning_Control_Plugin"


class TCPlugin(base_plugin.TBPlugin):
    """
    A demo plugin to start, pause, stop traning
    and adjust the batch size in runtime.
    """

    plugin_name = "Traning_Control_Plugin"

    def __init__(self, context):
        super(TCPlugin, self).__init__(context)
        # default batch size, TODO: change it
        self.batch_size = 32
        self._multiplexer = context.multiplexer
        logger.info("Traning_Control_Plugin is created.")

    def get_plugin_apps(self):
        return {
            "/start": self._start_traning,
            "/stop": self._stop_traning,
            "/set_batch_size": self._set_batch_size,
            "/": self._index_handler,
            "/index.js": self._index_js_handler,
        }

    @wrappers.Request.application
    def _index_js_handler(self, request):
        # return a js with render function
        # to server the index.html
        # document.body.innerHTML = `[HTML]`;
        with open(
            os.path.join(os.path.dirname(__file__), "static", "index.html"),
            "r",
            encoding="utf-8",
        ) as f:
            html = f.read()
        js = f"""
        const BASE_URL = '/data/plugin/{self.plugin_name}';

        export function render() {{
        // 1. Insert HTML
        document.body.innerHTML = {json.dumps(html)};

        // 2. Grab references to buttons/inputs
        const startBtn = document.getElementById('startBtn');
        const stopBtn = document.getElementById('stopBtn');
        const updateBtn = document.getElementById('updateBatchSizeBtn');
        const batchSizeInput = document.getElementById('batchSizeInput');
        const statusMsg = document.getElementById('statusMsg');

        // 3. Define the functions here or inline

        function startTraining() {{
            fetch(`${{BASE_URL}}/start`, {{ method: 'POST' }})
            .then(r => r.text())
            .then(msg => {{
                console.log(msg);
                statusMsg.innerText = 'Training started!';
            }});
        }}

        function stopTraining() {{
            fetch(`${{BASE_URL}}/stop`, {{ method: 'POST' }})
            .then(r => r.text())
            .then(msg => {{
                console.log(msg);
                statusMsg.innerText = 'Training stopped!';
            }});
        }}

        function updateBatchSize() {{
            const newSize = batchSizeInput.value;
            fetch(`${{BASE_URL}}/set_batch_size`, {{
            method: 'POST',
            headers: {{ 'Content-Type': 'application/json' }},
            body: JSON.stringify({{ batch_size: newSize }})
            }})
            .then(r => r.text())
            .then(msg => {{
            console.log(msg);
            statusMsg.innerText = `Batch size updated to ${{newSize}}`;
            }});
        }}

        // 4. Attach event listeners
        if (startBtn) startBtn.addEventListener('click', startTraining);
        if (stopBtn) stopBtn.addEventListener('click', stopTraining);
        if (updateBtn) updateBtn.addEventListener('click', updateBatchSize);
        }}"""

        # js = js.replace("[HTML]", html)
        return werkzeug.Response(js, content_type="text/javascript")

    def is_active(self):
        return True

    @wrappers.Request.application
    def _start_traning(self, request):
        # placeholder, TODO: add control flow
        # to the pytorch training loop
        logger.info("Start traning.")
        print("Start traning.")

        return werkzeug.Response("Command received: Start traning.", status=200)

    # @wrappers.Request.application
    # def _pause_traning(self, request):
    #     # placeholder, TODO: add control flow
    #     # to the pytorch training loop
    #     logger.info("Pause traning.")
    #     print("Pause traning.")
    #     return wrappers.response("Command received: Pause traning.", status=200)

    @wrappers.Request.application
    def _stop_traning(self, request):
        # placeholder, TODO: add control flow
        # to the pytorch training loop
        logger.info("Stop traning.")
        print("Stop traning.")
        return werkzeug.Response("Command received: Stop traning.", status=200)

    @wrappers.Request.application
    def _set_batch_size(self, request):
        # placeholder, TODO: add control flow
        # to the pytorch training loop
        # only allow post request
        if request.method != "POST":
            return wrappers.response("Only POST request is allowed.", status=405)
        # get the new batch size from request
        data = request.get_json()
        bs = data.get("batch_size", self.batch_size)
        print(f"Set batch size to {bs}.")  # TODO: change the pytorch loop
        logger.info(f"Set batch size to {bs}.")
        return werkzeug.Response(f"Set batch size to {bs}.", status=200)

    def frontend_metadata(self):
        return base_plugin.FrontendMetadata(es_module_path="/index.js")

    @wrappers.Request.application
    def _index_handler(self, request):
        html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")

        if not os.path.exists(html_path):
            return werkzeug.Response("Index page not found.", status=404)

        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        return werkzeug.Response(html, content_type="text/html")

    """
    tensorboard --logdir=/home/chenz1/toorange/TBtest/YOLOX/YOLOX_outputs/my_yolox_gender/tensorboard --port=6066
    """




