import os
import logging
import json
import queue
import threading
import subprocess

from tensorboard.plugins import base_plugin
from werkzeug import wrappers, Response

logger = logging.getLogger(__name__)


base_commands = [
    "python",
    "/home/chenz1/toorange/TBtest/YOLOX/tools/train.py",
    "-f",
    "/home/chenz1/toorange/TBtest/YOLOX/exps/example/custom/my_yolox_gender.py",
    "-d",
    "1",
    "--fp16",
    "-o",
]


class TCPlugin(base_plugin.TBPlugin):
    """
    A demo plugin to:
      - start YOLOX training,
      - stop YOLOX training,
      - update batch size,
      - display training logs (stdout) in the UI.
    """

    plugin_name = "Traning_Control_Plugin"

    def __init__(self, context):
        super(TCPlugin, self).__init__(context)
        self.batch_size = 32
        self.train_process = None  # Will hold subprocess.Popen if training is running
        self.log_queue = queue.Queue()  # Stores training logs
        self.log_thread = None  # Background thread that reads subprocess stdout
        self._stop_log_thread = False

        logger.info("Traning_Control_Plugin is created.")

    def is_active(self):
        return True

    def frontend_metadata(self):
        """
        Tells TensorBoard how to load our plugin front-end (es_module_path).
        We serve index.js at /index.js, but we'll embed everything in that file.
        """
        return base_plugin.FrontendMetadata(es_module_path="/index.js")

    def get_plugin_apps(self):
        """
        Map routes (like /start, /stop, /logs, etc.) to handler methods.
        All routes are under: /data/plugin/Traning_Control_Plugin/
        """
        return {
            "/start": self._start_training,
            "/stop": self._stop_training,
            "/set_batch_size": self._set_batch_size,
            "/logs": self._logs_handler,
            "/index.js": self._index_js_handler,
            "/": self._index_handler,
        }

    # ------------------------------------------------------
    # ROUTE: Serve the front-end HTML
    # ------------------------------------------------------
    @wrappers.Request.application
    def _index_handler(self, request):
        """
        Serves 'index.html' from the 'static' folder.
        """
        html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
        if not os.path.exists(html_path):
            return Response("Index page not found.", status=404)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()
        return Response(html, content_type="text/html")

    # ------------------------------------------------------
    # ROUTE: Serve the dynamic JavaScript (index.js)
    # ------------------------------------------------------
    @wrappers.Request.application
    def _index_js_handler(self, request):
        """
        Returns a JS module that defines `render()`.
        We'll insert our HTML (from index.html) into document.body,
        attach event listeners, and also auto-fetch logs every 5s.
        """
        html_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
        if not os.path.isfile(html_path):
            return Response("index.html not found", status=404)
        with open(html_path, "r", encoding="utf-8") as f:
            html = f.read()

        # Safely embed the HTML as a JS string
        safe_html = json.dumps(html)

        # We'll add a <div id="logs"></div> in the HTML to display logs.
        # If your index.html already has it, skip or adjust.

        js_code = f"""
const BASE_URL = '/data/plugin/{self.plugin_name}';

export function render() {{
  // Insert the static HTML into the body
  document.body.innerHTML = {safe_html};

  // Grab references to buttons/inputs
  const startBtn = document.getElementById('startBtn');
  const stopBtn = document.getElementById('stopBtn');
  const updateBtn = document.getElementById('updateBatchSizeBtn');
  const batchSizeInput = document.getElementById('batchSizeInput');
  const statusMsg = document.getElementById('statusMsg');

  // Add a <div> for logs if you want. (If index.html doesn't have it, create it here.)
  let logsDiv = document.getElementById("logs");
  if (!logsDiv) {{
    logsDiv = document.createElement("div");
    logsDiv.id = "logs";
    logsDiv.style.marginTop = "20px";
    logsDiv.style.padding = "10px";
    logsDiv.style.background = "#272727";
    logsDiv.style.color = "#f1f1f1";
    logsDiv.style.height = "300px";
    logsDiv.style.overflowY = "auto";
    logsDiv.style.whiteSpace = "pre-wrap";
    logsDiv.style.fontFamily = "monospace";
    document.body.appendChild(logsDiv);
  }}

  // 1) START training
  function startTraining() {{
    fetch(`${{BASE_URL}}/start`, {{ method: 'POST' }})
      .then(r => r.text())
      .then(msg => {{
        console.log(msg);
        statusMsg.innerText = 'Training started!';
      }})
      .catch(err => console.error(err));
  }}

  // 2) STOP training
  function stopTraining() {{
    fetch(`${{BASE_URL}}/stop`, {{ method: 'POST' }})
      .then(r => r.text())
      .then(msg => {{
        console.log(msg);
        statusMsg.innerText = 'Training stopped!';
      }})
      .catch(err => console.error(err));
  }}

  // 3) UPDATE batch size
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
    }})
    .catch(err => console.error(err));
  }}

  // 4) POLL logs every 5 seconds
  function fetchLogs() {{
    fetch(`${{BASE_URL}}/logs`)
      .then(r => r.text())
      .then(logText => {{
        // Append new log lines
        logsDiv.textContent += logText;
        // Auto-scroll to bottom
        logsDiv.scrollTop = logsDiv.scrollHeight;
      }})
      .catch(err => console.error("Failed to fetch logs:", err));
  }}
  setInterval(fetchLogs, 3000); // fetch logs every 3s

  // Attach event listeners
  if (startBtn) startBtn.addEventListener('click', startTraining);
  if (stopBtn) stopBtn.addEventListener('click', stopTraining);
  if (updateBtn) updateBtn.addEventListener('click', updateBatchSize);

  console.log("Traning_Control_Plugin rendered!");
}}
"""
        return Response(js_code, content_type="text/javascript")

    # ------------------------------------------------------
    # ROUTE: Start training (subprocess)
    # ------------------------------------------------------
    @wrappers.Request.application
    def _start_training(self, request):
        """
        POST /start
        Spawns a YOLOX training subprocess if not already running.
        """
        if request.method != "POST":
            return Response("Method not allowed", status=405)

        # If already running, don't start again
        if self.train_process and self.train_process.poll() is None:
            return Response("Training is already running!", status=409)

        logger.info("Received START training request.")
        print("Start training (placeholder).")

        # EXAMPLE: build YOLOX command here.
        # If your YOLOX script is "python tools/train.py --batch-size X"
        # you can do something like:
        command = base_commands + [
            "--batch-size",
            str(self.batch_size),
            # Add other YOLOX args as needed, e.g. -f, --devices, etc.
        ]

        # Launch subprocess
        self.train_process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1,
        )

        # Clear old logs
        with self.log_queue.mutex:
            self.log_queue.queue.clear()
        # Start thread to read logs
        self._stop_log_thread = False
        self.log_thread = threading.Thread(target=self._read_process_logs, daemon=True)
        self.log_thread.start()

        return Response(
            f"Command received: Start training with batch size {self.batch_size}.",
            status=200,
        )

    # ------------------------------------------------------
    # ROUTE: Stop training
    # ------------------------------------------------------
    @wrappers.Request.application
    def _stop_training(self, request):
        """
        POST /stop
        Terminates the training subprocess if running.
        """
        if request.method != "POST":
            return Response("Method not allowed", status=405)

        if not self.train_process or self.train_process.poll() is not None:
            return Response("No active training process.", status=200)

        logger.info("Received STOP training request.")
        print("Stop training.")
        self.train_process.terminate()
        self.train_process = None
        self._stop_log_thread = True

        return Response("Command received: Stop training.", status=200)

    # ------------------------------------------------------
    # ROUTE: Set batch size
    # ------------------------------------------------------
    @wrappers.Request.application
    def _set_batch_size(self, request):
        """
        POST /set_batch_size
        Update the plugin's batch_size. Next time we start, we'll use this.
        """
        if request.method != "POST":
            return Response("Only POST request is allowed.", status=405)

        data = request.get_json(silent=True) or {}
        new_bs = data.get("batch_size", self.batch_size)
        try:
            new_bs = int(new_bs)
        except ValueError:
            return Response("Invalid batch size", status=400)

        self.batch_size = new_bs
        logger.info(f"Set batch size to {self.batch_size}.")
        print(f"Set batch size to {self.batch_size}.")
        return Response(f"Set batch size to {self.batch_size}.", status=200)

    # ------------------------------------------------------
    # ROUTE: Logs Handler
    # ------------------------------------------------------
    @wrappers.Request.application
    def _logs_handler(self, request):
        """
        GET /logs
        Returns new lines from the training subprocess (if any).
        We store them in a queue. Once returned here, we remove them from the queue.
        """
        if request.method != "GET":
            return Response("Method not allowed", status=405)

        lines = []
        while not self.log_queue.empty():
            line = self.log_queue.get_nowait()
            lines.append(line)

        return Response("".join(lines), mimetype="text/plain")

    # ------------------------------------------------------
    # Internal: Thread to read process logs
    # ------------------------------------------------------
    def _read_process_logs(self):
        """
        Continuously read from self.train_process.stdout, push lines into self.log_queue.
        Stops if process ends or _stop_log_thread is True.
        """
        if not self.train_process or not self.train_process.stdout:
            return

        try:
            for line in self.train_process.stdout:
                if self._stop_log_thread:
                    break
                self.log_queue.put(line)
        except Exception as e:
            logger.error(f"Error reading process stdout: {e}")
        finally:
            logger.info("Log reading thread exiting.")
            if self.train_process:
                self.train_process.stdout.close()
