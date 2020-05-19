import json
import logging
import os
import random
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Tuple

from flask import Flask, Response, request, current_app
from werkzeug.datastructures import FileStorage

from model_server import process_request

# setting up logging
logging.basicConfig(
    format="[%(asctime)s %(levelname)s] %(message)s",
    datefmt="%d-%m-%Y %H:%M:%S",
    stream=sys.stdout,
    level=logging.DEBUG
)
logger = logging.getLogger()
sys.excepthook = lambda tp, val, tb: logger.error("Unhandled exception:", exc_info=val)

# creating flask app
app = Flask(__name__, root_path=os.getcwd())

# loading configs
app.config["temp"] = Path(tempfile.gettempdir()) / "project-model-server"
with open("config.json", "r") as f:
    app.config.update(json.load(f))


def stream_res(file_dir: Path, filename: str, delete_dir: bool = True) -> bytes:
    """
    Function to stream file to client and optionally delete the directory
    which contains the file.
    Args:
        file_dir (Path): The path to the directory containing the file.
        filename (str): The name of the file .
        delete_dir (bool, optional): If True, deletes the directory. Defaults
        to True.

    Returns:
        bytes: The contents of the file.
    """

    with open(str(file_dir / filename), "rb") as fh:
        yield from fh

    if delete_dir:
        # deleting the directory containing the file
        shutil.rmtree(file_dir)

    app.logger.info("Response sent.")


@app.route('/api/upload', methods=["POST"])
def upload() -> Response:
    app.logger.info("Received request.")

    file_dir: Path = app.config["temp"] / str(random.randint(0, 1000))
    Path.mkdir(file_dir, parents=True)

    # saving the files received
    item: Tuple[str, FileStorage]
    for item in request.files.items():
        item[1].save(str(file_dir / f"{item[0]}.nii.gz"))
        app.logger.info("Received %s.nii.gz.", item[0])

    res_dir, res_file = process_request(file_dir, **request.form)

    # creating response
    response: Response = current_app.response_class(
        stream_res(res_dir, res_file), mimetype="application/gzip"
    )
    response.headers.set("Content-Disposition", "attachment", filename=res_file)
    app.logger.info("Sending response...")

    # returning response
    return response


if __name__ == '__main__':
    app.run()
