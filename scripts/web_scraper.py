"""
I want this script to download and unzip a random thing from thingiverse, then find an STL and try to chop
it using default settings. It should work out of a temporary directory, but if any STL fails to chop,
it should move the STL and all of the logs, jsons and yamls to a timestamped folder in 'failed'. This
script should make 100 attempts to chop STLs, then report the success rate.
"""
import os
import tempfile
import numpy as np
import requests
import datetime
import shutil
import traceback
import sys
import logging

from main import run
from pychop3d.configuration import Configuration
from pychop3d import utils

url_template = "https://www.thingiverse.com/download:{}"
MAX_NUMBER = 7_400_000
logger = logging.getLogger(__name__)


def download_stl(directory):
    while True:
        thing_number = np.random.randint(1, MAX_NUMBER)
        url = url_template.format(thing_number)
        logging.info(url)
        req = requests.get(url)
        logging.info(f"request status code: {req.status_code}")
        if req.status_code != 200:
            continue
        logging.info(req.url.split('/')[-1])
        if req.url.split('.')[-1].lower() != 'stl':
            continue
        file_name = os.path.join(directory, f"{thing_number}.stl")
        try:
            content = req.content.decode()
            with open(file_name, 'w') as f:
                f.write(content)
        except UnicodeDecodeError:
            with open(file_name, 'wb') as f:
                f.write(req.content)
        except Exception as e:
            logging.error(e)
            raise e
        return file_name


def dump_error(timestamped_dir):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    traceback.print_exc()
    logging.error(repr(traceback.extract_tb(exc_traceback)))
    logging.error(repr(traceback.format_tb(exc_traceback)))
    with open(os.path.join(timestamped_dir, 'error.txt'), 'w') as f:
        traceback.print_tb(exc_traceback, file=f)
        f.write('\n\n')
        traceback.print_exception(exc_type, exc_value, exc_traceback, file=f)
        f.write('\n\n')
        traceback.print_exc(file=f)
        f.write('\n\n')
        f.write(repr(traceback.extract_tb(exc_traceback)))
        f.write('\n\n')
        f.write(repr(traceback.format_tb(exc_traceback)))


if __name__ == "__main__":
    webscraper_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'web_scraper'))
    if not os.path.exists(webscraper_dir):
        os.mkdir(webscraper_dir)
    log_path = os.path.join(webscraper_dir, "log_output.log")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  [%(levelname)s]  %(message)s",
        handlers=[
            logging.FileHandler(log_path, mode='w'),
            logging.StreamHandler()
        ]
    )
    N_ITERATIONS = 50
    n_failed = 0
    for _ in range(N_ITERATIONS):
        print(f"******************************************************************************************************************")
        print(f"                                 ITERATION: {_}, FAILED: {n_failed}")
        print(f"******************************************************************************************************************")
        # create temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            # create timestamped directory within temporary directory
            date_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            timestamped_dir = os.path.join(tempdir, date_string)
            os.mkdir(timestamped_dir)
            # download STL
            stl_file = download_stl(timestamped_dir)
            name = os.path.splitext(os.path.basename(stl_file))[0]
            # create config
            config = Configuration()
            config.name = name
            config.beam_width = 2
            config.plane_spacing = 30
            config.connector_diameter = 5
            config.connector_spacing = 10
            config.directory = timestamped_dir
            config.mesh = stl_file
            config.part_separation = True
            config.save()
            Configuration.config = config
            try:
                # run
                starter = utils.open_mesh()
                scale_factor = np.ceil(
                    1.1 * config.printer_extents / starter.bounding_box_oriented.primitive.extents).min()
                config.scale_factor = scale_factor
                starter = utils.open_mesh()
                logger.info(f"starter extents: {starter.extents}, starter vertices: {starter.vertices.shape[0]}")
                logger.info(f"running pychop on {name}, scale_factor: {scale_factor}")
                # split into separate components
                if config.part_separation and starter.body_count > 1:
                    starter = utils.separate_starter(starter)
                run(starter)
            # catch failure and move the timestamped directory to 'failed'
            except Exception as e:
                dump_error(timestamped_dir)
                n_failed += 1
                current_dir = os.path.join(webscraper_dir, date_string)
                shutil.move(timestamped_dir, webscraper_dir)
    logging.info(f"{N_ITERATIONS} attempts: {(N_ITERATIONS - n_failed) / N_ITERATIONS} success rate")
