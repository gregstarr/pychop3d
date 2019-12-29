"""
I want this script to download and unzip a random thing from thingiverse, then find an STL and try to chop
it using default settings. It should work out of a temporary directory, but if any STL fails to chop,
it should move the STL and all of the logs, jsons and yamls to a timestamped folder in 'failed'. This
script should make 100 attempts to chop STLs, then report the success rate.

TODO:
    - memory leak
"""
import os
import tempfile
import numpy as np
import requests
import datetime
import shutil
import traceback
import sys

from pychop3d import run
from pychop3d.configuration import Configuration

url_template = "https://www.thingiverse.com/download:{}"
MAX_NUMBER = 7_400_000


def download_stl(directory):
    while True:
        thing_number = np.random.randint(1, MAX_NUMBER)
        url = url_template.format(thing_number)
        print(url)
        req = requests.get(url)
        print(req.status_code)
        if req.status_code != 200:
            continue
        print(req.url.split('/')[-1])
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
            print(e)
            raise e
        return file_name


def dump_error(timestamped_dir):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    traceback.print_tb(exc_traceback)
    traceback.print_exception(exc_type, exc_value, exc_traceback)
    traceback.print_exc()
    print(repr(traceback.extract_tb(exc_traceback)))
    print(repr(traceback.format_tb(exc_traceback)))
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
    N_ITERATIONS = 20
    n_failed = 0
    for _ in range(N_ITERATIONS):
        print(f"*********************************************************")
        print(f"           ITERATION: {_}, FAILED: {n_failed}            ")
        print(f"*********************************************************")
        # create temporary directory
        with tempfile.TemporaryDirectory() as tempdir:
            # create timestamped directory within temporary directory
            date_string = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            timestamped_dir = os.path.join(tempdir, date_string)
            os.mkdir(timestamped_dir)
            # download STL
            stl_file = download_stl(timestamped_dir)
            # create config
            config = Configuration()
            config.beam_width = 2
            config.plane_spacing = 30
            config.directory = timestamped_dir
            config.mesh = stl_file
            Configuration.config = config
            # run
            try:
                run.run()
            # catch failure and move the timestamped directory to 'failed'
            except Exception as e:
                dump_error(timestamped_dir)
                n_failed += 1
                failed_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'failed', date_string))
                shutil.move(timestamped_dir, failed_directory)
                config.directory = failed_directory
                config.save()
    print(f"{N_ITERATIONS} attempts: {(N_ITERATIONS - n_failed) / N_ITERATIONS} success rate")
