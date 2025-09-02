import os
import sys
import json
import logging
import dill as pickle
import numpy as np

try:
    import zarr
    from zarr import ZipStore
    zarr_available = True
except ImportError:
    logging.warning("zarr >=2,<3 not installed. Zarr functionality will be disabled.")
    zarr_available = False

# If numba is installed import it and use njit decorator otherwise use a dummy decorator
try:
    from numba import njit
except ImportError:
    logging.warning("numba package is not installed. The code will run slower.")
    def njit(f=None, *args, **kwargs):
        def dummy_decorator(func):
            return func

        if callable(f):
            return f
        else:
            return dummy_decorator



class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    string_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: grey + string_format + reset,
        logging.INFO: grey + string_format + reset,
        logging.WARNING: yellow + string_format + reset,
        logging.ERROR: red + string_format + reset,
        logging.CRITICAL: bold_red + string_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


Logger = logging.getLogger("Optimizer")
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
Logger.addHandler(handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    Logger.error("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


class Randomizer:
    rng = np.random.default_rng()


class FileManager:
    saving_enabled = True
    saving_csv_enabled = True
    saving_json_enabled = True
    saving_zarr_enabled = False
    saving_pickle_enabled = True
    loading_enabled = False
    headers_enabled = False
    working_dir = "tmp"

    @classmethod
    def save_csv(cls, csv_list, filename="file.csv", headers=None):
        if not cls.saving_enabled or not cls.saving_csv_enabled:
            Logger.debug("Saving csv is disabled.")
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            Logger.debug("Creating folder '%s'", folder)
            os.makedirs(folder)
        Logger.debug("Saving to '%s'", full_path)
        arr = np.array(csv_list, dtype=float)
        if cls.headers_enabled and headers is not None:
            with open(full_path, 'w') as f:
                f.write(','.join(headers) + '\n')
                np.savetxt(f, arr, fmt='%.18f', delimiter=',')
        else:
            np.savetxt(full_path, arr, fmt='%.18f', delimiter=',')

    @classmethod
    def load_csv(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug("Loading from '%s'", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        if cls.headers_enabled:
            # If headers are enabled, we assume the first row is the header
            data = np.genfromtxt(full_path, delimiter=',', dtype=float, skip_header=1)
            headers = np.genfromtxt(full_path, delimiter=',', dtype=str, max_rows=1)
            return data, headers
        else:
            return np.genfromtxt(full_path, delimiter=',', dtype=float), None

    @classmethod
    def save_json(cls, dictionary, filename="file.json"):
        if not cls.saving_enabled or not cls.saving_json_enabled:
            Logger.debug("Saving json is disabled.")
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        Logger.debug("Saving to '%s'", full_path)
        with open(full_path, 'w', encoding='utf-8') as f:
            json.dump(dictionary, f, indent=4)

    @classmethod
    def load_json(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug("Loading from '%s'", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        with open(full_path, encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def save_pickle(cls, obj, filename):
        if not cls.saving_enabled or not cls.saving_pickle_enabled:
            Logger.debug("Saving pickle is disabled.")
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        Logger.debug("Saving to '%s'", full_path)
        with open(full_path, 'wb') as f:
            pickle.dump(obj, f, recurse=True)

    @classmethod
    def load_pickle(cls, filename):
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug("Loading from '%s'", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        with open(full_path, 'rb') as f:
            return pickle.load(f)
        
    @classmethod
    def save_zarr(cls, obj, filename, **kwargs):
        if not cls.saving_enabled or not cls.saving_zarr_enabled:
            Logger.debug("Saving Zarr is disabled.")
            return
        if not zarr_available:
            Logger.warning("zarr package is not installed. Skipping Zarr saving.")
            return
        full_path = os.path.join(cls.working_dir, filename)
        folder = os.path.dirname(full_path)
        if not os.path.exists(folder):
            os.makedirs(folder)
        Logger.debug("Saving to '%s'", full_path)
        
        store = ZipStore(full_path, mode='w')
        root_group = zarr.group(store=store)
        
        for key, value in obj.items():
            if isinstance(key, int):
                group_name = f"iteration_{key}"
            else:
                group_name = key
            
            group = root_group.create_group(group_name)
            group.create_dataset("data", data=value, overwrite=True)
        root_group.attrs.update(kwargs)
                
        store.close()

@njit
def get_dominated(particles, pareto_length):
    dominated_particles = np.full(len(particles), False, dtype=np.bool_)
    for i, pi in enumerate(particles):
        for j, pj in enumerate(particles):
            if (i < pareto_length and j < pareto_length) or i == j:
                continue
            if np.any(pi > pj) and \
                    np.all(pi >= pj):
                dominated_particles[i] = True
                break
    return dominated_particles.astype(np.bool_)
