"""Utility functions and classes for Patatune.

This module provides various utility functions and classes used throughout the patatune package,
including file management, logging, randomization, and dominance checking.

The import of [numba](https://numba.pydata.org/) and [zarr](https://zarr.readthedocs.io/en/stable/) is optional to allow for flexibility in environments where these
packages may not be installed.

If numba is not installed, a dummy njit decorator is provided that does nothing.

If zarr is not installed, a warning is logged and Zarr functionality is disabled.

A default logger named "patatune" is configured with a custom formatter that adds colors based on log level.
(See [CustomFormatter][patatune.util.CustomFormatter] for details.)

A system-wide exception handler is set up to log uncaught exceptions using the patatune logger.
(See [handle_exception][patatune.util.handle_exception] for details.)
"""

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
    """ Custom logging formatter to add colors based on log level.

    Notes:
        This class customizes the log format by adding color codes for different log levels.
        DEBUG and INFO messages are grey, WARNING messages are yellow, ERROR messages are red,
        and CRITICAL messages are bold red.

        The log is formatted as:

        `%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)`

        For example:

        `2024-01-01 12:00:00,000 - patatune - WARNING - numba package is not installed. The code will run slower. (util.py:206)`
    """
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


Logger = logging.getLogger("patatune")
handler = logging.StreamHandler()
handler.setFormatter(CustomFormatter())
Logger.addHandler(handler)


def handle_exception(exc_type, exc_value, exc_traceback):
    """ Global exception handler to log uncaught exceptions.

    The function logs uncaught exceptions using the default patatune logger.

    Args:
        exc_type (type): The exception type.
        exc_value (Exception): The exception instance.
        exc_traceback (traceback): The traceback object.
    """
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    Logger.error("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))


sys.excepthook = handle_exception


class Randomizer:
    """ Random number generator utility class.

    Implements a class-level random number generator using NumPy's default_rng.
    This class provides a shared random number generator that can be used throughout the package.
    Can be accessed via `Randomizer.rng`.

    For example, to set a seed:
    ```python
    Randomizer.rng = np.random.default_rng(seed=42)
    ```
    """
    rng = np.random.default_rng()


class FileManager:
    """ File management utility class for saving and loading data in various formats.

    This class provides methods to save and load data in CSV, JSON, Zarr, and Pickle formats.

    Attributes:
        saving_enabled (bool): Global flag to enable/disable saving.
        saving_csv_enabled (bool): Flag to enable/disable CSV saving.
        saving_json_enabled (bool): Flag to enable/disable JSON saving.
        saving_zarr_enabled (bool): Flag to enable/disable Zarr saving.
        saving_pickle_enabled (bool): Flag to enable/disable Pickle saving.
        loading_enabled (bool): Global flag to enable/disable loading.
        headers_enabled (bool): Flag to enable/disable headers when saving/loading CSV files.
        working_dir (str): Directory where files will be saved/loaded from.
    """

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
        """ Save a list of lists or 2D array to a CSV file.

        The method saves the data to a CSV file in the `working_dir` path.
        The CSV file is comma-separated and the data is written as floats with 18 decimal places and dot as decimal separator.
        The method creates the necessary directories if they do not exist.
        If headers are provided and `headers_enabled` is True, they will be written as the first row.

        Args:
            csv_list (list[list] | np.ndarray): Data to be saved.
            filename (str): Name of the output CSV file.
            headers (list[str], optional): Column headers for the CSV file.
        """
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
        """ Load data from a CSV file.

        The method loads data from a CSV file in the `working_dir` path.
        If `headers_enabled` is True, it assumes the first row contains headers.

        Args:
            filename (str): Name of the input CSV file.

        Returns:
            (tuple): A tuple containing the data as a NumPy array of floats and the headers (if any) as a NumPy array of strings.
        """
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
    def save_json(cls, dictionary, filename):
        """ Save a dictionary to a JSON file.

        The method saves a dictionary to a JSON file in the `working_dir` path.
        It creates the necessary directories if they do not exist.

        Args:
            dictionary (dict): Dictionary to be saved.
            filename (str): Name of the output JSON file.
        """
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
        """ Load a dictionary from a JSON file.

        The method loads a dictionary from a JSON file in the `working_dir` path.

        Args:
            filename (str): Name of the input JSON file.

        Returns:
            (dict): The loaded dictionary.
        """
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug("Loading from '%s'", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        with open(full_path, encoding='utf-8') as f:
            return json.load(f)

    @classmethod
    def save_pickle(cls, obj, filename):
        """ Save an object to a Pickle file.

        The method saves an object to a Pickle file in the `working_dir` path.
        It creates the necessary directories if they do not exist.

        Args:
            obj (any): The object to be saved.
            filename (str): Name of the output Pickle file.
        """

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
        """ Load an object from a Pickle file.

        The method loads an object from a Pickle file in the `working_dir` path.
        
        Args:
            filename (str): Name of the input Pickle file.

        Returns:
            (any): The loaded object.
        """
        full_path = os.path.join(cls.working_dir, filename)
        Logger.debug("Loading from '%s'", full_path)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"The file '{full_path}' does not exist.")
        with open(full_path, 'rb') as f:
            return pickle.load(f)
        
    @classmethod
    def save_zarr(cls, obj, filename, **kwargs):
        """ Save a dictionary of arrays to a Zarr file.

        The method saves a dictionary of arrays to a Zarr file in the `working_dir` path.
        It creates the necessary directories if they do not exist.
        The keys of the dictionary are used as group names in the Zarr file.
        If a key is an integer, it is prefixed with "iteration_" to form the group name.
        Additional attributes can be added to the root group via `kwargs`.

        Args:
            obj (dict): The dictionary of arrays to be saved.
            filename (str): Name of the output Zarr file.
            **kwargs (): Additional attributes to be saved in the root group as key-value pairs where the key is the attribute name and the value is the attribute value.

        """
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
    """ Determine which particles are dominated within a population.

    A particle is considered dominated if there exists at least one other particle that is better or equal in all objectives
    and strictly better in at least one objective.
    
    Args:
        particles (np.ndarray):
            2-D array of objective values for each particle (shape: [n_particles, n_objectives]).
        pareto_length (int):
            Number of particles considered part of the current Pareto set (these are skipped in comparisons).

    Returns:
        (np.ndarray): Boolean array of length `len(particles)` where True means the particle is dominated by at least one other particle.

    Notes:
        The function is decorated with a (possible) `njit` to allow optional numba acceleration.
    """
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
