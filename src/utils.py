import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional

import dxpy


def traverse_dict(d: Dict, path: List) -> Optional[Any]:
    """
    Traverse a dictionary using a list of keys as a path.

    Iterates through the provided list of keys, accessing nested values in the
    dictionary. If any key in the path is not found, returns None.

    Parameters
    ----------
    d : dict
        The dictionary to traverse.
    path : list
        A list of keys representing the path to traverse through the dict.

    Returns
    -------
    any or None
        The value at the end of the path, or None if the path cannot be fully
        traversed.
    """
    for key in path:
        d = d.get(key)
        # If specified path cannot be fully traversed, return None
        if d is None:
            break
    return d


def call_in_parallel(
    func: Callable,
    items: List,
    max_workers: int,
    ignore_missing: bool = True,
    ignore_all_errors: bool = True,
    **kwargs,
) -> List:
    """
    Calls the given function in parallel using concurrent.futures on
    the given set of items (i.e for calling dxpy.describe() on multiple
    object IDs).

    Additional arguments specified to kwargs are directly passed to the
    specified function.

    Parameters
    ----------
    func : callable
        Function to call on each item
    items : list
        List of items to call function on
    max_workers : int
        Number of parallel threads in thread pool executor
    ignore_missing : bool
        Controls if to just print a warning instead of raising an
        exception on a dxpy.exceptions.ResourceNotFound being raised.
        This is most likely from a file that has been deleted and we are
        just going to default to ignoring these
    ignore_all_errors : bool
        Controls if to just print a warning instead of raising an exception.

    Returns
    -------
    list
        List of responses
    """
    results = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        concurrent_jobs = {
            executor.submit(func, item, **kwargs): item for item in items
        }

        for future in as_completed(concurrent_jobs):
            # access returned output as each is returned in any order
            try:
                results.append(future.result())
            except Exception as exc:
                if ignore_missing and isinstance(
                    exc, dxpy.exceptions.ResourceNotFound
                ):
                    # dx object does not exist and specifying to skip,
                    # just print warning and continue
                    logging.warning(
                        "%s could not be found, skipping to not raise an "
                        "exception",
                        concurrent_jobs[future],
                    )
                    continue
                # catch any other errors that might get raised during querying
                logging.warning(
                    "Error getting data for %s: %s",
                    concurrent_jobs[future],
                    exc,
                )

                if ignore_all_errors is False:
                    raise

    return results


def filter_dict_by_keys(d: dict, exclude_keys: set) -> dict:
    """
    Filter a dictionary by excluding specified keys.

    Parameters
    ----------
    d : dict
        The input dictionary to filter.
    exclude_keys : set
        A set of keys to exclude from the input dictionary.

    Returns
    -------
    dict
        A new dictionary containing only the key-value pairs from the input
        dictionary where the keys are not in the exclude_keys set.
    """
    return {k: v for k, v in d.items() if k not in exclude_keys}
