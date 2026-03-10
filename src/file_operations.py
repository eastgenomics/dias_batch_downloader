import logging
import os
from typing import Dict, List, Set


def get_files_in_cwd() -> Set[str]:
    """
    Get the set of file names in the current working directory (non-recursive).

    Returns
    -------
    set
        Set of file names in the current working directory.
    """
    files = {
        entry.name
        for entry in os.scandir()
        if entry.is_file(follow_symlinks=True)
    }
    return files


def check_file_overwrites(
    files_for_download: List[Dict[str, str]], files_before_download: set
) -> List[Dict[str, str]]:
    """
    Checks a list of files intended for download against a set
    of files already present in the output directory. If a file with the same
    name already exists, it is excluded from the download list, and a warning
    is logged. Files that do not exist in the output directory are added to
    the filtered list for download.

    Parameters
    ----------
    files_for_download : list of dicts
        A list of dictionaries, where each dictionary contains metadata about
        a file to be downloaded. Each dictionary must include the keys:
        - "id" (str): The DNAnexus file ID.
        - "name" (str): The name of the file.
    files_before_download : set
        A set of file names that already exist in the output directory.

    Returns
    -------
    list of dict
        A filtered list of file metadata dictionaries, containing only files
        that do not already exist in the output directory. Each dictionary
        includes:
        - "id" (str): The DNAnexus file ID.
        - "name" (str): The name of the file.
    """

    files_for_download_filtered = []
    for file_desc in files_for_download:
        if file_desc["name"] not in files_before_download:
            # File not already in output directory, add to list for download
            files_for_download_filtered.append(
                {"id": file_desc["id"], "name": file_desc["name"]}
            )
        else:
            logging.warning(
                "File with name %s already exists in output directory, "
                "downloading of this file will be skipped to avoid "
                "overwriting.",
                file_desc["name"],
            )

    return files_for_download_filtered


def check_files_after_download(
    files_before_download: Set[str],
    files_for_download: Set[str],
    files_after_download: Set[str],
) -> None:
    """
    Checks whether the expected files are present in the output directory after
    the download process. It compares the files that existed before the
    download, the files intended for download, and the files present after the
    download. If there are discrepancies a warning is logged.

    Parameters
    ----------
    files_before_download : set
        A set of file names that were present in the output directory before
        the download process started.
    files_for_download : set
        A set of file names intended to be downloaded.
    files_after_download : set
        A set of file names that are present in the output directory after
        the download process.

    Returns
    -------
    None
    """
    logging.debug(
        "Checking files in output directory after download. Before download: "
        "%s, files for download: %s, after download: %s",
        files_before_download,
        files_for_download,
        files_after_download,
    )
    expected_files = files_before_download | files_for_download
    if expected_files != files_after_download:
        missing = expected_files ^ files_after_download
        raise RuntimeError(
            "Unexpected number of files in directory after download.\n"
            f"Missing files:\n{missing}"
        )
