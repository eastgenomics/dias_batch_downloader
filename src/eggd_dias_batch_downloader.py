"""Download files associated with an eggd_dias_batch job."""
import argparse
import importlib.util
import json
import logging
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from logging.handlers import RotatingFileHandler
from pathlib import Path, PureWindowsPath
from typing import Any, Callable, Optional

import dxpy


def parse_args() -> argparse.Namespace:
    """
    Parse arguments given at cmd line.

    Returns:
        - args (Namespace): object containing parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Download files associated with an eggd_dias_batch job"
    )

    parser.add_argument(
        "-c", "--config-file",
        help="Path to config file",
        type=Path,
        required=True
    )

    parser.add_argument(
        "-b", "--batch_job_id",
        help="eggd_dias_batch job ID that was used to create the reports you "
        " wish to download", required=True
    )

    # Mutually exclusive options for creating or finding an output directory
    # if neither are specified, files will be downloaded to the current working
    # directory
    output_dir_group = parser.add_mutually_exclusive_group()
    output_dir_group.add_argument(
        "--make-output-dir",
        action="store_true",
        help=(
            "If --dry-run is not set, create a new output directory named "
            "using the dx project in which the batch job was run and change "
            "into it before downloading files. If --dry-run is set, simulate "
            "creating the directory and include the path that would be made in"
            " the summary. Cannot be used with --find-output-dir."
        )
    )
    output_dir_group.add_argument(
        "--find-output-dir",
        action="store_true",
        help=(
            "Find an existing output directory for this batch job ID and "
            "change into it before downloading files. Cannot be used with "
            "--make-output-dir."
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help=(
            "If used with --find-output-dir, find and change into output "
            "directory, but do not download files. If used with "
            "--make-output-dir, do not make output directory or download "
            " files, but return the download location that would be used in "
            "the summary text."
        )
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, existing files will be overwritten during download. If "
        "not set, files with names that already exist in the output directory "
        "will not be downloaded."
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)"
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to the log file where logs will be written"
    )


    args = parser.parse_args()

    return args


def configure_logging(
    log_level: str,
    log_file: Optional[Path]
) -> None:
    """
    Configure logging for the script.

    Parameters
    ----------
    log_level : str
        Logging level (e.g., DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file : Path, optional
        Path to the log file. If not provided, logs will be printed to stdout.

    Returns
    -------
    None
    """
    # By default, log format does not include timestamp for easier reading in
    # stdout
    log_format = "%(levelname)s - %(message)s"
    handlers = []

    # Add a file handler with rotation if log_file is provided
    if log_file:
        log_format = "%(asctime)s - %(levelname)s - %(message)s"
        # Configure log to have a max size of 1MB and keep 1 backup file before
        # overwriting
        file_handler = RotatingFileHandler(
            log_file, mode='a', maxBytes=1024 * 1024, backupCount=1
        )
        file_handler.setFormatter(logging.Formatter(log_format))
        handlers.append(file_handler)

    # Else, print logs to stdout
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(log_format))
    handlers.append(console_handler)

    logging.basicConfig(level=log_level, handlers=handlers)
    # Suppress noisy logging from urllib3 called by dxpy when making API calls
    logging.getLogger("urllib3.util.retry").setLevel(logging.WARNING)


def load_config(config_path: Path) -> dict:
    """
    Load the configuration file

    Parameters
    ----------
    config_path : Path
        Path to the Python configuration file

    Returns
    -------
    dict
        Configuration dictionary loaded from the specified file
    """
    spec = importlib.util.spec_from_file_location(
        config_path.name.replace(".py", ""), config_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[config_path.name] = module
    spec.loader.exec_module(module)
    return module.CONFIG


def describe_batch_job(
    batch_job_id: str,
    fields: dict,
    exec_regex: str
) -> dict:
    """
    Describe a DNAnexus batch job and return the description dictionary.

    Parameters
    ----------
    batch_job_id : str
        The ID of the batch job to describe.
    fields : dict
        A dictionary specifying which fields to include in the description.
    exec_regex : str
        Regular expression to match the executable name of the batch job.

    Raises
    ------
    ValueError
        If the specified job ID does not correspond to an eggd_dias_batch job.

    Returns
    -------
    dict
        A dictionary containing the description of the specified batch job,
        including only the specified fields.
    """
    desc_dict = dxpy.bindings.DXJob(dxid=batch_job_id).describe(fields=fields)
    if not re.search(exec_regex, desc_dict["executableName"]):
        raise ValueError(
            f"The specified job ID {batch_job_id} does not correspond to an "
            "eggd_dias_batch job."
        )

    return desc_dict


def get_files_in_cwd() -> set:
    """
    Get the set of file names in the current working directory (non-recursive).

    Returns
    -------
    set
        Set of file names in the current working directory.
    """
    files = {
        entry.name for entry in os.scandir() if entry.is_file(
            follow_symlinks=True)
    }
    return files


def configure_output_directory(
    make_output_dir: bool,
    find_output_dir: bool,
    dry_run: bool,
    project_id: str,
    assay: str,
    output_config: dict
) -> str:
    """
    Configure the output directory for downloading files.

    Depending on the provided options, this function either creates a new
    output directory, navigates to an existing one, or defaults to the
    current working directory.

    See arg definitions for details on the behavior of each option.

    Parameters
    ----------
    make_output_dir : bool
    find_output_dir : bool
    dry_run : bool
    project_id : str
        DNAnexus project ID used to derive the output directory name.
    assay : str
        Assay type used to determine the folder path in the configuration.
    output_config : dict
        Configuration dictionary containing folder paths for assays.

    Returns
    -------
    str
        str to the configured output directory.
    """
    if not make_output_dir and not find_output_dir:
        cwd = os.getcwd()
        logging.debug("No output directory options specified, downloading "
                      "files to current working directory: %s", cwd)
        return cwd

    # Get output directory from DNAnexus project name
    project_name = dxpy.describe(project_id)["name"]
    run_name = re.sub(r'^[0-9]+_', '', project_name)
    run_name = re.sub(r'_(?:37|38)_', '_', run_name)

    # Ensure name does not contain problematic characters for file paths
    if not re.fullmatch(r"[A-Za-z0-9_]+", run_name):
        raise ValueError(
            "Project name contains characters not allowed in file paths "
            f"{project_name}"
        )

    output_dir = os.path.join(output_config["folder_paths"][assay], run_name)

    if make_output_dir and not dry_run:
        os.mkdir(output_dir)
        logging.debug("Created output directory: %s", output_dir)

    if find_output_dir or (make_output_dir and not dry_run):
        os.chdir(output_dir)
        logging.debug("Changed into output directory: %s", output_dir)

    return output_dir


def traverse_dict(d: dict, path: list) -> Optional[Any]:
    """Traverse a dictionary using a list of keys as a path."""
    for key in path:
        d = d.get(key)
        # If specified path cannot be fully traversed, return None
        if d is None:
            break
    return d


def get_batch_job_file_ids(desc_dict: dict, query_config: dict) -> dict:
    """
    For each file type (key) in the config dict, extract file IDs from a dx
    describe dictionary describing a eggd_dias_batch job, using paths defined
    in the config.

    If a file ID can be found, it is added to the result dict under the
    corresponding file type key; otherwise, a warning is printed.

    Parameters
    ----------
    desc_dict : dict
        DNAnexus batch job description dictionary.
    query_config : dict
        Config dict specifying keys and paths to extract file IDs.

    Returns
    -------
    dict
        Dict containing file IDs found in the batch job metadata, keyed by the
        file types in the config. If a file ID is not found, it is omitted and
        a warning is printed.
    """

    dx_ids = {}
    for k, config in query_config.items():
        # QC status, multiqc report are optional input for batch so
        # may not be present, so check for their presence and print warnings
        # if not
        dx_id = traverse_dict(desc_dict, config["desc_path"])
        if dx_id is None:
            logging.warning(
                "%s could not be found in eggd_dias_batch job metadata", k
            )
        else:
            dx_ids.setdefault(k, []).append(dx_id)
    return dx_ids


def call_in_parallel(
    func: Callable,
    items: list,
    max_workers: int,
    ignore_missing: bool = True,
    ignore_all_errors: bool = True,
    **kwargs
) -> list:
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

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        concurrent_jobs = {
            executor.submit(func, item, **kwargs): item for item in items
        }

        for future in as_completed(concurrent_jobs):
            # access returned output as each is returned in any order
            try:
                results.append(future.result())
            except Exception as exc:
                if (
                    ignore_missing and
                    isinstance(exc, dxpy.exceptions.ResourceNotFound)
                ):
                    # dx object does not exist and specifying to skip,
                    # just print warning and continue
                    logging.warning(
                        "%s could not be found, skipping to not raise an "
                        "exception",
                        concurrent_jobs[future]
                    )
                    continue
                # catch any other errors that might get raised during querying
                logging.warning(
                    "Error getting data for %s: %s",
                    concurrent_jobs[future], exc
                )

                if ignore_all_errors is False:
                    raise

    return results


def get_launched_job_file_ids(
    desc_dicts: list[dict],
    query_config: dict
) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """
    Extracts output file IDs from dx describe dictionies describing the jobs
    launched by eggd_dias_batch.

    For each job description, matches the job name using the regex in
    query_config. If matched, traverses the describe dictionary to extract the
    file IDs using the specified path. If configured, also extracts the sample
    name from the job name, used for counting samples with multiple runs
    of the same job type.

    Collects file IDs by type and logs jobs where output files could not be
    found.

    Parameters
    ----------
    desc_dicts : list of dict
        List of job description dictionaries.
    query_config : dict
        Config dict specifying regex and path for each file type.

    Returns
    -------
    tuple[dict[str, list[str]], dict[str, dict[str, int]]]
        Dictionary mapping file type keys to lists of file IDs and a dictionary
        mapping sample names to executable counts.
    """
    dx_ids = {}
    sample_exec_counts = {}
    for desc in desc_dicts:
        for exec_type, exec_config in query_config.items():
            # Not all launched job have files to extract, so check regex
            # matches before trying to extract files
            if re.search(exec_config["exec_regex"], desc["name"]):

                # Extract sample name if regex provided in config, else skip
                sample_name_regex = exec_config.get("sample_name_regex")
                if sample_name_regex:
                    sample_name_match = re.search(
                        sample_name_regex,
                        desc["name"]
                    )
                    if sample_name_match:
                        # Count number of times each sample has a launched job
                        # for each exec type
                        sample_name = sample_name_match.group(1)
                        sample_exec_counts.setdefault(sample_name, {})
                        sample_exec_counts[sample_name].setdefault(
                            exec_type, 0)
                        sample_exec_counts[sample_name][exec_type] += 1
                    else:
                        logging.warning(
                            "Sample name could not be extracted from "
                            "job name: %s (%s)",
                            desc["name"], desc["id"]
                        )

                # Extract output file IDs using paths specified in config
                for file_type, desc_path in exec_config["desc_paths"].items():
                    output_file_id = traverse_dict(desc, desc_path)
                    if output_file_id is None:
                        logging.warning(
                            "Output file for %s (%s) could not be found",
                            desc["name"], desc["id"]
                        )
                    else:
                        dx_ids.setdefault(file_type, []).append(output_file_id)
    return dx_ids, sample_exec_counts


def get_details(file_id: str, project_id: str) -> tuple[str, dict]:
    """
    Get details of a file in DNAnexus

    Parameters
    ----------
    file_id : str
        DNAnexus file ID.
    project_id : str
        DNANexus project ID.

    Returns
    -------
    tuple[str, dict]
        Tuple containing (DNAnexus file ID, corresponding file details).
    """
    return file_id, dxpy.DXFile(dxid=file_id, project=project_id).get_details()


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
    return {
        k: v for k, v in d.items() if k not in exclude_keys
    }


def filter_report_files(
    reports_details: list[tuple[str, dict]],
    details_key: str
) -> dict:
    """
    Summarises and filters SNV/CNV report files by variant content.

    Prints the number of reports that contain variants, do not contain
    variants, or lack file details. Returns a list of file IDs for reports
    that should be downloaded (i.e., those with variants).

    Parameters
    ----------
    reports_details : list[tuple[str, dict]]
        List of tuples: (file ID of the report, file details dict for the
        corresponding report).
    details_key : str
        Key in the file details dict providing the number of variants in the
        report.

    Returns
    -------
    dict
        Dictionary summarising SNV/CNV report files by variant content.
    """
    filtered_report_dict = {
        "dx_ids_for_download": [],
        "n_reports": len(reports_details),
    }

    for report_id, details in reports_details:
        # If old report and no "file details" metadata is available then
        # n_vars will = None
        n_vars = details.get(details_key)
        if isinstance(n_vars, int):
            if n_vars > 0:
                filtered_report_dict["dx_ids_for_download"].append(report_id)
        else:
            logging.warning(
                "The file details for %s could not be interpreted.", report_id
            )
    return filtered_report_dict


def linux_to_windows_path(
    linux_path: str,
    linux_prefix: str,
    windows_prefix: str
) -> PureWindowsPath:
    """
    Convert a Linux file path to a Windows file path.

    This function removes the specified Linux/server prefix from the given
    path, replaces it with the specified Windows prefix, and converts the
    resulting path to a Windows-compatible format (i.e. uses backslashes).

    Parameters
    ----------
    linux_path : str
        The Linux file path to be converted.
    linux_prefix : str
        The prefix in the Linux path to be replaced.
    windows_prefix : str
        The prefix to replace the Linux prefix in the resulting Windows path.

    Returns
    -------
    PureWindowsPath
        The converted Windows-compatible file path.
    """
    # Remove linux prefix
    relative_path = linux_path.removeprefix(linux_prefix).lstrip("/")

    # Add windows prefix and convert to PureWindowsPath
    return PureWindowsPath(windows_prefix) / relative_path


def gather_and_summarise_non_report_files(
    non_report_files: dict,
    summary_text_buffer: list[str],
    file_ids_for_download: list[str],
    indent: str = "\t"
) -> None:
    """
    This function iterates over the non-report files, appends a summary of the
    number/type of files being downloaded to the provided summary text buffer,
    and adds their file IDs to the list of file IDs for download.

    Parameters
    ----------
    non_report_files : dict
        Dictionary where keys are file types and values are lists of file IDs.
    summary_text_buffer : list
        List of strings to which the summary of non-report files will be
        appended.
    file_ids_for_download : list
        List of files IDs to be downloaded
    indent : str, optional
        String used for indentation in the summary text (default is a tab).

    Returns
    -------
    None
    """
    for file_type, dx_ids in non_report_files.items():
        summary_text_buffer.append(
            f"{indent}{len(dx_ids)} {file_type} file(s)\n"
        )
        file_ids_for_download.extend(dx_ids)
    summary_text_buffer.append("\n")


def gather_and_summarise_report_files(
    filtered_reports: dict,
    summary_text_buffer: list[str],
    file_ids_for_download: list[str],
    indent: str = "\t"
) -> None:
    """
    This function iterates over the filtered reports, appends a summary of
    the proportion of reports downloaded vs not to the specified summary text
    buffer, and adds their file IDs to the list of file IDs for download.

    Parameters
    ----------
    filtered_reports : dict
        Dictionary where keys are report types and values are dictionaries
        containing the total number of reports (i.e. those with and without
        variants) and the list of file IDs for reports with variants to be
        downloaded.
    summary_text_buffer : list
        List of strings to which the summary of report files will be appended.
    file_ids_for_download : list
        List of files to be downloaded.
    indent : str, optional
        String used for indentation in the summary text (default is a tab).

    Returns
    -------
    None
    """
    for k, v in filtered_reports.items():
        n_file_ids_for_download = len(v["dx_ids_for_download"])
        n_reports = v["n_reports"]
        summary_text_buffer.append(
            f"{indent}{n_file_ids_for_download} out of {n_reports} {k} "
            f"report(s) ({n_reports - n_file_ids_for_download} skipped)\n"
        )
        file_ids_for_download.extend(v["dx_ids_for_download"])
    summary_text_buffer.append("\n")


def summarise_sample_exec_counts(
    sample_exec_counts: dict[str, dict[str, int]],
    summary_text_buffer: list[str],
    indent: str = "\t"
) -> None:
    """
    This function iterates over the sample execution counts, summarises the
    number of samples with multiple launched jobs, and appends the summary
    to the provided summary text buffer.

    Parameters
    ----------
    sample_exec_counts : dict
        Dictionary mapping sample names to a nested dictionary of executable
        types and their counts.
    summary_text_buffer : list
        List of strings to which the summary of sample execution counts will
        be appended.
    indent : str, optional
        String used for indentation in the summary text (default is a tab).

    Returns
    -------
    None
    """
    logging.debug("Samples with launched jobs: %s", sample_exec_counts)
    # Number of samples with launched jobs
    summary_text_buffer.append(
        f"Total samples: {len(sample_exec_counts)}\n"
    )
    sample_report_count_summaries = []
    for sample, exec_counts in sample_exec_counts.items():
        sample_report_count_summary = ""
        for exec_type, count in exec_counts.items():
            # Check if a sample had multiple runs of an executable, if so
            # generate summary
            if count > 1:
                sample_report_count_summary += (
                    f"{indent*2}{count} {exec_type}\n"
                )
        # Once looped through exec types for a sample, if any had multiple runs
        # append their summaries to a list
        if sample_report_count_summary:
            sample_report_count_summaries.append(
                f"{indent}{sample}:\n{sample_report_count_summary}"
            )
    # If there are summaries in summary list, add summaries to summary text,
    # else add N/A
    if sample_report_count_summaries:
        summary_text_buffer.append("\nSamples with multiple launched jobs:\n")
        summary_text_buffer.append("".join(sample_report_count_summaries))
    else:
        summary_text_buffer.append(
            "\nSamples with multiple launched jobs: N/A"
        )
    summary_text_buffer.append("\n")


def check_file_overwrites(
    files_for_download: list[dict],
    files_before_download: set
) -> list[dict[str, str]]:
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
                {
                    "id": file_desc["id"],
                    "name": file_desc["name"]
                }
            )
        else:
            logging.warning(
                "File with name %s already exists in output directory, "
                "downloading of this file will be skipped to avoid "
                "overwriting.", file_desc["name"]
            )

    return files_for_download_filtered


def check_files_after_download(
    files_before_download: set,
    files_for_download: set,
    files_after_download: set
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
        files_before_download, files_for_download, files_after_download
    )
    expected_files = files_before_download | files_for_download
    if expected_files != files_after_download:
        missing = expected_files ^ files_after_download
        logging.warning(
            "Unexpected number of files in directory after download.\n"
            "Missing files:\n %s", missing
        )


def download_single_file(file_dict: dict, project: str) -> None:
    """
    Download a dx file using the specified file id and filename in the context
    of the specified project.

    Parameters
    ----------
    file_dict : dict
        dictionary containing file ID and name
    project : str
        project containing the file

    Returns
    -------
    None
    """
    dxpy.bindings.dxfile_functions.download_dxfile(
        dxid=file_dict["id"],
        filename=file_dict["name"],
        project=project
    )


def main():
    args = parse_args()
    configure_logging(log_level=args.log_level.upper(), log_file=args.log_file)

    config = load_config(args.config_file)
    logging.debug("Config %s loaded:\n %s", args.config_file, json.dumps(
        config, indent=2))

    logging.debug("Reading batch job (%s) metadata...", args.batch_job_id)
    batch_job_dx_desc = describe_batch_job(
        batch_job_id=args.batch_job_id,
        fields=config["batch_job_query"]["dx_desc_fields"],
        exec_regex=config["batch_job_query"]["exec_regex"]
    )

    project_id = batch_job_dx_desc["project"]
    output_config = config["output_config"]
    output_dir = configure_output_directory(
        make_output_dir=args.make_output_dir,
        find_output_dir=args.find_output_dir,
        dry_run=args.dry_run,
        project_id=project_id,
        assay=batch_job_dx_desc["input"]["assay"],
        output_config=output_config
    )

    batch_job_file_ids = get_batch_job_file_ids(
        desc_dict=batch_job_dx_desc,
        query_config=config["batch_job_query"]["files"]
    )

    launched_jobs = batch_job_dx_desc["output"]["launched_jobs"].split(",")
    logging.debug("DX describing launched jobs: %s", launched_jobs)
    max_workers = config["max_workers"]
    launched_job_desc_dicts = call_in_parallel(
        func=dxpy.describe,
        items=launched_jobs,
        fields=config["launched_job_query"]["dx_desc_fields"],
        max_workers=max_workers
    )

    logging.debug("Gathering launched job output file IDs...")
    launched_job_file_ids, sample_exec_counts = get_launched_job_file_ids(
        desc_dicts=launched_job_desc_dicts,
        query_config=config["launched_job_query"]["execs"]
    )

    logging.debug("Filtering variant reports...")
    filtered_reports = {}
    for report_type, rep_config in config["filter_reports"].items():
        # Batch may not have launched CNV reports workflows (e.g. for TWE runs)
        # therefore check report file IDs have been retrieved before trying to
        # get their details
        report_file_ids = launched_job_file_ids.get(report_type, [])
        if len(report_file_ids) > 0:
            report_details = call_in_parallel(
                get_details,
                report_file_ids,
                project_id=project_id,
                max_workers=max_workers,
            )
            filtered_reports[report_type] = filter_report_files(
                report_details,
                details_key=rep_config["details_key"]
            )

    logging.debug("Collecting file IDs for download...")
    file_ids_for_download = []
    summary_text_buffer = []
    summary_text_buffer.append(
        f"\nJob ID: {args.batch_job_id}\n"
        f"\nDownload folder:\n"
        f"\tServer path: '{output_dir}'\n"
        f"\tWindows path: '{
            linux_to_windows_path(
                output_dir,
                output_config['linux_prefix'],
                output_config['windows_prefix']
            )
        }'\n"
        f"\nFiles to be downloaded:\n"
    )

    # Merge dicts of batch job file IDs and launched job file IDs,
    # excluding report files to get a dict of non-report files for download
    non_report_file_ids = filter_dict_by_keys(
        d=launched_job_file_ids, exclude_keys=config["filter_reports"].keys()
    )
    non_report_files = batch_job_file_ids | non_report_file_ids

    gather_and_summarise_non_report_files(
        non_report_files=non_report_files,
        summary_text_buffer=summary_text_buffer,
        file_ids_for_download=file_ids_for_download
    )
    gather_and_summarise_report_files(
        filtered_reports=filtered_reports,
        summary_text_buffer=summary_text_buffer,
        file_ids_for_download=file_ids_for_download
    )
    if sample_exec_counts:
        summarise_sample_exec_counts(
            sample_exec_counts=sample_exec_counts,
            summary_text_buffer=summary_text_buffer
        )
    print("".join(summary_text_buffer))

    # Get file names
    files_for_download = call_in_parallel(
        dxpy.describe,
        file_ids_for_download,
        fields={"name": True},
        max_workers=max_workers
    )
    logging.debug("Files for download: %s", files_for_download)

    # If --dry-run and --make-output-dir, will not download files or make dir,
    # therefore no need to check for file overwrites/downloads (dir doesn't
    # exist), so can just exit
    if args.make_output_dir and args.dry_run:
        return

    files_before_download = get_files_in_cwd()
    if not args.overwrite:
        files_for_download = check_file_overwrites(
            files_for_download,
            files_before_download
        )

    # Download files in parallel and check output directory for expected files
    if not args.dry_run:
        logging.debug("Downloading files to: %s", os.getcwd())
        call_in_parallel(
            download_single_file,
            files_for_download,
            project=project_id,
            ignore_all_errors=False,
            max_workers=max_workers
        )
        check_files_after_download(
            files_before_download=files_before_download,
            files_for_download={f["name"] for f in files_for_download},
            files_after_download=get_files_in_cwd()
        )


if __name__ == "__main__":
    main()
