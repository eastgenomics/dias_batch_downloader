import logging
import os
import re
from typing import Any, Dict, List, Tuple

import dxpy

from utils import traverse_dict


def describe_batch_job(
    batch_job_id: str, fields: Dict[str, bool], exec_regex: str
) -> Dict[str, Any]:
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

    ValueError
        If the specified job is not in 'done' state.

    Returns
    -------
    dict
        A dictionary containing the description of the specified batch job,
        including only the specified fields.
    """
    desc_dict = dxpy.bindings.DXJob(dxid=batch_job_id).describe(fields=fields)
    if not re.search(exec_regex, desc_dict["executableName"]):
        raise ValueError(
            f"The specified job ({batch_job_id}) does not correspond to an "
            "eggd_dias_batch job."
        )
    if desc_dict["state"] != "done":
        raise ValueError(
            f"The specified job ({batch_job_id}) is not in 'done' state, "
            f"current state is '{desc_dict['state']}'. Only batch jobs in "
            "'done' state can be used for downloading files."
        )

    return desc_dict


def configure_output_directory(
    make_output_dir: bool,
    find_output_dir: bool,
    dry_run: bool,
    project_id: str,
    assay: str,
    output_config: Dict,
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
        logging.debug(
            "No output directory options specified, downloading "
            "files to current working directory: %s",
            cwd,
        )
        return cwd

    # Get output directory from DNAnexus project name
    project_name = dxpy.describe(project_id)["name"]
    run_name = re.sub(r"^[0-9]+_", "", project_name)
    run_name = re.sub(r"_(?:37|38)_", "_", run_name)

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


def get_batch_job_file_ids(desc_dict: Dict, query_config: Dict) -> Dict:
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


def get_launched_job_file_ids(
    desc_dicts: List[Dict], query_config: Dict
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, int]]]:
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
                        sample_name_regex, desc["name"]
                    )
                    if sample_name_match:
                        # Count number of times each sample has a launched job
                        # for each exec type
                        sample_name = sample_name_match.group(1)
                        sample_exec_counts.setdefault(sample_name, {})
                        sample_exec_counts[sample_name].setdefault(
                            exec_type, 0
                        )
                        sample_exec_counts[sample_name][exec_type] += 1
                    else:
                        logging.warning(
                            "Sample name could not be extracted from "
                            "job name: %s (%s)",
                            desc["name"],
                            desc["id"],
                        )

                # Extract output file IDs using paths specified in config
                for file_type, desc_path in exec_config["desc_paths"].items():
                    output_file_id = traverse_dict(desc, desc_path)
                    if output_file_id is None:
                        logging.warning(
                            "Output file for %s (%s) could not be found",
                            desc["name"],
                            desc["id"],
                        )
                    else:
                        dx_ids.setdefault(file_type, []).append(output_file_id)
    return dx_ids, sample_exec_counts


def get_details(file_id: str, project_id: str) -> Tuple[str, Dict]:
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


def filter_report_files(
    reports_details: list[tuple[str, dict]], details_key: str
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


def download_single_file(file_dict: Dict, project: str) -> None:
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
        dxid=file_dict["id"], filename=file_dict["name"], project=project
    )
