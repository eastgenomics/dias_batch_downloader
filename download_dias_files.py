"""
Given an eggd_dias_batch job ID, this script downloads the associated SNV/CNV
reports if they contain variants and prints a short summary describing the
number of SNV and CNV reports downloaded. The script also downloads the
artemis file, QC status file and MultiQC report - if these are able to be
found.
"""
import re
from typing import Any, Optional, Callable
import concurrent
import argparse
import dxpy

CONFIG = {
    "batch_job_query": {
        "qc_file": {
            "desc_path": [
                "input",
                "qc_file",
                "$dnanexus_link"
            ]
        },
        "multiqc_report": {
            "desc_path": [
                "input",
                "multiqc_report",
                "$dnanexus_link"
            ]
        }
    },
    "launched_job_query": {
        "snv_reports": {
            "exec_regex": r"^dias_reports",
            "desc_path": [
                "output",
                "stage-rpt_generate_workbook.xlsx_report",
                "$dnanexus_link"
            ],
        },
        "cnv_reports": {
            "exec_regex": r"^dias_cnvreports",
            "desc_path": [
                "output",
                "stage-cnv_generate_workbook.xlsx_report",
                "$dnanexus_link"
            ],
        },
        "artemis_file": {
            "exec_regex": r"^eggd_artemis",
            "desc_path": [
                "output",
                "url_file",
                "$dnanexus_link"
            ],
        }
    },
    "filter_reports": {
        "snv_reports": {
            "details_key": "included",
            "report_type": "SNV"  # Used for printing messages
        },
        "cnv_reports": {
            "details_key": "variants",
            "report_type": "CNV"
        }
    }
}


def parse_args() -> argparse.Namespace:
    """
    Parse arguments given at cmd line.

    Args: None

    Returns:
        - args (Namespace): object containing parsed arguments.
    """

    parser = argparse.ArgumentParser(
        description="Download SNV/CNV reports, artemis file, QC status file\
            and Multi QC file"
        )

    parser.add_argument(
        "-b", "--batch_job_id",
        help="eggd_dias_batch job ID that was used to create the reports you\
            wish to download", required=True
        )

    args = parser.parse_args()

    return args


def traverse_dict(d: dict, path: list) -> Optional[Any]:
    """Traverse a dictionary using a list of keys as a path."""
    for key in path:
        d = d.get(key)
        # If specified path cannot be fully traversed, return None
        if d is None:
            break
    return d


def read_batch_job_metadata(desc_dict: dict, query_config: dict) -> list[str]:
    """
    Extract file IDs from a dx describe dictionary describing a eggd_dias_batch
    job, using paths defined in a config.

    If a file ID can be found, it is added to the result list; otherwise,
    a warning is printed.

    Parameters
    ----------
    desc_dict : dict
        DNAnexus batch job description dictionary.
    query_config : dict
        Config dict specifying keys and paths to extract file IDs.

    Returns
    -------
    list
        List of file IDs found in the batch job metadata. If a file ID is not
        found, it is omitted from the list and a warning is printed.
    """
    dx_ids = []
    for k, config in query_config.items():
        # QC status, multiqc report are optional input for batch so
        # may not be present, so check for their presence and print warnings
        # if not
        dx_id = traverse_dict(desc_dict, config["desc_path"])
        if dx_id is None:
            print(
                f"{k} could not be found in eggd_dias_batch job metadata"
            )
        else:
            dx_ids.append(dx_id)
    return dx_ids


def call_in_parallel(func: Callable, items: list, ignore_missing: bool = True,
                     ignore_all_errors: bool = True, **kwargs) -> list:
    """
    Calls the given function in parallel using concurrent.futures on
    the given set of items (i.e for calling dxpy.describe() on multiple
    object IDs).

    Additional arguments specified to kwargs are directly passed to the
    specified function.

    Parameters
    ----------
    func : callable
        function to call on each item
    items : list
        list of items to call function on
    ignore_missing : bool
        controls if to just print a warning instead of raising an
        exception on a dxpy.exceptions.ResourceNotFound being raised.
        This is most likely from a file that has been deleted and we are
        just going to default to ignoring these
    ignore_all_errors : bool
        controls if to just print a warning instead of raising an exception.

    Returns
    -------
    list
        list of responses
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
        concurrent_jobs = {
            executor.submit(func, item, **kwargs): item for item in items
        }

        for future in concurrent.futures.as_completed(concurrent_jobs):
            # access returned output as each is returned in any order
            try:
                results.append(future.result())
            except Exception as exc:
                if (
                    ignore_missing and
                    isinstance(exc, dxpy.exceptions.ResourceNotFound)
                ):
                    # dx object does not exist and specifying to skip,
                    # just print warning and continue'
                    print(
                        f'WARNING: {concurrent_jobs[future]} could not be '
                        'found, skipping to not raise an exception'
                    )
                    continue
                # catch any other errors that might get raised during querying
                print(
                    f"Warning: Error getting data for {concurrent_jobs[future]}: {exc}"
                )

                if ignore_all_errors is False:
                    raise exc

    return results


def get_files_ids(
        desc_dicts: list[dict],
        query_config: dict
) -> dict[Any, list[str]]:
    """
    Extracts output file IDs from dx describe dictionies describing the jobs
    launched by eggd_dias_batch.

    For each job description, matches the executable name using the regex in
    query_config. If matched, traverses the describe dictionary to extract the
    file ID using the specified path.

    Collects file IDs by type and reports jobs where output files could not be
    found.

    Parameters
    ----------
    desc_dicts : list of dict
        List of job description dictionaries.
    query_config : dict
        Config dict specifying regex and path for each file type.

    Returns
    -------
    dict[Any, list[str]]
        Dictionary mapping file type keys to lists of file IDs.
    """
    dx_ids = {k: [] for k in query_config}
    failed_jobs = []
    for desc in desc_dicts:
        for k, config in query_config.items():
            if re.search(config["exec_regex"], desc["executableName"]):
                output_file_id = traverse_dict(desc, config["desc_path"])
                if output_file_id is None:
                    failed_jobs.append({
                        "name": desc["executableName"],
                        "id": desc["id"]
                    })
                else:
                    dx_ids[k].append(output_file_id)

    if len(failed_jobs) > 0:
        print(
            "Warning: output files could not be gathered for the following "
            f"jobs:\n {failed_jobs}"
        )
    return dx_ids


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


def organise_report_files(
        reports_details: list[tuple[str, dict]],
        details_key: str,
        report_type: str
) -> list[str]:
    """
    Summarises and filters SNV/CNV report files by variant content.

    Prints the number of reports that contain variants, do not contain
    variants, or lack file details. Returns a list of file IDs for reports
    that should be downloaded (i.e., those with variants).

    Parameters
    ----------
    reports_details : list
        List of tuples: (file ID of the report, file details dict for the
        corresponding report).
    details_key : str
        Key in the file details dict providing the number of variants in the
        report.
    report_type : str
        String label for the report type (e.g., "SNV" or "CNV") - used when
        printing messages.

    Returns
    -------
    list
        List of file IDs for SNV/CNV .xlsx reports to be downloaded.
    """

    n_reports_without_details = 0
    n_reports_without_vars = 0
    reports_for_download = []

    for report_id, details in reports_details:
        # If old report and no "file details" metadata is available then
        # n_vars will = None
        n_vars = details.get(details_key)
        if n_vars is None:
            n_reports_without_details += 1
        elif n_vars == 0:
            n_reports_without_vars += 1
        elif n_vars > 0:
            reports_for_download.append(report_id)
        else:
            print(
                "Warning: DNAnexus 'file details' metadata "
                f"for {report_id} are not interpretable"
            )

    print(
        f'{len(reports_for_download)} {report_type} report(s) with variants'
    )
    print(
        f'{n_reports_without_vars} {report_type} report(s) without'
        ' variants'
    )
    # Unlikely to have reports without details unless using an old batch
    # job, therefore do not need to print the number of reports without details
    # every time
    if n_reports_without_details > 0:
        print(
            f'{n_reports_without_details} {report_type} report(s) do not '
            'have DNAnexus metadata about number of variants in the report'
        )

    return reports_for_download


def download_single_file(dxid: str, project: str) -> None:
    """
    Given a single dx file ID, download it with the original filename

    Parameters
    ----------
    dxid : str
        file ID of object to download
    project : str
        project containing the file
    """
    dxpy.bindings.dxfile_functions.download_dxfile(
        dxid,
        dxpy.describe(dxid).get('name'),
        project=project
    )


def main():
    args = parse_args()

    batch_job_dx_desc = dxpy.bindings.DXJob(dxid=args.batch_job_id).describe(
        fields={"output": True, "input": True, "project": True}
    )
    project_id = batch_job_dx_desc["project"]

    print("Reading eggd_dias_batch job metadata...")
    batch_job_dx_ids = read_batch_job_metadata(
        desc_dict=batch_job_dx_desc,
        query_config=CONFIG["batch_job_query"]
    )

    print("DX describing launched jobs...")
    launched_job_desc_dicts = call_in_parallel(
        func=dxpy.describe,
        items=batch_job_dx_desc["output"]["launched_jobs"].split(","),
        fields={
            "executableName": True,
            "output": True
        }
    )
    print("Done DX describing launched jobs.")

    print("Gathering launched job output file IDs...")
    launched_job_dx_ids = get_files_ids(
        desc_dicts=launched_job_desc_dicts,
        query_config=CONFIG["launched_job_query"]
    )

    filtered_report_dx_ids = []
    for report_type, config in CONFIG["filter_reports"].items():
        report_dx_ids = launched_job_dx_ids[report_type]

        # Batch may not have launched CNV reports workflows (e.g. for TWE runs)
        # therefore check report file IDs have been retrieved before trying to
        # get their details
        if len(report_dx_ids) > 0:
            report_details = call_in_parallel(
                get_details,
                report_dx_ids,
                project_id=project_id
            )
            filtered_report_dx_ids.extend(
                organise_report_files(
                    report_details,
                    details_key=config["details_key"],
                    report_type=config["report_type"]
                )
            )

    # Gather files for download
    file_ids_for_download = []
    for dx_id_list in [
        batch_job_dx_ids,
        launched_job_dx_ids["artemis_file"],
        filtered_report_dx_ids
    ]:
        file_ids_for_download.extend(dx_id_list)

    call_in_parallel(
        download_single_file, file_ids_for_download, project=project_id
    )


if __name__ == "__main__":
    main()
