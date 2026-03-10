#!/opt/eggd_dias_batch_downloader/venvs/current/bin/python

"""Download files associated with an eggd_dias_batch job."""

import argparse
import json
import logging
import os
from pathlib import Path

import dxpy

from config import configure_logging, load_config
from dx_operations import (
    configure_output_directory,
    describe_batch_job,
    download_single_file,
    filter_report_files,
    get_batch_job_file_ids,
    get_details,
    get_launched_job_file_ids,
)
from file_operations import (
    check_file_overwrites,
    check_files_after_download,
    get_files_in_cwd,
)
from summary import (
    gather_and_summarise_non_report_files,
    gather_and_summarise_report_files,
    linux_to_windows_path,
    summarise_sample_exec_counts,
)
from utils import call_in_parallel, filter_dict_by_keys

VERSION = "1.0.0"


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
        "-c",
        "--config-file",
        help="Path to config file",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "-b",
        "--batch_job_id",
        help="eggd_dias_batch job ID that was used to create the reports you "
        " wish to download",
        required=True,
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
        ),
    )
    output_dir_group.add_argument(
        "--find-output-dir",
        action="store_true",
        help=(
            "Find an existing output directory for this batch job ID and "
            "change into it before downloading files. Cannot be used with "
            "--make-output-dir."
        ),
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
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, existing files will be overwritten during download. If "
        "not set, files with names that already exist in the output directory "
        "will not be downloaded.",
    )

    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (default: INFO)",
    )

    parser.add_argument(
        "--log-file",
        type=Path,
        help="Path to the log file where logs will be written",
    )

    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    configure_logging(log_level=args.log_level.upper(), log_file=args.log_file)
    logging.debug("Running eggd_dias_batch_downloader v%s", VERSION)

    config = load_config(args.config_file)
    logging.debug("Config loaded:\n %s", json.dumps(config, indent=2))

    logging.debug("Reading batch job (%s) metadata...", args.batch_job_id)
    batch_job_dx_desc = describe_batch_job(
        batch_job_id=args.batch_job_id,
        fields=config["batch_job_query"]["dx_desc_fields"],
        exec_regex=config["batch_job_query"]["exec_regex"],
    )

    project_id = batch_job_dx_desc["project"]
    output_config = config["output_config"]
    output_dir = configure_output_directory(
        make_output_dir=args.make_output_dir,
        find_output_dir=args.find_output_dir,
        dry_run=args.dry_run,
        project_id=project_id,
        assay=batch_job_dx_desc["input"]["assay"],
        output_config=output_config,
    )

    batch_job_file_ids = get_batch_job_file_ids(
        desc_dict=batch_job_dx_desc,
        query_config=config["batch_job_query"]["files"],
    )

    launched_jobs = batch_job_dx_desc["output"]["launched_jobs"].split(",")
    logging.debug("DX describing launched jobs: %s", launched_jobs)
    max_workers = config["max_workers"]
    launched_job_desc_dicts = call_in_parallel(
        func=dxpy.describe,
        items=launched_jobs,
        fields=config["launched_job_query"]["dx_desc_fields"],
        max_workers=max_workers,
    )

    logging.debug("Gathering launched job output file IDs...")
    launched_job_file_ids, sample_exec_counts = get_launched_job_file_ids(
        desc_dicts=launched_job_desc_dicts,
        query_config=config["launched_job_query"]["execs"],
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
                report_details, details_key=rep_config["details_key"]
            )

    logging.debug("Collecting file IDs for download...")
    file_ids_for_download = []
    summary_text_buffer = []
    windows_path = linux_to_windows_path(
        output_dir,
        output_config["linux_prefix"],
        output_config["windows_prefix"],
    )
    summary_text_buffer.append(
        f"\nJob ID: {args.batch_job_id}\n"
        f"\nDownload folder:\n"
        f"\tServer path: '{output_dir}'\n"
        f"\tWindows path: '{windows_path}'\n"
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
        file_ids_for_download=file_ids_for_download,
    )
    gather_and_summarise_report_files(
        filtered_reports=filtered_reports,
        summary_text_buffer=summary_text_buffer,
        file_ids_for_download=file_ids_for_download,
    )
    if sample_exec_counts:
        summarise_sample_exec_counts(
            sample_exec_counts=sample_exec_counts,
            summary_text_buffer=summary_text_buffer,
        )
    print("".join(summary_text_buffer))

    # Get file names
    files_for_download = call_in_parallel(
        dxpy.describe,
        file_ids_for_download,
        fields={"name": True},
        max_workers=max_workers,
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
            files_for_download, files_before_download
        )

    print("Downloading files...")
    # Download files in parallel and check output directory for expected files
    if not args.dry_run:
        logging.debug("Downloading files to: %s", os.getcwd())
        call_in_parallel(
            download_single_file,
            files_for_download,
            project=project_id,
            ignore_all_errors=False,
            max_workers=max_workers,
        )
        check_files_after_download(
            files_before_download=files_before_download,
            files_for_download={f["name"] for f in files_for_download},
            files_after_download=get_files_in_cwd(),
        )


if __name__ == "__main__":
    main()
