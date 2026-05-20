import logging
from typing import Dict, List


def linux_to_windows_path(
    linux_path: str, linux_prefix: str, windows_prefix: str
) -> str:
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
    str
        The converted Windows-compatible file path.
    """
    stripped_path = linux_path.removeprefix(linux_prefix)
    joined_path = windows_prefix + stripped_path
    return joined_path.replace("/", "\\")


def gather_and_summarise_non_report_files(
    non_report_files: Dict,
    summary_text_buffer: List[str],
    file_ids_for_download: List[str],
    indent: str = "\t",
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
    filtered_reports: Dict,
    summary_text_buffer: List[str],
    file_ids_for_download: List[str],
    indent: str = "\t",
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
    sample_exec_counts: Dict[str, Dict[str, int]],
    summary_text_buffer: List[str],
    indent: str = "\t",
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
    summary_text_buffer.append(f"Total samples: {len(sample_exec_counts)}\n")
    sample_report_count_summaries = []
    for sample, exec_counts in sample_exec_counts.items():
        sample_report_count_summary = ""
        for exec_type, count in exec_counts.items():
            # Check if a sample had multiple runs of an executable, if so
            # generate summary
            if count > 1:
                sample_report_count_summary += (
                    f"{indent * 2}{count} {exec_type}\n"
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
