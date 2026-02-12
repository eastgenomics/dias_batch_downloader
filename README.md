## What does this app do?
This app is a command-line tool designed to gather files associated with a given [eggd_dias_batch](https://github.com/eastgenomics/eggd_dias_batch) job, filter variant report files for download by removing those which do not contain variants, print a summary of the files gathered, check for potential file overwrites (if specified), download the gathered files to the desired output directory, and check that the output directory contains expected files after download.

## How does this app work?
The app will first configure the output directory for downloading, by creating or finding the apropriate folder by using the DNAnexus project name of the project in which the batch job was run, or by using the current working directory.

The app then collects files derived from two main types of DNAnexus describe calls, as specified in the [app config](https://github.com/eastgenomics/eggd_dias_batch_downloader_config).

Firstly, the app gathers files (e.g. QC status file and MultiQC) directly from the output of a dxpy `describe()` call on the specified eggd_dias_batch job ID, using the dictionary paths defined in the config under the `batch_job_query` key.

Next, the app calls dxpy `describe()` on all of the jobs launched by eggd_dias_batch (as provided via the `launched_jobs` output of the batch job). It retreives the names of all the launched jobs, if a job's name matches one of the jobs listed in the config (under the `exec_regex` key), then the corresponding dictionary path(s) are used to retrieve the file(s) (e.g. SNV/CNV/mosaic reports, athena reports, and artemis file).

Once the specified files have been gathered, varaint report files (listed under the `filter_reports` key of the config) are filtered to remove reports with 0 variants, as declared in the reports file details metadata in DNAnexus.

A summary of the files gathered for download is printed (see [description of the summary text](#what-are-the-outputs) below).

If specified, the app checks for any potential file overwrites and skips download of these files.

The gathered and filtered files are then downloaded to the desired output directory (see [Output directory options section](#output-directory-options) below) and a check is run to ensure the output directory contains the expected files after download.

## What are the typical use cases for this app?
This app can be used to return data to scientists when processing a Dias (CEN/TWE) run/re-analysis. It downloads the necessary files to either:
- a new directory (when processing a new run)
- an existing directory (when processing a re-analysis)
- the current working directory (for ad-hoc or custom downloads)

This app can be used to download `eggd_dias_batch` associated files (as specified in the configuration file) for the following `eggd_dias_batch` run modes, when used both individually or in combination (see [eggd_dias_batch readme.md](https://github.com/eastgenomics/eggd_dias_batch/blob/master/readme.md) for more details):
- cnv_reports
- snv_reports
- mosaic_reports
- artemis

## What are the inputs?
### Required inputs:
- #### Configuration file (`-c, --config-file`):
  - A [Python configuration file](https://github.com/eastgenomics/eggd_dias_batch_downloader_config) containing necessary settings for the app.

- #### Batch job ID (`-b, --batch_job_id`):
  - The ID of the `eggd_dias_batch` job whose files need to be downloaded.

### Optional inputs:
  - #### Output directory options (mutually exclusive):
    - `--make-output-dir`: If `--dry-run` is not set, create a new output directory named using the batch job ID and change into it before downloading files. If `--dry-run` is set, simulate creating the directory and include the path that would be made in the summary.

    **OR**

    - `--find-output-dir`: Find an existing output directory for this batch job ID and change into it before downloading files.
>[!NOTE]
> If neither `--make-output-dir` nor `--find-output-dir` are specified, files will be downloaded to the current working directory.
  - #### Dry run (`--dry-run`):
    - If used with --find-output-dir, find and change into output directory, but do not download files. If used with `--make-output-dir`, do not make output directory or download files, but return the download location that would be used in the summary text.
  - #### Overwrite (`--overwrite`):
    - Use the `--overwrite` flag to overwrite existing files during the download process.
  - #### Log level (`--log-level`):
    - Set the logging level using the `--log-level` argument. Options include `DEBUG`, `INFO`, `WARNING`, `ERROR`, and `CRITICAL`. The default is `INFO`.
  - #### Log file (`--log-file`):
    - Specify the path to the log file where logs will be written. If the file exists, logs will be appended to it. If the file does not exist, it will be created. If not specified, logs are only printed to the console.

## What are the outputs?
 - #### Download summary
    - A summary of the files gathered for download (see [example below](#example-summary-text)), including:
    - Job ID of the `eggd_dias_batch` queried
    - The download folders (both the path used by the server and the Window's path used by scientists)
    - The number of each file type to be downloaded including
      - Non-report files (e.g. QC status, MultiQC and artemis file)
      - Report files (e.g. SNV, CNV, mosaic) including the number of reports skipped due to having no variants
    - The total number of samples
    - A list of samples which have more than one reports workflow launched by batch (i.e. when >1 test code as been requested for a sample)
- #### The downloaded files and download directory (if making a new directory is specified)

##### Example summary text:
```
Download folder:
        Server path: '/path/on/server'
        Windows path: '\path\on\clingen'

Files to be downloaded:
        1 qc_file file(s)
        96 athena_report file(s)
        1 artemis file(s)

        29 out of 96 SNV report(s) (67 skipped)
        2 out of 94 CNV report(s) (92 skipped)

Total samples: 95

Samples with multiple launched jobs:
        <instrument-id>-<specimen-id>:
                2 cnv_reports_workflow
                2 snv_reports_workflow
```

## How to run this app from command line?

### Example commands:
Make new download folder, skip file overwrites and download files:
```bash
 python eggd_dias_batch_downloader.py -c /path/to/config.py -b job-xyz --make-output-dir
 ```

Find existing download folder, skip file overwrites and download files:
```bash
 python eggd_dias_batch_downloader.py -c /path/to/config.py -b job-xyz --find-output-dir
 ```

Perform a dry run to simulate creating a new download folder and downloading files:
```bash
 python eggd_dias_batch_downloader.py -c /path/to/config.py -b job-xyz --make-output-dir --dry-run
 ```

Make a new download folder and overwrite existing files during download:
```bash
 python eggd_dias_batch_downloader.py -c /path/to/config.py -b job-xyz --make-output-dir --overwrite
 ```

Download files to the current working directory without creating or finding an output folder:
```bash
 python eggd_dias_batch_downloader.py -c /path/to/config.py -b job-xyz
 ```

Set the logging level to DEBUG and make a new download folder:
```bash
 python eggd_dias_batch_downloader.py -c /path/to/config.py -b job-xyz --make-output-dir --log-level DEBUG
 ```



