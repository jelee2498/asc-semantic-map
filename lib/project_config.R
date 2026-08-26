# Shared path and parameter configuration (R side).
#
# The R counterpart of lib/project_config.py, reading the same
# config/paths.yml and config/params.yml so the two languages cannot drift.
# The Python and R halves of the P7 chain hand data to each other through CSV
# files whose directory names encode analysis parameters; if the two sides
# disagree about a single parameter, the handoff does not error - the R script
# simply reads an empty directory.
#
# The original scripts hard-coded `store7 <- "S:/"` (or "/MIPL/store7") in each
# file, which is machine-specific and is not distributed.
#
# Usage
# -----
#   source(file.path(dirname(sys.frame(1)$ofile), "..", "..", "lib", "project_config.R"))
#   or, more simply, from a pipeline script:
#       source(asc_find_lib("project_config.R"))
#
#   PROJECT   <- asc_project()
#   PIPELINE  <- asc_pipeline()
#   p         <- asc_params()
#   folder    <- asc_task_dir("normative_sem")

suppressMessages(library(yaml))


# ---------------------------------------------------------------------------
# Locate the repository
# ---------------------------------------------------------------------------

#' Directory of the script currently being executed.
#'
#' Works under Rscript, source() and RStudio; falls back to the working
#' directory when none of those apply.
asc_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  file_arg <- grep("^--file=", args, value = TRUE)
  if (length(file_arg) > 0) {
    return(normalizePath(dirname(sub("^--file=", "", file_arg[1])), mustWork = FALSE))
  }
  # source()d from another script
  for (i in seq_len(sys.nframe())) {
    ofile <- sys.frame(i)$ofile
    if (!is.null(ofile)) {
      return(normalizePath(dirname(ofile), mustWork = FALSE))
    }
  }
  # RStudio
  if (requireNamespace("rstudioapi", quietly = TRUE) &&
      rstudioapi::isAvailable()) {
    path <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) "")
    if (nzchar(path)) return(normalizePath(dirname(path), mustWork = FALSE))
  }
  normalizePath(getwd(), mustWork = FALSE)
}


#' Walk upward from `start` until a directory containing config/paths.yml is found.
asc_repo_root <- function(start = asc_script_dir()) {
  here <- normalizePath(start, mustWork = FALSE)
  for (i in 1:10) {
    if (file.exists(file.path(here, "config", "paths.yml"))) return(here)
    parent <- dirname(here)
    if (identical(parent, here)) break
    here <- parent
  }
  stop("Could not locate config/paths.yml above ", start,
       ". project_config.R must stay inside the repository.")
}


.asc_cache <- new.env(parent = emptyenv())

.asc_load <- function() {
  if (is.null(.asc_cache$paths)) {
    root <- asc_repo_root()
    .asc_cache$root   <- root
    .asc_cache$paths  <- yaml::read_yaml(file.path(root, "config", "paths.yml"))
    .asc_cache$params <- yaml::read_yaml(file.path(root, "config", "params.yml"))
  }
  invisible(NULL)
}


# ---------------------------------------------------------------------------
# Accessors
# ---------------------------------------------------------------------------

#' Repository root.
asc_root <- function() { .asc_load(); .asc_cache$root }

#' Project root - the directory containing 0_data/, 1_code/, 2_pipeline/.
asc_project <- function() { .asc_load(); .asc_cache$paths$roots$project }

#' Resolve a root that may be given relative to the repository.
asc_resolve <- function(value) {
  if (is.null(value)) return(NULL)
  if (grepl("^([A-Za-z]:|/|\\\\\\\\)", value)) return(value)
  normalizePath(file.path(asc_root(), value), mustWork = FALSE)
}

#' Vendored atlases and surfaces.
asc_templates <- function() { .asc_load(); asc_resolve(.asc_cache$paths$roots$templates) }

#' Generated figures, one subdirectory per pipeline.
asc_figures <- function(pipeline = NULL, create = TRUE) {
  .asc_load()
  out <- asc_resolve(.asc_cache$paths$roots$figures)
  if (!is.null(pipeline)) out <- file.path(out, pipeline)
  if (isTRUE(create)) dir.create(out, recursive = TRUE, showWarnings = FALSE)
  out
}

#' {project}/2_pipeline/99_main - analysis outputs, one directory per task.
asc_pipeline <- function() {
  .asc_load()
  sub("\\{project\\}", asc_project(), .asc_cache$paths$project$pipeline)
}

#' Output directory for a pipeline stage, by logical name or literal directory.
asc_task_dir <- function(task) {
  .asc_load()
  tasks <- .asc_cache$paths$pipeline_tasks
  name <- if (!is.null(tasks[[task]])) tasks[[task]] else task
  file.path(asc_pipeline(), name)
}

#' The manuscript configuration (config/params.yml) as a nested list.
#'
#' These values are not tuning knobs: every script encodes them into its output
#' directory path, so a changed value does not raise - it silently writes
#' somewhere else and reads back nothing.
asc_params <- function() { .asc_load(); .asc_cache$params }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
# The directory names were produced by Python's str() of floats and bools, so R
# has to reproduce that exactly: 1.0 stays "1.0", TRUE becomes "True".

#' Format a number the way Python's str() would (0.9 -> "0.9", 1 -> "1.0").
asc_fmt_num <- function(x) {
  if (x == floor(x)) sprintf("%.1f", x) else as.character(x)
}

#' Format a logical the way Python would (TRUE -> "True").
asc_fmt_bool <- function(x) if (isTRUE(x)) "True" else "False"
