# 12_heterogeneity_2sem_zscore.R
# Refit the selected SEM (Model 1) within each ASD subtype
#
# This script:
# 1. Loads the subtype-specific SEM tables from 12_heterogeneity_1save_data.py
# 2. Fits Model 1 -- the model selected in 11_sem_zscore_3analysis.R -- to
#      (a) the pooled sample  (TD + all clustered ASD), as the reference
#      (b) TD + ASD subtype-0
#      (c) TD + ASD subtype-1
# 3. Compares fit indices and standardized path estimates across the three fits
# 4. Refits Model 1 with labeled paths to compare the mediation / cascade
#    effects (ind_sal_dmn, total_sal_dmn, cascade_sal_srs, cascade_sal_cog)
# 5. Runs a multi-group invariance test on the ASD-only table (group = subtype)
#    as the formal between-subtype test
#
# Model 1 (from 11_sem_zscore_3analysis.R):
#   avg_cen ~ avg_sal
#   avg_dmn ~ avg_sal + avg_cen
#   cog_beh ~ avg_dmn
#   srs_beh ~ avg_dmn
#
# Why the pooled fit is included: the subtype fits use a smaller ASD n, so their
# estimates move even when nothing about the structure changed. The pooled row
# is the yardstick the subtype rows are read against.
#
# Important: TD subjects are SHARED between the two subtype tables, so those two
# fits are not independent -- do not read a difference between them as a test.
# Section 5 (ASD-only multi-group) is the test.
#
# Useful commands:
# - clear console window : Ctrl + L
# - clear data           : rm(list = ls())

library(lavaan)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths and parameters come from config/{paths,params}.yml via
# lib/project_config.R.  The original hard-coded `store7 <- "S:/"` here and
# repeated the analysis parameters as literals, which is the drift risk the
# shared config exists to prevent: the Python and R halves of this chain meet
# through a directory name built from those values, and a mismatch reads an
# empty directory rather than erroring.
.asc_here <- local({
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a) > 0) return(dirname(sub("^--file=", "", a[1])))
  for (i in seq_len(sys.nframe())) {
    o <- sys.frame(i)$ofile
    if (!is.null(o)) return(dirname(o))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    pth <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) "")
    if (nzchar(pth)) return(dirname(pth))
  }
  getwd()
})
.lib <- normalizePath(file.path(.asc_here, "..", "..", "lib"), mustWork = TRUE)
source(file.path(.lib, "project_config.R"))
.p <- asc_params()

proj_root <- asc_project()

# Parameters (matching 12_heterogeneity_1save_data.py)
proj <- "02_asd_semantic_map"
pipe <- "99_main"
task <- "12_heterogeneity"
conf_option <- "default+me"
atlas <- "mmp"
chunk_option <- 9
srm_option_pc <- 0
srm_option_dim <- 50
bias_weight <- 0.9
verb_weight <- 1.0
reg_wordnet <- TRUE
enc_single_alpha <- TRUE
perf_method <- "fdr"
perf_alpha <- 0.01
weight_across_delays <- "Avg"
weight_add_superordinate <- TRUE
screen <- .p$normative_sem$screen# matching 11_sem_zscore_0save_raw.py
sig_overlap <- .p$normative_sem$sig_overlap# matching 11_sem_zscore_0save_raw.py

# Subtype source (matching 12_heterogeneity_1save_data.py)
cluster_method <- .p$subtyping$clustering_method# "kmeans" or "hierarchical"
cluster_mode <- "profile"     # "profile" or "severity"

# Analysis options
run_multigroup <- TRUE   # ASD-only multi-group invariance test (Section 5)
save_outputs <- TRUE

# =============================================================================
# CONSTRUCT DATA PATH
# =============================================================================

# Format numbers to match Python output (preserve .0 for whole numbers)
fmt_num <- function(x) {
  if (x == floor(x)) {
    return(sprintf("%.1f", x))
  } else {
    return(as.character(x))
  }
}

# Format boolean to match Python output (True/False instead of TRUE/FALSE)
fmt_bool <- function(x) {
  if (x) return("True") else return("False")
}

# Note: 12_heterogeneity_1save_data.py keys this folder on screen/overlap/cluster
# only, not on the full upstream parameter tree -- the full tree exceeds the
# Windows 260-character path limit here. The upstream parameters above are kept
# for reference and are recorded in source_paths.txt inside the folder.
data_folder <- file.path(proj_root, "2_pipeline", pipe, task, "out",
                         paste0("screen-", fmt_bool(screen), "_overlap-", fmt_bool(sig_overlap)),
                         paste0("cluster-", cluster_method, "_", cluster_mode))

cat("=======================================================================\n")
cat("12_heterogeneity_2sem_zscore: Model 1 refit within ASD subtypes\n")
cat("=======================================================================\n")
cat("\nData folder:", data_folder, "\n")

if (!dir.exists(data_folder)) {
  stop("Data folder not found. Run 12_heterogeneity_1save_data.py first.")
}

# =============================================================================
# LOAD DATA
# =============================================================================

cat("\n[1/5] Loading subtype-specific SEM tables...\n")

# Pooled reference
pooled_file <- file.path(data_folder, "sem_input_pooled.csv")
if (!file.exists(pooled_file)) {
  stop("sem_input_pooled.csv not found. Run 12_heterogeneity_1save_data.py first.")
}
pooled_data <- read.csv(pooled_file, row.names = 1)

# Subtype tables (one per cluster; discovered from the folder)
subtype_files <- sort(list.files(data_folder, pattern = "^sem_input_subtype-[0-9]+\\.csv$"))
if (length(subtype_files) == 0) {
  stop("No sem_input_subtype-*.csv files found in the data folder.")
}

subtype_ids <- gsub("^sem_input_subtype-([0-9]+)\\.csv$", "\\1", subtype_files)
subtype_data <- lapply(file.path(data_folder, subtype_files),
                       read.csv, row.names = 1)
names(subtype_data) <- paste0("subtype-", subtype_ids)

# Datasets to fit: pooled reference first, then each subtype
datasets <- c(list(pooled = pooled_data), subtype_data)

for (nm in names(datasets)) {
  d <- datasets[[nm]]
  n_td <- sum(d$DX == "TD")
  n_asd <- sum(d$DX == "ASD")
  cat(sprintf("  %-12s n = %3d  (TD = %3d, ASD = %3d)\n", nm, nrow(d), n_td, n_asd))
}

cat("\n  Note: TD subjects are shared across the subtype tables, so the subtype\n")
cat("        fits are not independent of each other or of the pooled fit.\n")

# =============================================================================
# DEFINE MODEL 1 (selected in 11_sem_zscore_3analysis.R)
# =============================================================================

# Plain specification, used for the fit-index comparison
model1 <- '
  # Measurement model
  srs_beh =~ SRS_AWR_T + SRS_COG_T + SRS_COM_T + SRS_MOT_T + SRS_RRB_T
  cog_beh =~ NIH7_Card_P + NIH7_Flanker_P + NIH7_List_P + NIH7_Pattern_P

  # Structural model
  avg_cen ~ avg_sal
  avg_dmn ~ avg_sal + avg_cen
  cog_beh ~ avg_dmn
  srs_beh ~ avg_dmn
  srs_beh ~~ 0*cog_beh
'

# Labeled specification with defined indirect / cascade effects
model1_med <- '
  # Measurement model
  srs_beh =~ SRS_AWR_T + SRS_COG_T + SRS_COM_T + SRS_MOT_T + SRS_RRB_T
  cog_beh =~ NIH7_Card_P + NIH7_Flanker_P + NIH7_List_P + NIH7_Pattern_P

  # Structural model (with labels for mediation)
  avg_cen ~ a*avg_sal
  avg_dmn ~ c*avg_sal + b*avg_cen

  srs_beh ~ d*avg_dmn
  cog_beh ~ e*avg_dmn

  # Constraints
  srs_beh ~~ 0*cog_beh

  # Defined parameters
  ind_sal_dmn     := a*b        # SAL -> CEN -> DMN
  total_sal_dmn   := c + (a*b)  # total effect of SAL on DMN
  cascade_sal_srs := a*b*d      # SAL -> CEN -> DMN -> SRS
  cascade_sal_cog := a*b*e      # SAL -> CEN -> DMN -> COG
'

# =============================================================================
# FIT MODEL 1 IN EACH SUBSET
# =============================================================================

cat("\n[2/5] Fitting Model 1 in each subset...\n")

fits <- list()
fits_med <- list()

for (nm in names(datasets)) {
  cat("  Fitting:", nm, "\n")

  fits[[nm]] <- tryCatch(
    sem(model1, data = datasets[[nm]]),
    error = function(e) {
      cat("    ERROR:", conditionMessage(e), "\n")
      NULL
    }
  )

  fits_med[[nm]] <- tryCatch(
    sem(model1_med, data = datasets[[nm]]),
    error = function(e) {
      cat("    ERROR (mediation):", conditionMessage(e), "\n")
      NULL
    }
  )

  if (!is.null(fits[[nm]])) {
    if (!lavInspect(fits[[nm]], "converged")) {
      cat("    WARNING: model did not converge\n")
    } else {
      # Negative variances / correlations > 1 make estimates uninterpretable
      if (any(lavInspect(fits[[nm]], "post.check") == FALSE)) {
        cat("    WARNING: post-fit check failed (e.g. negative variance estimate)\n")
      }
    }
  }
}

# =============================================================================
# FIT INDEX COMPARISON
# =============================================================================

cat("\n[3/5] Fit index comparison\n")
cat("=======================================================================\n")

get_fit_row <- function(fit, label) {
  if (is.null(fit) || !lavInspect(fit, "converged")) {
    return(data.frame(Subset = label, N = NA, AIC = NA, BIC = NA, Chisq = NA,
                      df = NA, pvalue = NA, CFI = NA, TLI = NA,
                      RMSEA = NA, SRMR = NA, stringsAsFactors = FALSE))
  }
  fm <- fitMeasures(fit, c("aic", "bic", "chisq", "df", "pvalue",
                           "cfi", "tli", "rmsea", "srmr"))
  data.frame(
    Subset = label,
    # "ntotal", not "nobs": nobs returns one value per group, which would
    # duplicate the row for every multi-group fit.
    N      = lavInspect(fit, "ntotal"),
    AIC    = fm[["aic"]],
    BIC    = fm[["bic"]],
    Chisq  = fm[["chisq"]],
    df     = fm[["df"]],
    pvalue = fm[["pvalue"]],
    CFI    = fm[["cfi"]],
    TLI    = fm[["tli"]],
    RMSEA  = fm[["rmsea"]],
    SRMR   = fm[["srmr"]],
    stringsAsFactors = FALSE
  )
}

fit_table <- do.call(rbind, lapply(names(fits), function(nm) get_fit_row(fits[[nm]], nm)))
rownames(fit_table) <- NULL

print(fit_table, digits = 4, row.names = FALSE)

cat("\nConventional cut-offs: CFI/TLI >= .95 (good), .90 (acceptable);\n")
cat("                       RMSEA <= .06 (good), .08 (acceptable); SRMR <= .08\n")
cat("Note: AIC/BIC are NOT comparable across subsets (different N and data).\n")
cat("      Only the descriptive indices (CFI/TLI/RMSEA/SRMR) are.\n")

# =============================================================================
# STRUCTURAL PATH COMPARISON
# =============================================================================

cat("\n[4/5] Structural path comparison (standardized)\n")
cat("=======================================================================\n")

# The five structural paths of Model 1
structural_paths <- data.frame(
  lhs = c("avg_cen", "avg_dmn", "avg_dmn", "cog_beh", "srs_beh"),
  op  = rep("~", 5),
  rhs = c("avg_sal", "avg_sal", "avg_cen", "avg_dmn", "avg_dmn"),
  label = c("a: SAL->CEN", "c: SAL->DMN", "b: CEN->DMN",
            "e: DMN->COG", "d: DMN->SRS"),
  stringsAsFactors = FALSE
)

get_path_rows <- function(fit, label) {
  if (is.null(fit) || !lavInspect(fit, "converged")) return(NULL)

  std <- standardizedSolution(fit)

  out <- lapply(seq_len(nrow(structural_paths)), function(i) {
    p <- structural_paths[i, ]
    row <- std[std$lhs == p$lhs & std$op == p$op & std$rhs == p$rhs, ]
    if (nrow(row) == 0) return(NULL)
    data.frame(
      Subset   = label,
      Path     = p$label,
      Est_std  = row$est.std[1],
      SE       = row$se[1],
      z        = row$z[1],
      p        = row$pvalue[1],
      CI_lower = row$ci.lower[1],
      CI_upper = row$ci.upper[1],
      stringsAsFactors = FALSE
    )
  })

  do.call(rbind, out)
}

path_table <- do.call(rbind, lapply(names(fits), function(nm) get_path_rows(fits[[nm]], nm)))
rownames(path_table) <- NULL

# Print grouped by path so the subsets sit side by side
for (lbl in structural_paths$label) {
  cat("\n", lbl, "\n", sep = "")
  sub <- path_table[path_table$Path == lbl, c("Subset", "Est_std", "SE", "z", "p",
                                              "CI_lower", "CI_upper")]
  sub$sig <- ifelse(sub$p < 0.001, "***",
             ifelse(sub$p < 0.01, "**",
             ifelse(sub$p < 0.05, "*", "")))
  print(sub, digits = 3, row.names = FALSE)
}

# ---- Mediation / cascade effects -------------------------------------------

cat("\n-----------------------------------------------------------------------\n")
cat("Defined effects (mediation / cascade), unstandardized with 95% CI\n")
cat("-----------------------------------------------------------------------\n")

defined_labels <- c("ind_sal_dmn", "total_sal_dmn", "cascade_sal_srs", "cascade_sal_cog")

get_defined_rows <- function(fit, label) {
  if (is.null(fit) || !lavInspect(fit, "converged")) return(NULL)

  pe <- parameterEstimates(fit, ci = TRUE)
  pe <- pe[pe$op == ":=", ]
  if (nrow(pe) == 0) return(NULL)

  data.frame(
    Subset   = label,
    Effect   = pe$label,
    Est      = pe$est,
    SE       = pe$se,
    z        = pe$z,
    p        = pe$pvalue,
    CI_lower = pe$ci.lower,
    CI_upper = pe$ci.upper,
    stringsAsFactors = FALSE
  )
}

defined_table <- do.call(rbind, lapply(names(fits_med),
                                       function(nm) get_defined_rows(fits_med[[nm]], nm)))
rownames(defined_table) <- NULL

if (!is.null(defined_table)) {
  for (eff in defined_labels) {
    cat("\n", eff, "\n", sep = "")
    sub <- defined_table[defined_table$Effect == eff,
                         c("Subset", "Est", "SE", "z", "p", "CI_lower", "CI_upper")]
    sub$sig <- ifelse(sub$p < 0.001, "***",
               ifelse(sub$p < 0.01, "**",
               ifelse(sub$p < 0.05, "*", "")))
    print(sub, digits = 3, row.names = FALSE)
  }
} else {
  cat("  No defined effects available (mediation models failed to fit).\n")
}

# ---- Full summary of each subtype fit --------------------------------------

for (nm in names(fits_med)) {
  if (is.null(fits_med[[nm]])) next
  cat("\n=======================================================================\n")
  cat("MODEL 1 SUMMARY (with defined effects) --", nm, "\n")
  cat("=======================================================================\n")
  print(summary(fits_med[[nm]], fit.measures = TRUE, standardized = TRUE, ci = TRUE))
}

# =============================================================================
# MULTI-GROUP INVARIANCE TEST (ASD ONLY)
# =============================================================================

cat("\n[5/5] Between-subtype test (ASD only, multi-group)\n")
cat("=======================================================================\n")

mg_table <- NULL

if (run_multigroup) {
  asd_file <- file.path(data_folder, "sem_input_asd_only.csv")

  if (!file.exists(asd_file)) {
    cat("  sem_input_asd_only.csv not found; skipping.\n")
  } else {
    asd_data <- read.csv(asd_file, row.names = 1)
    asd_data$cluster <- factor(asd_data$cluster)

    cat("  ASD-only sample:", nrow(asd_data), "subjects\n")
    print(table(asd_data$cluster))

    cat("\n  Why ASD only: the subtype tables above share their TD subjects, so a\n")
    cat("  multi-group model over them would enter each TD subject twice. The\n")
    cat("  ASD-only groups are disjoint, so this is the defensible test of\n")
    cat("  whether the Model 1 paths differ between subtypes.\n")

    min_n <- min(table(asd_data$cluster))
    if (min_n < 60) {
      cat("\n  CAUTION: smallest group n =", min_n, ". A 9-indicator, 2-factor model\n")
      cat("  at this n is underpowered; a non-significant invariance test is weak\n")
      cat("  evidence of equivalence, not proof of it.\n")
    }

    # Configural: same structure, all parameters free across subtypes
    fit_cfg <- tryCatch(
      sem(model1, data = asd_data, group = "cluster"),
      error = function(e) { cat("  ERROR (configural):", conditionMessage(e), "\n"); NULL }
    )

    # Metric: factor loadings held equal
    fit_metric <- tryCatch(
      sem(model1, data = asd_data, group = "cluster",
          group.equal = c("loadings")),
      error = function(e) { cat("  ERROR (metric):", conditionMessage(e), "\n"); NULL }
    )

    # Structural: loadings + regression paths held equal -- the hypothesis of
    # interest, i.e. the Model 1 paths are the same in both subtypes
    fit_struct <- tryCatch(
      sem(model1, data = asd_data, group = "cluster",
          group.equal = c("loadings", "regressions")),
      error = function(e) { cat("  ERROR (structural):", conditionMessage(e), "\n"); NULL }
    )

    if (!is.null(fit_cfg) && !is.null(fit_metric) && !is.null(fit_struct)) {
      cat("\n  Fit of each invariance level:\n")
      mg_table <- rbind(
        get_fit_row(fit_cfg,    "configural"),
        get_fit_row(fit_metric, "metric (loadings)"),
        get_fit_row(fit_struct, "structural (loadings + regressions)")
      )
      rownames(mg_table) <- NULL
      print(mg_table, digits = 4, row.names = FALSE)

      cat("\n  Chi-square difference tests (a significant p means the constrained\n")
      cat("  model fits WORSE, i.e. that set of parameters differs by subtype):\n")

      # Call lavTestLRT with the fits named directly. Passing them via
      # do.call() makes lavaan deparse each entire fit object into the row
      # names, which balloons the console output to megabytes.
      lrt <- lavTestLRT(fit_cfg, fit_metric, fit_struct)
      rownames(lrt) <- c("configural", "metric (loadings)",
                         "structural (loadings + regressions)")
      print(lrt)

      cat("\n  Per-subtype standardized structural paths (configural model):\n")
      std <- standardizedSolution(fit_cfg)
      std_struct <- std[std$op == "~" & std$lhs %in% c("avg_cen", "avg_dmn",
                                                       "cog_beh", "srs_beh"), ]
      print(std_struct[, c("lhs", "op", "rhs", "group", "est.std",
                           "se", "z", "pvalue")],
            digits = 3, row.names = FALSE)
    } else {
      cat("\n  Not enough converged multi-group fits to compare.\n")
    }
  }
} else {
  cat("  Skipped (run_multigroup = FALSE).\n")
}

# =============================================================================
# SAVE RESULTS
# =============================================================================

if (save_outputs) {
  cat("\n=======================================================================\n")
  cat("SAVING RESULTS\n")
  cat("=======================================================================\n")

  write.csv(fit_table, file.path(data_folder, "subtype_model1_fit_measures.csv"),
            row.names = FALSE)
  cat("  Saved: subtype_model1_fit_measures.csv\n")

  if (!is.null(path_table)) {
    write.csv(path_table, file.path(data_folder, "subtype_model1_paths_std.csv"),
              row.names = FALSE)
    cat("  Saved: subtype_model1_paths_std.csv\n")
  }

  if (!is.null(defined_table)) {
    write.csv(defined_table, file.path(data_folder, "subtype_model1_defined_effects.csv"),
              row.names = FALSE)
    cat("  Saved: subtype_model1_defined_effects.csv\n")
  }

  if (!is.null(mg_table)) {
    write.csv(mg_table, file.path(data_folder, "subtype_model1_multigroup_fit.csv"),
              row.names = FALSE)
    cat("  Saved: subtype_model1_multigroup_fit.csv\n")
  }

  cat("\n  Output folder:", data_folder, "\n")
}

cat("\n=======================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("=======================================================================\n")
cat("How to read this:\n")
cat("  - Fit indices (CFI/TLI/RMSEA/SRMR) similar across subsets  -> Model 1\n")
cat("    describes both subtypes as well as it describes the pooled sample.\n")
cat("  - Path estimates with overlapping CIs across subsets       -> the\n")
cat("    structural pattern is reproduced, not driven by one subtype.\n")
cat("  - Non-significant chi-square difference in Section 5       -> no\n")
cat("    detectable between-subtype difference in the paths (subject to the\n")
cat("    power caveat printed above).\n")
cat("=======================================================================\n")
