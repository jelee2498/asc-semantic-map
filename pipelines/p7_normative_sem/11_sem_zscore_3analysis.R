# 11_sem_zscore_3analysis.R
# SEM analysis for z-score-based triple network data
#
# This script:
# 1. Loads z-score-based data from 11_sem_zscore_2save_data.py
# 2. Uses averaged network scores (avg_sal, avg_dmn, avg_cen) computed from:
#    - avg_sal = (sem_sal + dim_sal) / 2
#    - avg_dmn = (-1 * sem_dmn + dim_dmn) / 2  # sem_dmn flipped
#    - avg_cen = (sem_cen + dim_cen) / 2
# 3. Runs 4 competing SEM models
# 4. Compares model fit and reports results
#
# Key difference from 11_sem_1analysis.R:
# - Uses z-scores (normalized deviation from TD norm) instead of raw residualized scores
# - Z-scores preserve linearity (SEM assumption) and normal distribution
# - sem_dmn is flipped with -1 before averaging to align direction with dim_dmn
# - This makes averaging valid because both measures are on same scale and direction
#
# Useful commands:
# - clear console window : Ctrl + L
# - clear data           : rm(list = ls())

library(lavaan)
library(semhelpinghands)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths and parameters come from config/{paths,params}.yml via
# lib/project_config.R.  The original working tree hard-coded `store7 <- "S:/"`
# (or "/MIPL/store7") here and repeated every analysis parameter as a literal,
# which is the exact drift risk the shared config exists to prevent: the Python
# and R halves of this chain meet through a directory name built from these
# values, and a mismatch reads an empty directory rather than erroring.
# Locate lib/ without depending on how this script was launched: Rscript passes
# --file=, source() sets ofile, RStudio has neither.
.asc_here <- local({
  a <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(a) > 0) return(dirname(sub("^--file=", "", a[1])))
  for (i in seq_len(sys.nframe())) {
    o <- sys.frame(i)$ofile
    if (!is.null(o)) return(dirname(o))
  }
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- tryCatch(rstudioapi::getSourceEditorContext()$path, error = function(e) "")
    if (nzchar(p)) return(dirname(p))
  }
  getwd()
})
.lib <- normalizePath(file.path(.asc_here, "..", "..", "lib"), mustWork = TRUE)
source(file.path(.lib, "project_config.R"))
.p <- asc_params()

proj_root <- asc_project()

# Parameters (matching previous scripts)
proj <- "02_asd_semantic_map"
pipe <- "99_main"
task <- "11_sem_zscore"
conf_option <- .p$denoising$conf_option
atlas <- .p$parcellation$atlas
chunk_option <- .p$encoding$chunk_option
srm_option_pc <- .p$encoding$srm_option
srm_option_dim <- .p$encoding$srm_option_dim
bias_weight <- .p$features$bias_weight
verb_weight <- .p$features$verb_weight
reg_wordnet <- .p$features$reg_wordnet
enc_single_alpha <- .p$encoding$enc_single_alpha_semantic
perf_method <- .p$performance_mask$perf_method
perf_alpha <- .p$performance_mask$perf_alpha
weight_across_delays <- .p$features$weight_across_delays
weight_add_superordinate <- .p$features$weight_add_superordinate
screen <- .p$normative_sem$screen# Age-based screening (matching 0save_raw.py)
sig_overlap <- .p$normative_sem$sig_overlap# Use overlap significance mask (matching 0save_raw.py)

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

data_folder <- file.path(proj_root, "2_pipeline", pipe, task, "out",
                         conf_option, atlas,
                         paste0("chunk-", chunk_option),
                         "fold-avg", "results",
                         paste0("k-", srm_option_pc, "_", srm_option_dim),
                         paste0("bias-", fmt_num(bias_weight), "_verb-", fmt_num(verb_weight)),
                         paste0("wn-", fmt_bool(reg_wordnet)),
                         paste0("s_alpha-", fmt_bool(enc_single_alpha)),
                         "sem",
                         paste0("mask_", perf_method, "-", perf_alpha),
                         paste0("delay-", weight_across_delays, "_super-", fmt_bool(weight_add_superordinate)),
                         paste0("screen-", fmt_bool(screen), "_overlap-", fmt_bool(sig_overlap)))

cat("=======================================================================\n")
cat("11_sem_zscore_3analysis: SEM Analysis on Z-Score-Based Data\n")
cat("=======================================================================\n")
cat("\nData folder:", data_folder, "\n")

# =============================================================================
# LOAD DATA
# =============================================================================

cat("\nLoading z-score-based data...\n")

triple_data <- read.csv(file.path(data_folder, "all_sem_dim_zscore.csv"), row.names = 1)

cat("Data dimensions:", nrow(triple_data), "subjects x", ncol(triple_data), "variables\n")
cat("Variables:", paste(colnames(triple_data), collapse = ", "), "\n")

cat("\nFirst few rows:\n")
print(head(triple_data))

# =============================================================================
# VERIFY AVERAGED NETWORK SCORES
# =============================================================================

cat("\n=======================================================================\n")
cat("Z-SCORE-BASED AVERAGED NETWORK SCORES\n")
cat("=======================================================================\n")
cat("These scores were computed in Python with sem_dmn flipping:\n")
cat("  avg_sal = (sem_sal + dim_sal) / 2\n")
cat("  avg_dmn = (-1 * sem_dmn + dim_dmn) / 2  # sem_dmn flipped\n")
cat("  avg_cen = (sem_cen + dim_cen) / 2\n\n")

cat("Summary of averaged scores:\n")
print(summary(triple_data[, c("avg_sal", "avg_dmn", "avg_cen")]))

# =============================================================================
# DEFINE SEM MODELS
# =============================================================================
# Four competing models for triple network -> behavior relationships
# Using averaged z-score-based network scores

# Model 1: SAL -> CEN, SAL+CEN -> DMN, DMN -> both behaviors
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

# Model 2: SAL -> CEN, SAL+CEN -> DMN, CEN -> cog, DMN -> srs
model2 <- '
  # Measurement model
  srs_beh =~ SRS_AWR_T + SRS_COG_T + SRS_COM_T + SRS_MOT_T + SRS_RRB_T
  cog_beh =~ NIH7_Card_P + NIH7_Flanker_P + NIH7_List_P + NIH7_Pattern_P

  # Structural model
  avg_cen ~ avg_sal
  avg_dmn ~ avg_sal + avg_cen
  cog_beh ~ avg_cen
  srs_beh ~ avg_dmn
  srs_beh ~~ 0*cog_beh
'

# Model 3: SAL -> DMN, SAL+DMN -> CEN, CEN -> both behaviors
model3 <- '
  # Measurement model
  srs_beh =~ SRS_AWR_T + SRS_COG_T + SRS_COM_T + SRS_MOT_T + SRS_RRB_T
  cog_beh =~ NIH7_Card_P + NIH7_Flanker_P + NIH7_List_P + NIH7_Pattern_P

  # Structural model
  avg_dmn ~ avg_sal
  avg_cen ~ avg_sal + avg_dmn
  cog_beh ~ avg_cen
  srs_beh ~ avg_cen
  srs_beh ~~ 0*cog_beh
'

# Model 4: SAL -> both CEN and DMN (independent), DMN -> both behaviors
model4 <- '
  # Measurement model
  srs_beh =~ SRS_AWR_T + SRS_COG_T + SRS_COM_T + SRS_MOT_T + SRS_RRB_T
  cog_beh =~ NIH7_Card_P + NIH7_Flanker_P + NIH7_List_P + NIH7_Pattern_P

  # Structural model
  avg_cen ~ avg_sal
  avg_dmn ~ avg_sal
  cog_beh ~ avg_dmn
  srs_beh ~ avg_dmn
  srs_beh ~~ 0*cog_beh
  avg_cen ~~ 0*avg_dmn
'

# =============================================================================
# FIT MODELS
# =============================================================================

cat("\n=======================================================================\n")
cat("FITTING SEM MODELS (Z-SCORE-BASED)\n")
cat("=======================================================================\n")

fit1 <- sem(model1, data = triple_data)
fit2 <- sem(model2, data = triple_data)
fit3 <- sem(model3, data = triple_data)
fit4 <- sem(model4, data = triple_data)

# =============================================================================
# MODEL COMPARISON
# =============================================================================

cat("\n=======================================================================\n")
cat("MODEL FIT COMPARISON\n")
cat("=======================================================================\n")

# Create fit measures table
fit_measures <- data.frame(
  Model = c("Model1", "Model2", "Model3", "Model4"),
  AIC   = c(fitMeasures(fit1, "aic"),
            fitMeasures(fit2, "aic"),
            fitMeasures(fit3, "aic"),
            fitMeasures(fit4, "aic")),
  BIC   = c(fitMeasures(fit1, "bic"),
            fitMeasures(fit2, "bic"),
            fitMeasures(fit3, "bic"),
            fitMeasures(fit4, "bic")),
  Chisq = c(fitMeasures(fit1, "chisq"),
            fitMeasures(fit2, "chisq"),
            fitMeasures(fit3, "chisq"),
            fitMeasures(fit4, "chisq")),
  pvalue = c(fitMeasures(fit1, "pvalue"),
             fitMeasures(fit2, "pvalue"),
             fitMeasures(fit3, "pvalue"),
             fitMeasures(fit4, "pvalue")),
  CFI  = c(fitMeasures(fit1, "cfi"),
           fitMeasures(fit2, "cfi"),
           fitMeasures(fit3, "cfi"),
           fitMeasures(fit4, "cfi")),
  TLI  = c(fitMeasures(fit1, "tli"),
           fitMeasures(fit2, "tli"),
           fitMeasures(fit3, "tli"),
           fitMeasures(fit4, "tli")),
  RMSEA = c(fitMeasures(fit1, "rmsea"),
            fitMeasures(fit2, "rmsea"),
            fitMeasures(fit3, "rmsea"),
            fitMeasures(fit4, "rmsea")),
  SRMR = c(fitMeasures(fit1, "srmr"),
           fitMeasures(fit2, "srmr"),
           fitMeasures(fit3, "srmr"),
           fitMeasures(fit4, "srmr"))
)

print(fit_measures)

# =============================================================================
# BEST MODEL SELECTION
# =============================================================================

cat("\n=======================================================================\n")
cat("BEST MODEL FOR EACH FIT MEASURE\n")
cat("=======================================================================\n")
cat("(AIC, BIC, Chisq, RMSEA, SRMR: lower is better; pvalue, CFI, TLI: higher is better)\n\n")

for (measure in colnames(fit_measures)) {
  if (measure == "Model") {
    next
  } else if (measure %in% c("AIC", "BIC", "Chisq", "RMSEA", "SRMR")) {
    best_model <- fit_measures[which.min(fit_measures[, measure]), "Model"]
    best_value <- min(fit_measures[, measure])
    cat(sprintf("%s: %s (%.3f)\n", measure, best_model, best_value))
  } else {
    best_model <- fit_measures[which.max(fit_measures[, measure]), "Model"]
    best_value <- max(fit_measures[, measure])
    cat(sprintf("%s: %s (%.3f)\n", measure, best_model, best_value))
  }
}

# =============================================================================
# DETAILED MODEL SUMMARIES
# =============================================================================

cat("\n=======================================================================\n")
cat("MODEL 1 SUMMARY (SAL->CEN, SAL+CEN->DMN, DMN->behaviors)\n")
cat("=======================================================================\n")
summary(fit1, fit.measures = TRUE, standardized = TRUE)

cat("\n=======================================================================\n")
cat("MODEL 2 SUMMARY (SAL->CEN, SAL+CEN->DMN, CEN->cog, DMN->srs)\n")
cat("=======================================================================\n")
summary(fit2, fit.measures = TRUE, standardized = TRUE)

cat("\n=======================================================================\n")
cat("MODEL 3 SUMMARY (SAL->DMN, SAL+DMN->CEN, CEN->behaviors)\n")
cat("=======================================================================\n")
summary(fit3, fit.measures = TRUE, standardized = TRUE)

cat("\n=======================================================================\n")
cat("MODEL 4 SUMMARY (SAL->CEN/DMN independent, DMN->behaviors)\n")
cat("=======================================================================\n")
summary(fit4, fit.measures = TRUE, standardized = TRUE)

# =============================================================================
# MODIFICATION INDICES
# =============================================================================

cat("\n=======================================================================\n")
cat("MODIFICATION INDICES\n")
cat("=======================================================================\n")

mod_indices1 <- modindices(fit1)
mod_indices2 <- modindices(fit2)
mod_indices3 <- modindices(fit3)
mod_indices4 <- modindices(fit4)

mod_indices1 <- mod_indices1[order(-mod_indices1$mi), ]
mod_indices2 <- mod_indices2[order(-mod_indices2$mi), ]
mod_indices3 <- mod_indices3[order(-mod_indices3$mi), ]
mod_indices4 <- mod_indices4[order(-mod_indices4$mi), ]

cat("\nTop modification indices for Model 1:\n")
print(head(mod_indices1))

cat("\nTop modification indices for Model 2:\n")
print(head(mod_indices2))

cat("\nTop modification indices for Model 3:\n")
print(head(mod_indices3))

cat("\nTop modification indices for Model 4:\n")
print(head(mod_indices4))

# =============================================================================
# GROUP-SPECIFIC SAMPLE SIZES
# =============================================================================

cat("\n=======================================================================\n")
cat("GROUP-SPECIFIC SAMPLE SIZES\n")
cat("=======================================================================\n")

if ("DX" %in% colnames(triple_data)) {
  cat("TD subjects:", sum(triple_data$DX == "TD"), "\n")
  cat("ASD subjects:", sum(triple_data$DX == "ASD"), "\n")
}

# =============================================================================
# MEDIATION ANALYSIS (MODEL 1)
# =============================================================================

# Model 1 with Defined Indirect Effects
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

  # Defined Parameters (The "Paper-Ready" Metrics)
  # 1. Indirect effect of SAL on DMN via CEN
  ind_sal_dmn := a*b

  # 2. Total effect of SAL on DMN (Direct + Indirect)
  total_sal_dmn := c + (a*b)

  # 3. Cascading effect: SAL -> CEN -> DMN -> SRS
  cascade_sal_srs := a*b*d

  # 4. Cascading effect: SAL -> CEN -> DMN -> COG
  cascade_sal_cog := a*b*e
'

fit1_med <- sem(model1_med, data = triple_data)
summary(fit1_med, standardized = TRUE, ci = TRUE) # Added CI for publication

# =============================================================================
# PERSIST RESULTS
# =============================================================================
#
# The original printed everything to the console and saved nothing, so the
# Fig. 5 fit indices and path coefficients existed only in a terminal
# scrollback.  Write them out, mirroring what 12_heterogeneity_2sem_zscore.R
# already does for the subtype fits, so the reported numbers can be checked
# against a file rather than re-read off a screen.

cat("\n=======================================================================\n")
cat("WRITING RESULTS\n")
cat("=======================================================================\n")

fits <- list(fit1, fit2, fit3, fit4)

# 1. Fit indices for all four competing models (Fig. 5 + Supp. Fig. 8-10)
fit_out <- fit_measures
fit_out$N  <- sapply(fits, lavaan::nobs)
fit_out$df <- sapply(fits, function(f) unname(fitMeasures(f, "df")))
fit_out$Chisq_df <- fit_out$Chisq / fit_out$df
fit_out <- fit_out[, c("Model", "N", "Chisq", "df", "Chisq_df",
                       setdiff(colnames(fit_measures), c("Model", "Chisq")))]
write.csv(fit_out, file.path(data_folder, "sem_model_fit_measures.csv"), row.names = FALSE)
cat("  sem_model_fit_measures.csv\n")

# 2. Path coefficients of the selected model (Model 1).
#
# Report every variant, because they differ and the figure uses only one.
# With all observed variables z-scored, lavaan's unstandardized `est` equals
# `std.lv`; `std.all` additionally rescales by the latent variances and is NOT
# what Fig. 5 shows.  Fig. 5's beta values are the `est` column
# (e.g. SAL->CEN = 0.684, where std.all = 0.557).
paths_all <- parameterEstimates(fit1_med, standardized = TRUE)
write.csv(paths_all, file.path(data_folder, "sem_model1_paths_std.csv"), row.names = FALSE)
cat("  sem_model1_paths_std.csv\n")

# 3. Defined mediation / cascade effects (ind_sal_dmn, cascade_sal_srs, ...)
defined <- subset(parameterEstimates(fit1_med, standardized = TRUE), op == ":=")
write.csv(defined, file.path(data_folder, "sem_model1_defined_effects.csv"), row.names = FALSE)
cat("  sem_model1_defined_effects.csv\n")

# 4. The headline numbers, as one row, for direct comparison with Fig. 5
sel <- fit_out[fit_out$Model == "Model1", ]
reg <- subset(paths_all, op == "~")
grab <- function(lhs_, rhs_, col) {
  r <- reg[reg$lhs == lhs_ & reg$rhs == rhs_, ]
  if (nrow(r) == 0) NA_real_ else r[[col]][1]
}
PATHS <- list(sal_to_cen = c("avg_cen", "avg_sal"),
              cen_to_dmn = c("avg_dmn", "avg_cen"),
              sal_to_dmn = c("avg_dmn", "avg_sal"),
              dmn_to_srs = c("srs_beh", "avg_dmn"),
              dmn_to_cog = c("cog_beh", "avg_dmn"))
headline <- data.frame(
  N = sel$N, chisq = sel$Chisq, df = sel$df, chisq_df = sel$Chisq_df,
  RMSEA = sel$RMSEA, CFI = sel$CFI, TLI = sel$TLI, SRMR = sel$SRMR
)
for (nm in names(PATHS)) {
  lr <- PATHS[[nm]]
  headline[[paste0(nm, "_beta")]]    <- grab(lr[1], lr[2], "est")     # Fig. 5
  headline[[paste0(nm, "_z")]]       <- grab(lr[1], lr[2], "z")       # Fig. 5
  headline[[paste0(nm, "_p")]]       <- grab(lr[1], lr[2], "pvalue")
  headline[[paste0(nm, "_std_all")]] <- grab(lr[1], lr[2], "std.all")
}
write.csv(headline, file.path(data_folder, "sem_model1_headline.csv"), row.names = FALSE)
cat("  sem_model1_headline.csv\n")

cat("\nModel 1 (Fig. 5): chi2 =", round(headline$chisq, 3),
    " df =", headline$df, " chi2/df =", round(headline$chisq_df, 3),
    " RMSEA =", round(headline$RMSEA, 4), " CFI =", round(headline$CFI, 4), "\n")
cat("  paths (beta, z):",
    sprintf("SAL->CEN %.3f (%.2f) | CEN->DMN %.3f (%.2f) | SAL->DMN %.3f (%.2f)",
            headline$sal_to_cen_beta, headline$sal_to_cen_z,
            headline$cen_to_dmn_beta, headline$cen_to_dmn_z,
            headline$sal_to_dmn_beta, headline$sal_to_dmn_z), "\n")
cat("                  ",
    sprintf("DMN->SRS %.3f (%.2f) | DMN->cog %.3f (%.2f)",
            headline$dmn_to_srs_beta, headline$dmn_to_srs_z,
            headline$dmn_to_cog_beta, headline$dmn_to_cog_z), "\n")

cat("\n=======================================================================\n")
cat("ANALYSIS COMPLETE\n")
cat("=======================================================================\n")
