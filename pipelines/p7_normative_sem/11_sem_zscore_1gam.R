# 11_sem_zscore_1gam.R
# Fit GAM to TD data and calculate Z-SCORES for all subjects
#
# This script:
# 1. Loads raw triple network scores from 11_sem_zscore_0save_raw.py
# 2. Fits GAM models to TD data for each variable
# 3. Calculates Z-SCORES (not percentiles) for TD and ASD subjects
# 4. Saves z-scores as CSV for next step
#
# Why z-scores instead of percentiles:
# - SEM assumes linear relationships - z-scores preserve linearity
# - Averaging z-scores is mathematically valid
# - Z-scores preserve normal distribution (SEM assumption)
# - No floor/ceiling effects at extremes
#
# Useful commands:
# - clear console window : Ctrl + L
# - clear data           : rm(list = ls())

library(mgcv)
library(ggplot2)
library(dplyr)

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

# Parameters (matching 11_sem_zscore_0save_raw.py)
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

# Model selection parameter
smooth_k <- 3  # Number of knots for GAM spline (2: linear, 3: quadratic, 4: cubic)
delta_thres <- 2  # Threshold for AIC difference (non-linear must be better by at least this)

# Outlier threshold (MUST match Python CONFIG['outlier_threshold'])
outlier_threshold <- .p$subtyping$outlier_threshold# z-score threshold (options: 1.96(95%), 1.645(90%))

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
cat("11_sem_zscore_1gam: Fit GAM and Calculate Z-Scores\n")
cat("=======================================================================\n")
cat("\nData folder:", data_folder, "\n")

# =============================================================================
# LOAD RAW DATA
# =============================================================================

cat("\n[1/5] Loading raw triple network scores...\n")

sem_data_td <- read.csv(file.path(data_folder, "sem_data_td_raw.csv"), row.names = 1)
sem_data_asd <- read.csv(file.path(data_folder, "sem_data_asd_raw.csv"), row.names = 1)
dim_data_td <- read.csv(file.path(data_folder, "dim_data_td_raw.csv"), row.names = 1)
dim_data_asd <- read.csv(file.path(data_folder, "dim_data_asd_raw.csv"), row.names = 1)

cat("  Semantic TD:", nrow(sem_data_td), "subjects\n")
cat("  Semantic ASD:", nrow(sem_data_asd), "subjects\n")
cat("  Dimensionality TD:", nrow(dim_data_td), "subjects\n")
cat("  Dimensionality ASD:", nrow(dim_data_asd), "subjects\n")

# Variables to process
network_vars <- c("sal", "dmn", "cen")

# =============================================================================
# VISUAL INSPECTION - Plot GAM comparison (Linear vs Non-linear)
# =============================================================================

cat("\n[2/6] Visual inspection - Plotting sample networks...\n")

plot_gam_comparison <- function(var_name, data_td, data_asd, title_prefix, delta_thres = 2) {
  #' Plot GAM model comparison (linear vs non-linear) for TD and ASD groups
  #'
  #' @param var_name Character: variable name (e.g., "sal", "dmn", "cen")
  #' @param data_td DataFrame: TD data
  #' @param data_asd DataFrame: ASD data
  #' @param title_prefix Character: prefix for plot title (e.g., "Semantic", "Dimensional")
  #' @param delta_thres Numeric: AIC threshold for model selection

  # Define formulas
  form_linear <- as.formula(paste0(var_name, " ~ age + sex"))
  form_nonlinear <- as.formula(paste0(var_name, " ~ s(age, k=", smooth_k, ") + sex"))

  # Fit models for TD
  fit_td_lin <- gam(form_linear, family = gaussian(), data = data_td, method = "REML")
  fit_td_non <- gam(form_nonlinear, family = gaussian(), data = data_td, method = "REML")

  # Compare AIC for TD
  aic_td_lin <- AIC(fit_td_lin)
  aic_td_non <- AIC(fit_td_non)

  # Determine winner for TD
  if ((aic_td_lin - aic_td_non) > delta_thres) {
    winner_td <- "Non-Linear"
  } else {
    winner_td <- "Linear"
  }

  # Fit models for ASD
  fit_asd_lin <- gam(form_linear, family = gaussian(), data = data_asd, method = "REML")
  fit_asd_non <- gam(form_nonlinear, family = gaussian(), data = data_asd, method = "REML")

  # Compare AIC for ASD
  aic_asd_lin <- AIC(fit_asd_lin)
  aic_asd_non <- AIC(fit_asd_non)

  # Determine winner for ASD
  if ((aic_asd_lin - aic_asd_non) > delta_thres) {
    winner_asd <- "Non-Linear"
  } else {
    winner_asd <- "Linear"
  }

  # Create prediction grid
  age_seq <- seq(min(c(data_td$age, data_asd$age)),
                 max(c(data_td$age, data_asd$age)),
                 length.out = 100)

  pred_data <- data.frame(
    age = age_seq,
    sex = median(data_td$sex)
  )

  # Predict for both types
  p_td_lin <- predict(fit_td_lin, newdata = pred_data, se.fit = TRUE)
  p_td_non <- predict(fit_td_non, newdata = pred_data, se.fit = TRUE)
  p_asd_lin <- predict(fit_asd_lin, newdata = pred_data, se.fit = TRUE)
  p_asd_non <- predict(fit_asd_non, newdata = pred_data, se.fit = TRUE)

  # Prepare plot dataframes
  make_plot_df <- function(preds, grp, type) {
    data.frame(
      age = age_seq,
      fit = preds$fit,
      lower = preds$fit - 1.96 * preds$se.fit,
      upper = preds$fit + 1.96 * preds$se.fit,
      group = grp,
      model_type = type
    )
  }

  plot_df <- rbind(
    make_plot_df(p_td_lin, "TD", "Linear"),
    make_plot_df(p_td_non, "TD", "Non-Linear"),
    make_plot_df(p_asd_lin, "ASD", "Linear"),
    make_plot_df(p_asd_non, "ASD", "Non-Linear")
  )

  # Add observed data
  obs_data <- rbind(
    data.frame(age = data_td$age, y = data_td[[var_name]], group = "TD"),
    data.frame(age = data_asd$age, y = data_asd[[var_name]], group = "ASD")
  )

  # Network label mapping
  network_labels <- c(sal = "Salience", dmn = "Default Mode", cen = "Central Executive")
  var_label <- network_labels[var_name]

  # Plot
  p <- ggplot() +
    geom_point(data = obs_data, aes(x = age, y = y, color = group), alpha = 0.3, size = 1.5) +

    # Plot Lines: Solid for Non-Linear, Dashed for Linear
    geom_line(data = plot_df, aes(x = age, y = fit, color = group, linetype = model_type), size = 1) +

    # Plot Ribbons: Only for the WINNING model type for each group
    geom_ribbon(data = subset(plot_df, (group == "TD" & model_type == winner_td) |
                                       (group == "ASD" & model_type == winner_asd)),
                aes(x = age, ymin = lower, ymax = upper, fill = group), alpha = 0.2) +

    scale_color_manual(values = c("TD" = "#2E86AB", "ASD" = "#A23B72")) +
    scale_fill_manual(values = c("TD" = "#2E86AB", "ASD" = "#A23B72")) +
    labs(
      title = paste0(title_prefix, ": ", var_label, " Network (", var_name, ")"),
      subtitle = paste0("TD: ", winner_td, " (ΔAIC: ", round(aic_td_lin - aic_td_non, 2),
                       "), ASD: ", winner_asd, " (ΔAIC: ", round(aic_asd_lin - aic_asd_non, 2), ")"),
      x = "Age (years)",
      y = "Network Score"
    ) +
    theme_bw(base_size = 12) +
    theme(legend.position = "bottom")

  print(p)

  cat("\n--- Comparison for ", title_prefix, " ", var_name, " ---\n")
  cat("TD Group:\n")
  cat("  AIC Linear: ", round(aic_td_lin, 2), " | AIC Non-Linear: ", round(aic_td_non, 2), "\n")
  cat("  Delta AIC: ", round(aic_td_lin - aic_td_non, 2), "\n")
  cat("  Selected: ", winner_td, "\n")
  cat("\nASD Group:\n")
  cat("  AIC Linear: ", round(aic_asd_lin, 2), " | AIC Non-Linear: ", round(aic_asd_non, 2), "\n")
  cat("  Delta AIC: ", round(aic_asd_lin - aic_asd_non, 2), "\n")
  cat("  Selected: ", winner_asd, "\n")
}

# Plot all 6 network-feature combinations
cat("\n  --- Semantic Networks ---\n")
for (var in network_vars) {
  plot_gam_comparison(var, sem_data_td, sem_data_asd, "Semantic", delta_thres)
}

cat("\n  --- Dimensionality Networks ---\n")
for (var in network_vars) {
  plot_gam_comparison(var, dim_data_td, dim_data_asd, "Dimensional", delta_thres)
}

cat("\nVisual inspection complete. Proceeding to calculate z-scores...\n\n")

# =============================================================================
# FUNCTION: Fit GAM and calculate z-scores
# =============================================================================

fit_gam_and_calculate_zscores <- function(var_name, data_td, data_asd, delta_thres = 2) {
  #' Fit GAM to TD data and calculate z-scores for both TD and ASD
  #'
  #' @param var_name Character: variable name (e.g., "sal", "dmn", "cen")
  #' @param data_td DataFrame: TD data with var_name, age, sex columns
  #' @param data_asd DataFrame: ASD data with var_name, age, sex columns
  #' @param delta_thres Numeric: AIC threshold for selecting non-linear model
  #' @return List with z_score_td, z_score_asd, model_info

  cat("  Processing:", var_name, "\n")

  # Define formulas
  form_linear <- as.formula(paste0(var_name, " ~ age + sex"))
  form_nonlinear <- as.formula(paste0(var_name, " ~ s(age, k=", smooth_k, ") + sex"))

  # Fit models to TD data
  fit_lin <- gam(form_linear, family = gaussian(), data = data_td, method = "REML")
  fit_non <- gam(form_nonlinear, family = gaussian(), data = data_td, method = "REML")

  # Compare AIC
  aic_lin <- AIC(fit_lin)
  aic_non <- AIC(fit_non)
  delta_aic <- aic_lin - aic_non

  # Select model (non-linear must be better by delta_thres)
  if (delta_aic > delta_thres) {
    fit_selected <- fit_non
    model_type <- "nonlinear"
  } else {
    fit_selected <- fit_lin
    model_type <- "linear"
  }

  cat("    Model selected:", model_type, "(Delta AIC:", round(delta_aic, 2), ")\n")

  # Get sigma from selected model (residual standard deviation)
  sigma <- sqrt(fit_selected$sig2)

  # Calculate z-scores for TD subjects
  pred_td <- predict(fit_selected, newdata = data_td, type = "response")
  mu_td <- as.numeric(pred_td)
  y_td <- data_td[[var_name]]
  z_td <- (y_td - mu_td) / sigma

  # Calculate z-scores for ASD subjects
  pred_asd <- predict(fit_selected, newdata = data_asd, type = "response")
  mu_asd <- as.numeric(pred_asd)
  y_asd <- data_asd[[var_name]]
  z_asd <- (y_asd - mu_asd) / sigma

  # Model info
  model_info <- list(
    var_name = var_name,
    model_type = model_type,
    aic_linear = aic_lin,
    aic_nonlinear = aic_non,
    delta_aic = delta_aic,
    sigma = sigma,
    r_sq = summary(fit_selected)$r.sq
  )

  return(list(
    z_score_td = z_td,
    z_score_asd = z_asd,
    mu_td = mu_td,
    mu_asd = mu_asd,
    model_info = model_info
  ))
}

# =============================================================================
# FIT GAM AND CALCULATE Z-SCORES FOR ALL VARIABLES
# =============================================================================

cat("\n[3/6] Fitting GAM models and calculating z-scores...\n")

# Initialize result dataframes
zscore_td_df <- data.frame(row.names = rownames(sem_data_td))
zscore_asd_df <- data.frame(row.names = rownames(sem_data_asd))

model_info_list <- list()

# Process SEMANTIC variables
cat("\n  --- Semantic Variables ---\n")
for (var in network_vars) {
  var_name <- var
  result <- fit_gam_and_calculate_zscores(var_name, sem_data_td, sem_data_asd, delta_thres)

  # Store z-scores with "sem_" prefix
  col_name <- paste0("sem_", var)
  zscore_td_df[[col_name]] <- result$z_score_td
  zscore_asd_df[[col_name]] <- result$z_score_asd

  # Store model info
  model_info_list[[col_name]] <- result$model_info
}

# Process DIMENSIONALITY variables
cat("\n  --- Dimensionality Variables ---\n")
for (var in network_vars) {
  var_name <- var
  result <- fit_gam_and_calculate_zscores(var_name, dim_data_td, dim_data_asd, delta_thres)

  # Store z-scores with "dim_" prefix
  col_name <- paste0("dim_", var)
  zscore_td_df[[col_name]] <- result$z_score_td
  zscore_asd_df[[col_name]] <- result$z_score_asd

  # Store model info
  model_info_list[[col_name]] <- result$model_info
}

# =============================================================================
# ADD DEMOGRAPHICS TO Z-SCORE DATAFRAMES
# =============================================================================

cat("\n[4/6] Adding demographics to z-score dataframes...\n")

# Add demographics from original data
zscore_td_df$age <- sem_data_td$age
zscore_td_df$sex <- sem_data_td$sex
zscore_asd_df$age <- sem_data_asd$age
zscore_asd_df$sex <- sem_data_asd$sex

# Add group indicator
zscore_td_df$DX <- "TD"
zscore_asd_df$DX <- "ASD"

cat("  TD z-score dataframe:", nrow(zscore_td_df), "x", ncol(zscore_td_df), "\n")
cat("  ASD z-score dataframe:", nrow(zscore_asd_df), "x", ncol(zscore_asd_df), "\n")

# =============================================================================
# SAVE Z-SCORES
# =============================================================================

cat("\n[5/6] Saving z-scores...\n")

write.csv(zscore_td_df, file.path(data_folder, "zscore_td.csv"))
write.csv(zscore_asd_df, file.path(data_folder, "zscore_asd.csv"))

cat("  Saved: zscore_td.csv\n")
cat("  Saved: zscore_asd.csv\n")

# =============================================================================
# SAVE TRAJECTORY PREDICTIONS FOR PLOTTING
# =============================================================================

cat("\n[6/7] Saving trajectory predictions for plotting...\n")

save_trajectory_predictions <- function(data_td, data_asd, feature_prefix, delta_thres = 2) {
  #' Generate trajectory predictions for plotting
  #'
  #' @param data_td DataFrame: TD data with network scores, age, sex

  #' @param data_asd DataFrame: ASD data with network scores, age, sex
  #' @param feature_prefix Character: "sem" or "dim"
  #' @param delta_thres Numeric: AIC threshold for model selection
  #' @return List with td and asd dataframes containing predictions

  network_vars <- c("sal", "dmn", "cen")

  # Create age sequence spanning both groups
  age_min <- min(c(data_td$age, data_asd$age))
  age_max <- max(c(data_td$age, data_asd$age))
  age_seq <- seq(age_min, age_max, length.out = 100)

  # Initialize result dataframes
  result_td <- data.frame(age = age_seq)
  result_asd <- data.frame(age = age_seq)

  for (var in network_vars) {
    # Fit GAM models (same logic as z-score fitting)
    form_linear <- as.formula(paste0(var, " ~ age + sex"))
    form_nonlinear <- as.formula(paste0(var, " ~ s(age, k=", smooth_k, ") + sex"))

    # TD model
    fit_td_lin <- gam(form_linear, family = gaussian(), data = data_td, method = "REML")
    fit_td_non <- gam(form_nonlinear, family = gaussian(), data = data_td, method = "REML")
    fit_td <- if ((AIC(fit_td_lin) - AIC(fit_td_non)) > delta_thres) fit_td_non else fit_td_lin

    # ASD model
    fit_asd_lin <- gam(form_linear, family = gaussian(), data = data_asd, method = "REML")
    fit_asd_non <- gam(form_nonlinear, family = gaussian(), data = data_asd, method = "REML")
    fit_asd <- if ((AIC(fit_asd_lin) - AIC(fit_asd_non)) > delta_thres) fit_asd_non else fit_asd_lin

    # Predict for male (sex=0)
    pred_data <- data.frame(age = age_seq, sex = 0)

    # TD predictions
    p_td <- predict(fit_td, newdata = pred_data, se.fit = TRUE)
    sigma_td <- sqrt(fit_td$sig2)
    result_td[[paste0(var, "_mu")]] <- p_td$fit
    result_td[[paste0(var, "_se")]] <- p_td$se.fit
    result_td[[paste0(var, "_sigma")]] <- sigma_td
    result_td[[paste0(var, "_ci_lower")]] <- p_td$fit - 1.96 * p_td$se.fit
    result_td[[paste0(var, "_ci_upper")]] <- p_td$fit + 1.96 * p_td$se.fit
    result_td[[paste0(var, "_pi_lower")]] <- p_td$fit - outlier_threshold * sigma_td
    result_td[[paste0(var, "_pi_upper")]] <- p_td$fit + outlier_threshold * sigma_td

    # ASD predictions
    p_asd <- predict(fit_asd, newdata = pred_data, se.fit = TRUE)
    sigma_asd <- sqrt(fit_asd$sig2)
    result_asd[[paste0(var, "_mu")]] <- p_asd$fit
    result_asd[[paste0(var, "_se")]] <- p_asd$se.fit
    result_asd[[paste0(var, "_sigma")]] <- sigma_asd
    result_asd[[paste0(var, "_ci_lower")]] <- p_asd$fit - 1.96 * p_asd$se.fit
    result_asd[[paste0(var, "_ci_upper")]] <- p_asd$fit + 1.96 * p_asd$se.fit
    result_asd[[paste0(var, "_pi_lower")]] <- p_asd$fit - outlier_threshold * sigma_asd
    result_asd[[paste0(var, "_pi_upper")]] <- p_asd$fit + outlier_threshold * sigma_asd
  }

  return(list(td = result_td, asd = result_asd))
}

# Generate and save semantic trajectories
sem_traj <- save_trajectory_predictions(sem_data_td, sem_data_asd, "sem", delta_thres)
write.csv(sem_traj$td, file.path(data_folder, "gam_trajectory_td_male_sem.csv"), row.names = FALSE)
write.csv(sem_traj$asd, file.path(data_folder, "gam_trajectory_asd_male_sem.csv"), row.names = FALSE)
cat("  Saved: gam_trajectory_td_male_sem.csv\n")
cat("  Saved: gam_trajectory_asd_male_sem.csv\n")

# Generate and save dimensional trajectories
dim_traj <- save_trajectory_predictions(dim_data_td, dim_data_asd, "dim", delta_thres)
write.csv(dim_traj$td, file.path(data_folder, "gam_trajectory_td_male_dim.csv"), row.names = FALSE)
write.csv(dim_traj$asd, file.path(data_folder, "gam_trajectory_asd_male_dim.csv"), row.names = FALSE)
cat("  Saved: gam_trajectory_td_male_dim.csv\n")
cat("  Saved: gam_trajectory_asd_male_dim.csv\n")

# =============================================================================
# PRINT MODEL SELECTION SUMMARY
# =============================================================================

cat("\n[7/7] Model Selection Summary\n")
cat("=======================================================================\n")
cat("Delta AIC Threshold:", delta_thres, "(non-linear selected if Delta AIC >", delta_thres, ")\n\n")

model_summary_df <- do.call(rbind, lapply(model_info_list, function(info) {
  data.frame(
    Variable = info$var_name,
    Model = info$model_type,
    AIC_Linear = round(info$aic_linear, 2),
    AIC_NonLinear = round(info$aic_nonlinear, 2),
    Delta_AIC = round(info$delta_aic, 2),
    Sigma = round(info$sigma, 4),
    R_sq = round(info$r_sq, 4),
    stringsAsFactors = FALSE
  )
}))

# Fix variable names to include sem_/dim_ prefix
model_summary_df$Variable <- names(model_info_list)

print(model_summary_df)

# Count model types
n_linear <- sum(model_summary_df$Model == "linear")
n_nonlinear <- sum(model_summary_df$Model == "nonlinear")
cat("\nLinear models:", n_linear, "/ Non-linear models:", n_nonlinear, "\n")

# =============================================================================
# VISUALIZE Z-SCORE DISTRIBUTIONS
# =============================================================================

cat("\n=======================================================================\n")
cat("Z-Score Distribution Summary\n")
cat("=======================================================================\n")

zscore_vars <- c("sem_sal", "sem_dmn", "sem_cen", "dim_sal", "dim_dmn", "dim_cen")

cat("\nTD Z-Scores (should be ~N(0,1) if model fits well):\n")
for (var in zscore_vars) {
  cat(sprintf("  %s: mean=%.2f, sd=%.2f, range=[%.2f, %.2f]\n",
              var,
              mean(zscore_td_df[[var]]),
              sd(zscore_td_df[[var]]),
              min(zscore_td_df[[var]]),
              max(zscore_td_df[[var]])))
}

cat("\nASD Z-Scores (deviation from TD norm):\n")
for (var in zscore_vars) {
  cat(sprintf("  %s: mean=%.2f, sd=%.2f, range=[%.2f, %.2f]\n",
              var,
              mean(zscore_asd_df[[var]]),
              sd(zscore_asd_df[[var]]),
              min(zscore_asd_df[[var]]),
              max(zscore_asd_df[[var]])))
}

# Check for potential issues (ASD mean different from 0)
cat("\nASD deviation from TD norm (z=0 = TD average):\n")
for (var in zscore_vars) {
  mean_diff <- mean(zscore_asd_df[[var]])
  direction <- ifelse(mean_diff > 0, "higher", "lower")
  cat(sprintf("  %s: ASD is %.2f SD %s than TD average\n",
              var, abs(mean_diff), direction))
}

cat("\n=======================================================================\n")
cat("Done!\n")
cat("  - Z-scores saved: zscore_td.csv, zscore_asd.csv\n")
cat("  - Trajectories saved: gam_trajectory_*.csv (for plotting)\n")
cat("  - Next step: Run 11_sem_zscore_2save_data.py to combine with behavioral data\n")
cat("=======================================================================\n")
