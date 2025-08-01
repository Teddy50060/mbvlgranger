# Single Dataset Comparison Script
# Compare Your Frequency-Band Method vs VL-Granger vs Normal Granger vs VL-TE vs TE vs Granger-Geweke
# on famous datasets like Old Faithful Geyser, Gas Furnace, etc.

library(VLTimeCausality)
library(R.matlab)
library(RTransferEntropy)
library(grangers)  # For Granger-Geweke causality

# PLACEHOLDER: Replace with your actual file path
dataset_file <- "data/OldFFGeyserData.mat" 

cat("SINGLE DATASET CAUSALITY COMPARISON\n")
cat("===================================\n")

# Main analysis function
analyze_single_dataset <- function(filepath) {
  cat(sprintf("Analyzing: %s\n", basename(filepath)))
  cat("=" %R% 50, "\n\n")
  
  # Check if file exists
  if (!file.exists(filepath)) {
    stop(sprintf("File not found: %s", filepath))
  }
  
  # Load dataset
  cat("Loading dataset...\n")
  mat_data <- readMat(filepath)
  
  # Extract time series (adjust field names as needed)
  if ("x" %in% names(mat_data) && "y" %in% names(mat_data)) {
    X <- as.vector(mat_data$x)
    Y <- as.vector(mat_data$y)
  } else if ("OldFFGeyser" %in% names(mat_data)) {
    # For Old Faithful Geyser format
    X <- as.vector(mat_data$OldFFGeyser[1,])
    Y <- as.vector(mat_data$OldFFGeyser[2,])
  } 
  else if("gasfurnace" %in% names(mat_data)){
    X <- as.vector(mat_data$gasfurnace[1,])
    Y <- as.vector(mat_data$gasfurnace[2,])
  } 
  else if("eeg" %in% names(mat_data)){
    X <- as.vector(mat_data$eeg[1,])
    Y <- as.vector(mat_data$eeg[2,])
  } 
  else {
    # Show available fields for debugging
    cat("Available fields in .mat file:\n")
    print(names(mat_data))
    stop("Cannot find X and Y time series. Please check field names.")
  }
  
  cat(sprintf("X length: %d, Y length: %d\n", length(X), length(Y)))
  
  # Check for minimum length
  if (length(X) < 50 || length(Y) < 50) {
    stop("Time series too short for analysis (minimum 50 samples required)")
  }
  
  cat("\nRunning causality analyses...\n")
  cat("-" %R% 30, "\n")
  
  # Initialize results
  results <- list()
  
  # 1. VL-Granger
  cat("1. Testing VL-Granger...\n")
  tryCatch({
    vl_granger_result <- VLGrangerFunc(Y = Y, X = X, maxLag=60)
    results$vl_granger <- list(
      success = TRUE,
      causality = vl_granger_result$XgCsY,
      bic_ratio = vl_granger_result$BICDiffRatio,
      p_value = vl_granger_result$p.val,
      method = "VL-Granger"
    )
    cat(sprintf("   Result: %s (BIC ratio: %.3f, p-value: %.3f)\n", 
                vl_granger_result$XgCsY, vl_granger_result$BICDiffRatio, vl_granger_result$p.val))
  }, error = function(e) {
    results$vl_granger <- list(success = FALSE, error = as.character(e))
    cat(sprintf("   Error: %s\n", e$message))
  })
  
  # 2. Normal Granger
  cat("2. Testing Normal Granger...\n")
  tryCatch({
    granger_result <- GrangerFunc(Y = Y, X = X, maxLag=60)
    results$granger <- list(
      success = TRUE,
      causality = granger_result$XgCsY,
      bic_ratio = granger_result$BICDiffRatio,
      p_value = granger_result$p.val,
      method = "Normal Granger"
    )
    cat(sprintf("   Result: %s (BIC ratio: %.3f, p-value: %.3f)\n", 
                granger_result$XgCsY, granger_result$BICDiffRatio, granger_result$p.val))
  }, error = function(e) {
    results$granger <- list(success = FALSE, error = as.character(e))
    cat(sprintf("   Error: %s\n", e$message))
  })
  
  # 3. Granger-Geweke
  cat("3. Testing Granger-Geweke...\n")
  tryCatch({
    # Calculate adaptive max lag based on series length
    adaptive_max_lag <- min(
      max(10, floor(length(X)/25)),
      50
    )
    
    geweke_result <- Granger.unconditional(
      x = X,  # Cause variable 
      y = Y,  # Effect variable
      ic.chosen = "SC",
      max.lag = adaptive_max_lag,
      plot = FALSE,
      type.chosen = "const",
      p = 0
    )
    
    # Extract causality measures
    geweke_x_to_y <- sum(geweke_result$Unconditional_causality_x.to.y, na.rm = TRUE)
    geweke_y_to_x <- sum(geweke_result$Unconditional_causality_y.to.x, na.rm = TRUE)
    geweke_causality <- geweke_x_to_y > geweke_y_to_x
    geweke_ratio <- if(geweke_y_to_x > 0) geweke_x_to_y / geweke_y_to_x else Inf
    
    results$geweke <- list(
      success = TRUE,
      causality = geweke_causality,
      spectral_ratio = geweke_ratio,
      x_to_y = geweke_x_to_y,
      y_to_x = geweke_y_to_x,
      method = "Granger-Geweke"
    )
    cat(sprintf("   Result: %s (Spectral ratio: %.3f, X->Y: %.3f, Y->X: %.3f)\n", 
                geweke_causality, geweke_ratio, geweke_x_to_y, geweke_y_to_x))
  }, error = function(e) {
    results$geweke <- list(success = FALSE, error = as.character(e))
    cat(sprintf("   Error: %s\n", e$message))
  })
  
  # 4. VL-Transfer Entropy
  cat("4. Testing VL-Transfer Entropy...\n")
  tryCatch({
    vl_te_result <- VLTransferEntropy(Y = Y, X = X, maxLag=60)
    results$vl_te <- list(
      success = TRUE,
      causality = vl_te_result$XgCsY_trns,
      te_ratio = vl_te_result$TEratio,
      p_value = vl_te_result$pval,
      method = "VL-Transfer Entropy"
    )
    cat(sprintf("   Result: %s (TE ratio: %.3f, p-value: %.3f)\n", 
                vl_te_result$XgCsY_trns, vl_te_result$TEratio, 
                ifelse(is.null(vl_te_result$pval), NA, vl_te_result$pval)))
  }, error = function(e) {
    results$vl_te <- list(success = FALSE, error = as.character(e))
    cat(sprintf("   Error: %s\n", e$message))
  })
  
  # 5. Normal Transfer Entropy
  cat("5. Testing Normal Transfer Entropy...\n")
  tryCatch({
    te_result <- transfer_entropy(x = X, y = Y, nboot = 0, quiet = TRUE)
    te_ratio <- te_result$coef[1] / te_result$coef[2]  # TE(X->Y) / TE(Y->X)
    te_causality <- !is.na(te_ratio) && te_ratio > 1
    
    results$te <- list(
      success = TRUE,
      causality = te_causality,
      te_ratio = te_ratio,
      te_xy = te_result$coef[1],
      te_yx = te_result$coef[2],
      method = "Normal Transfer Entropy"
    )
    cat(sprintf("   Result: %s (TE ratio: %.3f, TE(X->Y): %.3f, TE(Y->X): %.3f)\n", 
                te_causality, te_ratio, te_result$coef[1], te_result$coef[2]))
  }, error = function(e) {
    results$te <- list(success = FALSE, error = as.character(e))
    cat(sprintf("   Error: %s\n", e$message))
  })
  
  return(results)
}

# Summary function
print_summary <- function(results, dataset_name) {
  cat("\n")
  cat("=" %R% 70, "\n")
  cat(sprintf("SUMMARY FOR %s\n", toupper(dataset_name)))
  cat("=" %R% 70, "\n")
  
  cat(sprintf("%-25s | %-10s | %-20s\n", "Method", "Causality", "Details"))
  cat("-" %R% 60, "\n")
  
  methods <- c("vl_granger", "granger", "geweke", "vl_te", "te")
  method_names <- c("VL-Granger", "Normal Granger", "Granger-Geweke", "VL-TE", "Normal TE")
  
  for (i in seq_along(methods)) {
    method <- methods[i]
    name <- method_names[i]
    
    if (method %in% names(results) && results[[method]]$success) {
      causality <- results[[method]]$causality
      
      # Create details string based on method
      if (method %in% c("vl_granger", "granger")) {
        details <- sprintf("BIC: %.3f", results[[method]]$bic_ratio)
      } else if (method == "geweke") {
        details <- sprintf("Spectral: %.3f", results[[method]]$spectral_ratio)
      } else {  # Transfer entropy methods
        details <- sprintf("Ratio: %.3f", results[[method]]$te_ratio)
      }
      
      cat(sprintf("%-25s | %-10s | %-20s\n", name, causality, details))
    } else {
      cat(sprintf("%-25s | %-10s | %-20s\n", name, "ERROR", "Failed"))
    }
  }
  
  cat("\n")
  
  # Agreement analysis
  successful_results <- results[sapply(results, function(x) x$success)]
  if (length(successful_results) > 1) {
    causalities <- sapply(successful_results, function(x) x$causality)
    agree_count <- sum(causalities)
    total_count <- length(causalities)
    
    cat("AGREEMENT ANALYSIS:\n")
    cat("-" %R% 20, "\n")
    cat(sprintf("Methods detecting causality: %d/%d\n", agree_count, total_count))
    if (agree_count == total_count) {
      cat("All methods agree: CAUSALITY detected\n")
    } else if (agree_count == 0) {
      cat("All methods agree: NO CAUSALITY detected\n")
    } else {
      cat("Methods disagree on causality\n")
      for (name in names(successful_results)) {
        result_char <- if (successful_results[[name]]$causality) "YES" else "NO"
        cat(sprintf("  %s: %s\n", successful_results[[name]]$method, result_char))
      }
    }
  }
}

# Helper function
`%R%` <- function(x, n) {
  paste(rep(x, n), collapse = "")
}
if (dataset_file == "..") {
  cat("Please update the dataset_file variable with your actual file path\n")
  cat("\nExample usage:\n")
  cat('dataset_file <- "data/OldFFGeyserData.mat"\n')
  cat('results <- analyze_single_dataset(dataset_file)\n')
  cat('print_summary(results, "Old Faithful Geyser")\n')
} else {
  # Run analysis if path is updated
  results <- analyze_single_dataset(dataset_file)
  dataset_name <- tools::file_path_sans_ext(basename(dataset_file))
  print_summary(results, dataset_name)
}