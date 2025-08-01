# CSV Causality Testing Script
# Compare VL-Granger vs Normal Granger vs Granger-Geweke vs VL-TE vs TE on CSV datasets
# Handles various CSV formats including sunspot data

library(VLTimeCausality)
library(RTransferEntropy)
library(R.matlab)
library(grangers)  # For Granger-Geweke causality

dataset_file <- "data/eegdata_Fc5._Fc3.csv"

cat("CSV DATASET CAUSALITY COMPARISON\n")
cat("=================================\n")

# Helper function
`%R%` <- function(x, n) {
  paste(rep(x, n), collapse = "")
}

# Data quality check function
check_data_quality <- function(X, Y) {
  cat("\nDATA QUALITY DIAGNOSTICS:\n")
  cat("-" %R% 25, "\n")
  
  # Basic stats
  cat(sprintf("X: length=%d, range=[%.6f, %.6f]\n", length(X), min(X, na.rm=TRUE), max(X, na.rm=TRUE)))
  cat(sprintf("Y: length=%d, range=[%.6f, %.6f]\n", length(Y), min(Y, na.rm=TRUE), max(Y, na.rm=TRUE)))
  
  # Check for problematic values
  x_issues <- sum(is.na(X) | is.infinite(X) | is.nan(X))
  y_issues <- sum(is.na(Y) | is.infinite(Y) | is.nan(Y))
  cat(sprintf("X issues (NA/Inf/NaN): %d\n", x_issues))
  cat(sprintf("Y issues (NA/Inf/NaN): %d\n", y_issues))
  
  # Variance check
  x_var <- var(X, na.rm = TRUE)
  y_var <- var(Y, na.rm = TRUE)
  cat(sprintf("X variance: %.6f\n", x_var))
  cat(sprintf("Y variance: %.6f\n", y_var))
  
  # Check for constant series
  x_unique <- length(unique(X[!is.na(X)]))
  y_unique <- length(unique(Y[!is.na(Y)]))
  cat(sprintf("X unique values: %d\n", x_unique))
  cat(sprintf("Y unique values: %d\n", y_unique))
  
  # Correlation
  correlation <- cor(X, Y, use = "complete.obs")
  cat(sprintf("Correlation: %.6f\n", correlation))
  
  # Check data types
  cat(sprintf("X class: %s\n", class(X)))
  cat(sprintf("Y class: %s\n", class(Y)))
  
  # Sample values
  cat("First 5 X values:", paste(head(X, 5), collapse=", "), "\n")
  cat("First 5 Y values:", paste(head(Y, 5), collapse=", "), "\n")
  
  # Check for potential issues
  issues <- c()
  if (x_var < 1e-10 || y_var < 1e-10) issues <- c(issues, "Near-constant series")
  if (x_issues > 0 || y_issues > 0) issues <- c(issues, "Invalid values")
  if (x_unique < 10 || y_unique < 10) issues <- c(issues, "Very few unique values")
  if (abs(correlation) > 0.999) issues <- c(issues, "Extremely high correlation")
  
  if (length(issues) > 0) {
    cat("POTENTIAL ISSUES:", paste(issues, collapse=", "), "\n")
  } else {
    cat("Data appears clean\n")
  }
  
  return(list(
    has_issues = length(issues) > 0,
    issues = issues
  ))
}

# Function to load and prepare CSV data
load_csv_data <- function(filepath) {
  cat("Loading CSV data...\n")
  
  # First try to read as normal CSV with different separators
  separators <- c(",", ";", "\t", " ")
  data <- NULL
  
  for (sep in separators) {
    tryCatch({
      data <- read.csv(filepath, header = FALSE, sep = sep, stringsAsFactors = FALSE)
      if (nrow(data) > 10 && ncol(data) >= 2) {
        cat(sprintf("Successfully loaded with separator '%s'\n", sep))
        break
      }
    }, error = function(e) {
      # Try next separator
    })
  }
  
  # If normal CSV reading failed, try manual parsing
  if (is.null(data) || ncol(data) < 2) {
    cat("Normal CSV parsing failed, trying manual parsing...\n")
    
    # Read file as text lines
    lines <- readLines(filepath)
    cat(sprintf("Read %d lines from file\n", length(lines)))
    
    # Show sample line
    cat("Sample line:", lines[1], "\n")
    
    # Try to split each line by semicolon
    if (length(lines) > 0 && grepl(";", lines[1])) {
      cat("Detected semicolon-separated values within lines\n")
      
      # Split each line and extract columns
      split_lines <- strsplit(lines, ";")
      
      # Find maximum number of columns
      max_cols <- max(sapply(split_lines, length))
      cat(sprintf("Maximum columns found: %d\n", max_cols))
      
      # Create matrix
      data_matrix <- matrix(NA, nrow = length(lines), ncol = max_cols)
      
      for (i in seq_along(split_lines)) {
        row_data <- trimws(split_lines[[i]])  # Remove whitespace
        if (length(row_data) >= 2) {
          data_matrix[i, 1:length(row_data)] <- row_data
        }
      }
      
      # Convert to data frame
      data <- as.data.frame(data_matrix, stringsAsFactors = FALSE)
      cat(sprintf("Manually parsed data: %d rows, %d columns\n", nrow(data), ncol(data)))
      
    } else {
      stop("Could not parse CSV format")
    }
  }
  
  cat("Parsed CSV structure:\n")
  cat(sprintf("  Rows: %d, Columns: %d\n", nrow(data), ncol(data)))
  
  # Show first few rows for debugging
  cat("First few parsed rows:\n")
  print(head(data, 3))
  
  # Extract X and Y based on structure
  if (ncol(data) >= 3) {
    # Try to extract two numeric columns
    X_raw <- data[, 2]
    Y_raw <- data[, 3]
    
    cat(sprintf("Raw X sample: %s\n", paste(head(X_raw, 3), collapse=", ")))
    cat(sprintf("Raw Y sample: %s\n", paste(head(Y_raw, 3), collapse=", ")))
    
    # Convert to numeric, handling different formats
    X <- suppressWarnings(as.numeric(as.character(X_raw)))
    Y <- suppressWarnings(as.numeric(as.character(Y_raw)))
    
    cat(sprintf("Converted to numeric - X sample: %s\n", paste(head(X[!is.na(X)], 3), collapse=", ")))
    cat(sprintf("Converted to numeric - Y sample: %s\n", paste(head(Y[!is.na(Y)], 3), collapse=", ")))
    
  } else if (ncol(data) == 1) {
    # Single column - try to extract numbers
    values_raw <- data[, 1]
    
    # Try to extract numbers from the text
    # Look for the first number in each line
    values <- suppressWarnings(as.numeric(as.character(values_raw)))
    
    # If that fails, try to extract the first number from each string
    if (all(is.na(values))) {
      cat("Direct conversion failed, trying to extract first number from each line\n")
      values <- sapply(values_raw, function(x) {
        # Extract first number from string
        nums <- regmatches(x, gregexpr("[-+]?[0-9]*\\.?[0-9]+", x))[[1]]
        if (length(nums) > 0) {
          return(as.numeric(nums[1]))
        } else {
          return(NA)
        }
      })
    }
    
    # Remove NAs
    values <- values[!is.na(values)]
    
    if (length(values) < 2) {
      stop("Could not extract numeric values from single column")
    }
    
    # Create lag relationship
    X <- values[-length(values)]
    Y <- values[-1]
    
    cat(sprintf("Single column: extracted %d numeric values\n", length(values)))
    cat(sprintf("Sample values: %s\n", paste(head(values, 5), collapse=", ")))
    
  } else {
    stop("No data columns found")
  }
  
  # Count NAs before removal
  na_count_x <- sum(is.na(X))
  na_count_y <- sum(is.na(Y))
  
  if (na_count_x > 0 || na_count_y > 0) {
    cat(sprintf("Found %d NAs in X, %d NAs in Y - removing...\n", na_count_x, na_count_y))
  }
  
  # Remove any NA values
  valid_idx <- !is.na(X) & !is.na(Y)
  X <- X[valid_idx]
  Y <- Y[valid_idx]
  
  cat(sprintf("Final data: X length=%d, Y length=%d\n", length(X), length(Y)))
  
  if (length(X) > 0 && length(Y) > 0) {
    cat(sprintf("X range: [%.3f, %.3f]\n", min(X), max(X)))
    cat(sprintf("Y range: [%.3f, %.3f]\n", min(Y), max(Y)))
  }
  
  return(list(X = X, Y = Y))
}

# Main analysis function
analyze_csv_dataset <- function(filepath) {
  cat(sprintf("Analyzing: %s\n", basename(filepath)))
  cat("=" %R% 50, "\n\n")
  
  # Check if file exists
  if (!file.exists(filepath)) {
    stop(sprintf("File not found: %s", filepath))
  }
  
  # Load CSV data
  data_result <- tryCatch({
    load_csv_data(filepath)
  }, error = function(e) {
    stop(sprintf("Error loading CSV: %s", e$message))
  })
  
  X <- data_result$X
  Y <- data_result$Y
  
  # ADD DATA QUALITY CHECK HERE
  quality_check <- check_data_quality(X, Y)
  
  # Clean data if issues detected
  if (quality_check$has_issues) {
    cat("\nCLEANING DATA...\n")
    
    # Remove any infinite or NaN values
    valid_mask <- is.finite(X) & is.finite(Y) & !is.na(X) & !is.na(Y)
    X_clean <- X[valid_mask]
    Y_clean <- Y[valid_mask]
    
    cat(sprintf("Removed %d problematic samples\n", length(X) - length(X_clean)))
    
    # Check if we still have enough data
    if (length(X_clean) < 50) {
      stop("Too few valid samples after cleaning")
    }
    
    X <- X_clean
    Y <- Y_clean
    
    cat("Data after cleaning:\n")
    check_data_quality(X, Y)
  }
  
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
    vl_granger_result <- VLGrangerFunc(Y = Y, X = X, maxLag=50)
    results$vl_granger <- list(
      success = TRUE,
      causality = vl_granger_result$XgCsY,
      bic_ratio = vl_granger_result$BICDiffRatio,
      p_value = vl_granger_result$p.val,
      method = "VL-Granger"
    )
  }, error = function(e) {
    results$vl_granger <- list(success = FALSE, error = as.character(e))
    cat(sprintf("   Error: %s\n", e$message))
  })
  
  # 2. Normal Granger
  cat("2. Testing Normal Granger...\n")
  tryCatch({
    granger_result <- GrangerFunc(Y = Y, X = X, maxLag=50)
    results$granger <- list(
      success = TRUE,
      causality = granger_result$XgCsY,
      bic_ratio = granger_result$BICDiffRatio,
      p_value = granger_result$p.val,
      method = "Normal Granger"
    )
  }, error = function(e) {
    results$granger <- list(success = FALSE, error = as.character(e))
    cat(sprintf("   Error: %s\n", e$message))
  })
  
  # 3. Granger-Geweke with enhanced diagnostics
  cat("3. Testing Granger-Geweke...\n")
  
  # Pre-check data for Granger-Geweke specific issues
  x_range <- max(X) - min(X)
  y_range <- max(Y) - min(Y)
  cat(sprintf("   Data ranges: X=%.6f, Y=%.6f\n", x_range, y_range))
  
  if (x_range < 1e-10 || y_range < 1e-10) {
    cat("   Skipping Granger-Geweke: Near-constant time series\n")
    results$geweke <- list(
      success = FALSE, 
      error = "Near-constant time series unsuitable for spectral analysis",
      method = "Granger-Geweke"
    )
  } else {
    tryCatch({
      data_length <- length(X)
      
      # Conservative settings for CSV data
      if (data_length > 5000) {
        subsample_rate <- ceiling(data_length / 3000)
        subsample_idx <- seq(1, data_length, by = subsample_rate)
        X_analysis <- X[subsample_idx]
        Y_analysis <- Y[subsample_idx]
        cat(sprintf("   Subsampling: using every %d points (%d -> %d samples)\n", 
                    subsample_rate, data_length, length(X_analysis)))
      } else {
        X_analysis <- X
        Y_analysis <- Y
      }
      
      adaptive_max_lag <- min(8, floor(length(X_analysis)/50))
      cat(sprintf("   Using max.lag = %d\n", adaptive_max_lag))
      
      geweke_result <- Granger.unconditional(
        x = X_analysis, 
        y = Y_analysis, 
        ic.chosen = "SC",
        max.lag = 50,
        plot = FALSE,
        type.chosen = "const",
        p = 0
      )
      
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
      
    }, error = function(e) {
      results$geweke <- list(
        success = FALSE, 
        error = sprintf("Granger-Geweke failed: %s", as.character(e)),
        method = "Granger-Geweke"
      )
      cat(sprintf("   Error: %s\n", e$message))
    })
  }
  
  # 4. VL-Transfer Entropy
  cat("4. Testing VL-Transfer Entropy...\n")
  tryCatch({
    vl_te_result <- VLTransferEntropy(Y = Y, X = X, maxLag=50)
    results$vl_te <- list(
      success = TRUE,
      causality = vl_te_result$XgCsY_trns,
      te_ratio = vl_te_result$TEratio,
      p_value = vl_te_result$pval,
      method = "VL-Transfer Entropy"
    )
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

# MAIN EXECUTION
cat("CSV Causality Testing Script\n")
cat("============================\n\n")

# Usage instructions
cat("Supported CSV formats:\n")
cat("1. Two columns: X,Y (e.g., time,value or cause,effect)\n")
cat("2. Single column: value (creates lag relationship X=t, Y=t+1)\n")
cat("3. Multiple columns: uses first two columns\n\n")

# Example for sunspot data format you showed
cat("For sunspot data format (year;sunspot;...;...):\n")
cat("The script will use year as X and sunspot as Y\n\n")

# Run analysis
results <- analyze_csv_dataset(dataset_file)
dataset_name <- tools::file_path_sans_ext(basename(dataset_file))
print_summary(results, dataset_name)