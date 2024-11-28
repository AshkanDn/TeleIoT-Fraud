# ------------------------------------------------------------------------------
# Processing and Aggregation of Sensor-Generated JSON Files with Varying Features
#
# Description:
#   This script processes JSON files containing sensor data, extracts and pull features up, 
#   and creates an aggregated dataset. Finally, it generates a 
#   comprehensive HTML report summarizing the processing outcomes.
#
# @Ashkan Dashtban
# ------------------------------------------------------------------------------

# Load required packages and install any missing ones
install_packages <- function(package_list) {
  missing_packages <- package_list[!package_list %in% installed.packages()[, "Package"]]
  if (length(missing_packages) > 0) {
    install.packages(missing_packages, type = "source")
  }
}

options(install.packages.compile.from.source = "always")
required_packages <- c("data.table", "dplyr", "lubridate", "rio", "rmarkdown", "yaml")
install_packages(required_packages)

# Load libraries
library(data.table)
library(dplyr)
library(lubridate)
library(rio)
library(rmarkdown)
library(yaml)

# Define custom operator for "not in"
"%ni%" <- Negate("%in%")

# ------------------------------------------------------------------------------
# Initialise Project and Configuration
# ------------------------------------------------------------------------------

# Set the project directory and load configuration
project_dir <- "./dashtban/TeleIoT-Fraud/"
setwd(project_dir)
cat("Working directory set to:", project_dir, "\n")

# Load the YAML configuration file
config <- yaml::read_yaml("config.yml")
source_data_dir <- config$path$sdata
processed_data_dir <- file.path(project_dir, "Data/Rdata/")

# ------------------------------------------------------------------------------
# Function to Process JSON Files
# ------------------------------------------------------------------------------

process_json_files <- function(json_files, data_dir) {
  successful_files <- character()
  failed_files <- character()
  df <- data.table()
  file_counter <- 1
  
  for (file in json_files) {
    cat("Processing file:", file, "\n")
    tryCatch({
      # Read and process JSON file
      new_file <- jsonlite::fromJSON(file)
      new_file$ID <- file_counter
      
      # Convert JSON to data.table and bind rows
      new_data <- as.data.table(do.call(cbind, new_file))
      df <- rbind(df, new_data, fill = TRUE)
      
      file_counter <- max(df$ID) + 1
      successful_files <- c(successful_files, file)
    }, error = function(e) {
      failed_files <- c(failed_files, file)
      message("Error processing file '", file, "': ", e)
    })
  }
  
  # Process date columns
  date_columns <- c("receivedAt", "startedAt", "endedAt", "positions.ts")
  df[, (date_columns) := lapply(.SD, as.POSIXct, format = "%Y-%m-%dT%H:%M:%SZ"), .SDcols = date_columns]
  
  # Impute missing values and create features
  df[is.na(cleaned), cleaned := FALSE]
  df[, duration := difftime(max(positions.ts), min(positions.ts), units = "secs"), by = ID]
  
  # Categorize data into events and no-events
  incident_ids <- unique(df[!is.na(positions.seconds_diff_from_event)]$ID)
  df_events <- df[ID %in% incident_ids]
  
  # Remove columns with all NULL values in event data
  null_columns <- names(df_events)[colSums(is.na(df_events)) == nrow(df_events)]
  df_events <- df_events[, !null_columns, with = FALSE]
  
  # Add incident indicator and create no-event data
  df[, incident := ID %in% incident_ids]
  df_noevent <- df[ID %ni% incident_ids]
  
  # Export data
  rio::export(df_noevent, file.path(data_dir, "processed_data_no_events.csv"))
  rio::export(df_events, file.path(data_dir, "processed_data_with_events.csv"))
  rio::export(df, file.path(data_dir, "processed_data_complete.csv"))
  
  # Save RData file
  save(df, df_events, file = file.path(data_dir, "processed_data.Rdata"))
  
  # Generate HTML report
  generate_report(successful_files, failed_files, names(df), data_dir)
  
  return(list(df = df, df_noevent = df_noevent, df_events = df_events))
}

# ------------------------------------------------------------------------------
# Function to Generate HTML Report
# ------------------------------------------------------------------------------

generate_report <- function(successful_files, failed_files, feature_names, data_dir) {
  num_successful <- length(successful_files)
  num_failed <- length(failed_files)
  num_features <- length(feature_names)
  
  report_content <- paste0(
    "<h1>JSON File Processing Report</h1>",
    "<h2>Summary:</h2>",
    "<p>Total files: ", num_successful + num_failed, "</p>",
    "<p>Successfully processed: ", num_successful, "</p>",
    "<p>Failed to process: ", num_failed, "</p>",
    "<p>Total features: ", num_features, "</p>",
    "<h2>Successfully Processed Files:</h2>",
    if (num_successful > 0) paste0("<ul><li>", paste(successful_files, collapse = "</li><li>"), "</li></ul>") else "<p>None</p>",
    "<h2>Files with Errors:</h2>",
    if (num_failed > 0) paste0("<ul><li>", paste(failed_files, collapse = "</li><li>"), "</li></ul>") else "<p>None</p>",
    "<h2>Feature List:</h2>",
    "<ul><li>", paste(feature_names, collapse = "</li><li>"), "</li></ul>"
  )
  
  report_file <- file.path(data_dir, "processing_report.html")
  writeLines(report_content, report_file)
  browseURL(report_file)
}

# ------------------------------------------------------------------------------
# Main Script Execution
# ------------------------------------------------------------------------------

# Get list of JSON files and process them
json_files <- list.files(source_data_dir, pattern = "\\.json$", full.names = TRUE)
result <- process_json_files(json_files, processed_data_dir)

# Extract results
df <- result$df
df_noevent <- result$df_noevent
df_events <- result$df_events

cat("JSON processing complete. Processed data exported successfully.\n")

