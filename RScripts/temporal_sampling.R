# ------------------------------------------------------------------------------
# Temporal Sampling and Sliding Window Processing for Telemetry Data
#
# Description:
# This script creates temporal samples by scanning, sliding, and sub-sampling 
# time-series signals. It generates a new dataset for further analysis or use 
# in Python environments. Outputs are saved as CSV and RData files.
#
# @Ashkan Dashtban @2024
# ------------------------------------------------------------------------------

# Load required libraries
library(data.table)
library(dplyr)
library(yaml)
library(rio)

# Define utility functions
"%ni%" <- Negate("%in%")  # Negation of `%in%`

# Set project directories
project_dir <- './dashtban/TeleIoT-Fraud/'
setwd(project_dir)

# Load configuration
config <- yaml::read_yaml("config.yml")
source_data_dir <- config$path$sdata
processed_data_dir <- file.path(project_dir, "Data/Rdata/")

# Load processed data
load(file.path(processed_data_dir, "processed_data.Rdata"))

# ------------------------------------------------------------------------------
# Scan, Slide, and Oversample Event Signals
# ------------------------------------------------------------------------------
cat("Processing event signals...\n")

# Ensure data is ordered
setorderv(df_events, c("ID", "positions.ts"))

# Random sampling of signals
post_ind_r <- sample(df_events[positions.seconds_diff_from_event < -2,]$positions.seconds_diff_from_event, 10, replace = TRUE)
pre_ind_r <- sample(df_events[positions.seconds_diff_from_event > 2,]$positions.seconds_diff_from_event, 10, replace = TRUE)

# Balanced sampling
post_ind_b <- sample(df_events[positions.seconds_diff_from_event < -4 & positions.seconds_diff_from_event > -8,]$positions.seconds_diff_from_event, 10, replace = TRUE)
pre_ind_b <- sample(df_events[positions.seconds_diff_from_event > 4,]$positions.seconds_diff_from_event, 10, replace = TRUE)

# Generate combinations of sampling indices
combinations_r <- expand.grid(pre_ind_r, post_ind_r)
combinations_b <- expand.grid(pre_ind_b, post_ind_b)
combinations <- rbind(combinations_r, combinations_b)
combinations <- data.table(combinations)
combinations[, len := Var1 - Var2]

# Create sub-samples
df_events_sub <- data.table()
UID <- 1
for (i in unique(df_events$ID)) {
  for (j in 1:nrow(combinations)) {
    df_sub <- df_events[ID == i & 
                        positions.seconds_diff_from_event <= combinations[j,]$Var1 & 
                        positions.seconds_diff_from_event >= combinations[j,]$Var2]
    df_sub[, `:=`(IDS = j, UID = UID)]
    df_events_sub <- rbind(df_events_sub, df_sub)
    UID <- UID + 1
  }
}

# Add count of samples per `IDS`
df_events_sub[, n := .N, by = IDS]
cat("Event signal processing complete.\n")

# ------------------------------------------------------------------------------
# Scan, Slide, and Subsample Non-Event Signals
# ------------------------------------------------------------------------------
cat("Processing non-event signals...\n")

# Filter and order non-event data
df_noevent <- df[ID %ni% df_events$ID,]
setorderv(df_noevent, c("ID", "positions.ts"))

# Compute sampling parameters per ID
num_samples_per_ID <- sort(sample(50:150, length(unique(df_noevent$ID))))
num_step_len_per_ID <- df_noevent[, .(duration = max(as.numeric(duration))), by = ID]
setorder(num_step_len_per_ID, duration)
num_step_len_per_ID[, `:=`(num = num_samples_per_ID, 
                           step = pmax(ceiling(duration / num) - 1, 1),
                           len = ifelse(step < 14, 15, step + round(0.1 * step) + 1))]

# Sliding and subsampling
df_noevent_sub <- data.table()
for (i in num_step_len_per_ID$ID) {
  params <- num_step_len_per_ID[ID == i]
  base_pointer <- 1
  
  for (j in 1:params$num) {
    next_pointer <- base_pointer + params$len
    df_sub <- df_noevent[ID == i & sliding_pointer >= base_pointer & sliding_pointer < next_pointer,]
    df_sub[, `:=`(IDS = j, UID = UID)]
    df_noevent_sub <- rbind(df_noevent_sub, df_sub)
    
    base_pointer <- base_pointer + params$step
    if (base_pointer >= params$duration - 14) break
    UID <- UID + 1
  }
}

# Add count of samples per `IDS`
df_noevent_sub[, n := .N, by = IDS]
cat("Non-event signal processing complete.\n")

# ------------------------------------------------------------------------------
# Export Processed Data
# ------------------------------------------------------------------------------
cat("Exporting processed datasets...\n")

# Remove unnecessary columns
df_noevent_sub[, sliding_pointer := NULL]
df_events_sub[, n := NULL]
df_noevent_sub[, n := NULL]

# Add incident labels
df_events_sub[, incident := 1]
df_noevent_sub[, incident := 0]

# Combine event and non-event datasets
dfc <- rbind(df_events_sub, df_noevent_sub, fill = TRUE)

# Save to CSV and RData
rio::export(df_events_sub, file.path(processed_data_dir, "processed_data_with_events_scanned.csv"))
rio::export(df_noevent_sub, file.path(processed_data_dir, "processed_data_no_events_scanned.csv"))
rio::export(dfc, file.path(processed_data_dir, "processed_data_scanned.csv"))
save(df_events_sub, df_noevent_sub, dfc, file = file.path(processed_data_dir, "processed_data_scanned.Rdata"))

cat("All files exported successfully to:", processed_data_dir, "\n")
