#!/bin/bash

# Set the debug level (info, debug)
DEBUG_LEVEL=${1:-info}  # Default to info if not provided

# Function to print debug messages
debug() {
    if [ "$DEBUG_LEVEL" == "debug" ]; then
        echo "$@"
    fi
}

# This script is designed to run all benchmark tests.
#
# Before executing, ensure that all necessary configuration settings have been properly set up in your environment.
# If you are not using a Docker container, make sure to follow the instructions in docker/Dockerfile.build
# to set up Seahorn (BMC) and the frameworks for verifying AWS code and Firedancer.
#
# Usage:
#   ./run_all_experiments.sh info # or debug
#
# The script assumes that the necessary AWS resources and permissions are in place.
# It is intended to validate the artifact for the paper:
# "Template TVPI: A New Weakly Relational Domain for Efficient Memory-Access Validation"


################################################################################
# Run aws experiments                                                          #
################################################################################
echo "Running AWS experiments..."
./run_aws_benchs.sh $DEBUG_LEVEL

################################################################################
# Run Firedancer experiments                                                   #
################################################################################
echo "Running Firedancer experiments..."
./run_firedancer_benchs.sh $DEBUG_LEVEL


# Gather statistics
printf "\n\n================================================\n"
echo "                 STATISTICS                 "
printf "================================================\n\n"
python3 get_paper_results.py