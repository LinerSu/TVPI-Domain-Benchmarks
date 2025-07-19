#!/bin/bash

# This script is designed to run all benchmark tests.
#
# Before executing, ensure that all necessary configuration settings have been properly set up in your environment.
# If you are not using a Docker container, make sure to follow the instructions in docker/Dockerfile.build
# to set up Seahorn (BMC) and the frameworks for verifying AWS code and Firedancer.
#
# Usage:
#   ./run_all_experiments.sh
#
# The script assumes that the necessary AWS resources and permissions are in place.
# It is intended to validate the artifact for the paper:
# "Template TVPI: A New Weakly Relational Domain for Efficient Memory-Access Validation"


################################################################################
# Run aws experiments                                                          #
################################################################################
echo "Running AWS experiments..."
./run_aws_benchs.sh

################################################################################
# Run Firedancer experiments                                                   #
################################################################################
echo "Running Firedancer experiments..."
./run_firedancer_benchs.sh