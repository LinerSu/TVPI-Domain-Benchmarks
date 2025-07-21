#!/bin/bash

# Set the debug level (info, debug)
DEBUG_LEVEL=${1:-info}  # Default to info if not provided

# Function to print debug messages
debug() {
    if [ "$DEBUG_LEVEL" == "debug" ]; then
        echo "$@"
    fi
}

DIVIDER="\n================================================\n\n"
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
VSTTE_RES_DIR="$SCRIPT_DIR/firedancer/res/vstte"
if [ ! -d "$VSTTE_RES_DIR" ]; then
    mkdir -p "$VSTTE_RES_DIR"
fi

# Function to run experiments
run_experiment() {
    local domain=$1
    printf $DIVIDER
    echo "Running AI4BMC with the numerical domain $domain..."
    printf $DIVIDER
    if [ "$DEBUG_LEVEL" == "debug" ]; then
        python3 get_exper_res.py --seahorn --seahorn-root $SEAHORN_ROOT --timeout 600 --bleed_edge --debug --crab --domain $domain
    else
        python3 get_exper_res.py --seahorn --seahorn-root $SEAHORN_ROOT --timeout 600 --bleed_edge --crab --domain $domain
    fi
    printf $DIVIDER
    echo "Done running AI4BMC with the numerical domain $domain."
    printf $DIVIDER
}

################################################################################
# Operations for Firedancer experiments                                        #
################################################################################
echo "mkdir -p /tmp/results/firedancer/crab"
mkdir -p /tmp/results/firedancer/crab

# This script is designed to run firedancer benchmark tests.
echo "cd $SCRIPT_DIR/firedancer/scripts"
cd firedancer/scripts

# Run the experiments
# 1. Run experiment for the zones domain (in crab, we called split dbm)
run_experiment "zones"
# 2. Run experiment for the template TVPI domain (in crab, we called fixed tvpi dbm)
run_experiment "fixed-tvpi-dbm"
# 3. Run experiment for the convex polyhedra domain (we used Elina version)
run_experiment "pk"

# Gather statistics
printf "\n\n================================================\n"
echo "                 STATISTICS                 "
printf "================================================\n\n"