import pandas as pd
import numpy as np
import copy
import math
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
import matplotlib.pyplot as plt
from termcolor import colored
from scipy.stats import gaussian_kde

# look at sample
IS_SAMPLE = True
SAMPLE_FOLDER = "sample/" if IS_SAMPLE else ""

# debugging
DEBUG = False

# do not limit number of rows displayed
pd.set_option("display.max_rows", None)

TIMEOUT_THRESHOLD = 600
# load csv file
AWS_CSV_PATH = "aws/res/data/" + SAMPLE_FOLDER
FIRE_DANCER_PATH = "firedancer/res/data/" + SAMPLE_FOLDER


output_domain_name_map = {
    "utvpi": "Zones",
    "template_tvpi": "Template TVPI",
    "pk": "PK",
}

csv_name_map = {
    "utvpi": "AI4BMC_zones_Z3",
    "template_tvpi": "AI4BMC_fixed-tvpi-dbm_Z3",
    "pk": "AI4BMC_pk_Z3",
}

configurations = {
    "pp_time": {"color": "tab:blue", "marker": "o", "name": "before unrolling"},
    "opsem_time": {"color": "tab:green", "marker": "x", "name": "after unrolling"},
}

dom_names = ["utvpi", "template_tvpi", "pk"]
categories = {
    "aws": ["array_list", "hash_table", "others"],
    "fire_dancer": ["tango", "util", "others"],
    "all": ["total"],
}


###########################################
#              Args functions             #
###########################################
def print_centered_title(title, single="*", width=34):
    """
    Prints a centered title within asterisks of the given width.

    :param title: The title to be centered and printed.
    :param width: The total width of the line (default is 34).
    """
    # Calculate the padding needed to center the title
    padding = (width - len(title)) // 2

    # Adjust padding for titles with odd lengths
    if len(title) % 2 != 0 and width % 2 == 0:
        title += " "

    # Print the formatted output
    print(single * width)
    print(" " * padding + title + " " * padding)
    print(single * width)


def second_distinct_max(df, cols):
    vals = df[cols].to_numpy().ravel()
    uniq = np.unique(vals)
    if uniq.size < 2:
        raise ValueError("Need at least two distinct values")
    return uniq[-2]


###########################################
#             Column functions            #
###########################################
def set_failed_cases(df):
    # df = df.fillna(0)
    # keep timeout cases before unrolling
    df["pp_to"] = np.where(df["pp_crab_time"].isna(), True, False)
    # kepp timeout cases after unrolling
    df["opsem_to"] = np.where(
        df["opsem_crab_time"].isna() & df["seahorn_total_time"].isna(), True, False
    )
    df["result"] = np.where(~df["seahorn_total_time"].isna(), "TRUE", df["result"])
    df["pp.isderef.not.solve"] = np.where(
        df["pp.isderef.not.solve"].isna() & ~df["pp_to"], 0, df["pp.isderef.not.solve"]
    )
    df["pp.isderef.solve"] = np.where(
        df["pp.isderef.solve"].isna() & ~df["pp_to"], 0, df["pp.isderef.solve"]
    )
    df["pp_crab_time"] = np.where(df["pp_to"], TIMEOUT_THRESHOLD, df["pp_crab_time"])
    df["pp_crab_time"] = np.where(
        df["pp_crab_time"].isna() & ~df["pp_to"], 0, df["pp_crab_time"]
    )
    df["opsem.isderef.not.solve"] = np.where(
        df["opsem.isderef.not.solve"].isna() & ~df["opsem_to"],
        0,
        df["opsem.isderef.not.solve"],
    )
    df["opsem.isderef.solve"] = np.where(
        df["opsem.isderef.solve"].isna() & ~df["opsem_to"], 0, df["opsem.isderef.solve"]
    )
    df["opsem_crab_time"] = np.where(
        df["opsem_to"], TIMEOUT_THRESHOLD, df["opsem_crab_time"]
    )
    df["opsem_crab_time"] = np.where(
        df["opsem_crab_time"].isna() & ~df["opsem_to"], 0, df["opsem_crab_time"]
    )
    df["opsem_to"] = np.where(df["opsem_to"] & ~df["pp_to"], True, df["opsem_to"])
    # after = df.columns.get_loc('result') + 1
    # df.iloc[:, after:] = df.iloc[:, after:].fillna(0)
    # df['result'] = np.where(df['result'] == 0, "FALSE", df['result'])
    return df


def clean_format(csv):
    csv["seahorn_total_time"] = csv["seahorn_total_time"].fillna(0).astype(float)
    csv["bmc_solve_time"] = csv["bmc_solve_time"].fillna(0).astype(float)
    csv["bmc_dag_size"] = csv["bmc_dag_size"].fillna(0).astype(int)
    csv["bmc_circuit_size"] = csv["bmc_circuit_size"].fillna(0).astype(int)
    csv["opsem_crab_time"] = csv["opsem_crab_time"].fillna(0).astype(float)
    csv["opsem_crab_range_time"] = csv["opsem_crab_range_time"].fillna(0).astype(float)
    csv["pp_crab_time"] = csv["pp_crab_time"].fillna(0).astype(float)
    csv["pp_crab_range_time"] = csv["pp_crab_range_time"].fillna(0).astype(float)
    csv["pp.isderef.not.solve"] = csv["pp.isderef.not.solve"].fillna(0).astype(int)
    csv["pp.isderef.solve"] = csv["pp.isderef.solve"].fillna(0).astype(int)
    csv["opsem.isderef.not.solve"] = (
        csv["opsem.isderef.not.solve"].fillna(0).astype(int)
    )
    csv["opsem.isderef.solve"] = csv["opsem.isderef.solve"].fillna(0).astype(int)
    return csv


def validate_timeout_statistics(df):
    bpp_to = df[df["pp_to"]]
    mask = (
        (bpp_to["pp_crab_time"] == TIMEOUT_THRESHOLD)
        & bpp_to["pp.isderef.not.solve"].isna()
        & bpp_to["pp.isderef.solve"].isna()
    )
    assert mask.any(), f"Found unexpected rows:\n{bpp_to[mask]['job_name']}"
    bpp_good = df[~df["pp_to"]]
    mask = (
        (bpp_good["pp_crab_time"] <= TIMEOUT_THRESHOLD)
        & (~bpp_good["pp.isderef.not.solve"].isna())
        & (~bpp_good["pp.isderef.solve"].isna())
    )
    assert mask.any(), f"Found unexpected rows:\n{bpp_good[mask]['job_name']}"
    app_to = df[df["opsem_to"]]
    mask = (
        (app_to["opsem_crab_time"] == TIMEOUT_THRESHOLD)
        & app_to["opsem.isderef.not.solve"].isna()
        & app_to["opsem.isderef.solve"].isna()
    )
    assert mask.any(), f"Found unexpected rows:\n{app_to[mask]['job_name']}"
    app_good = df[~df["opsem_to"]]
    mask = (
        (app_good["opsem_crab_time"] <= TIMEOUT_THRESHOLD)
        & (~app_good["opsem.isderef.not.solve"].isna())
        & (~app_good["opsem.isderef.solve"].isna())
    )
    assert mask.any(), f"Found unexpected rows:\n{app_good[mask]['job_name']}"


def check_timeout_cases_same(df1, df2):
    df1 = pd.DataFrame(df1)
    df2 = pd.DataFrame(df2)
    sorted_df1 = df1.sort_values(by="job_name").reset_index(drop=True)
    sorted_df2 = df2.sort_values(by="job_name").reset_index(drop=True)
    res = sorted_df1["job_name"].equals(sorted_df2["job_name"])
    diff_df1 = sorted_df1[~sorted_df1["job_name"].isin(sorted_df2["job_name"])]
    diff_df2 = sorted_df2[~sorted_df2["job_name"].isin(sorted_df1["job_name"])]
    return res, list(diff_df1["job_name"]), list(diff_df2["job_name"])


def process_csv(csv_name, repo_name):
    CRAB_CSV = f"{csv_name_map[csv_name]}.csv"
    CSV_PATH = AWS_CSV_PATH if repo_name == "aws" else FIRE_DANCER_PATH
    if DEBUG:
        print("Processing csv file: ", CRAB_CSV)
    crab_csv = set_failed_cases(pd.read_csv(CSV_PATH + CRAB_CSV))
    if DEBUG:
        print(f"Total cases from {repo_name}: {len(crab_csv)}")
    crab_csv = crab_csv.dropna(subset=["result"])
    # crab_csv = clean_format(crab_csv)
    validate_timeout_statistics(crab_csv)
    matches = crab_csv[crab_csv["job_name"] == "der"]
    crab = crab_csv[
        [
            "job_name",
            "pp_crab_time",
            "pp_crab_range_time",
            "pp.isderef.not.solve",
            "pp.isderef.solve",
            "opsem_crab_time",
            "opsem_crab_range_time",
            "opsem.isderef.not.solve",
            "opsem.isderef.solve",
            "bmc_solve_time",
            "seahorn_total_time",
            "bmc_circuit_size",
            "bmc_dag_size",
        ]
    ].rename(
        columns={
            "pp_crab_time": f"{csv_name}_pp_time",
            "opsem_crab_time": f"{csv_name}_opsem_time",
            "pp.isderef.not.solve": f"{csv_name}_pp_warn",
            "pp.isderef.solve": f"{csv_name}_pp_safe",
            "opsem.isderef.not.solve": f"{csv_name}_opsem_warn",
            "opsem.isderef.solve": f"{csv_name}_opsem_safe",
            "seahorn_total_time": f"{csv_name}_total_time",
        }
    )
    crab["repo"] = repo_name
    crab[f"{csv_name}_pp_to"] = crab_csv["pp_to"]
    crab[f"{csv_name}_opsem_to"] = crab_csv["opsem_to"]

    return crab


def compute_stats(crab, csv_name):
    crab = crab.copy()
    no_is_dref = crab[
        (~crab[f"{csv_name}_pp_to"])
        & (crab[f"{csv_name}_pp_warn"] == 0.0)
        & (crab[f"{csv_name}_pp_safe"] == 0.0)
    ]
    no_is_dref_lst = no_is_dref["job_name"].to_list()
    # exclude no isderef cases
    crab = crab[~crab["job_name"].isin(no_is_dref_lst)]
    if DEBUG:
        print(f"Total cases with is_deref checks: {len(crab)}")
    crab[f"{csv_name}_pp%"] = round(
        crab[f"{csv_name}_pp_safe"]
        / (crab[f"{csv_name}_pp_warn"] + crab[f"{csv_name}_pp_safe"])
        * 100,
        0,
    )
    crab[f"{csv_name}_opsem%"] = round(
        crab[f"{csv_name}_opsem_safe"]
        / (crab[f"{csv_name}_opsem_warn"] + crab[f"{csv_name}_opsem_safe"])
        * 100,
        0,
    )
    before_pp_tmlst = crab.loc[crab[f"{csv_name}_pp_to"] == True]["job_name"].to_list()
    after_pp_tmlst = crab.loc[crab[f"{csv_name}_opsem_to"] == True][
        "job_name"
    ].to_list()
    after_pp_tmlst = list(filter(lambda x: x not in before_pp_tmlst, after_pp_tmlst))
    print(
        f"{output_domain_name_map[csv_name]} before loop unrolling timeouts {len(before_pp_tmlst)} cases:"
    )
    print(f"\t {before_pp_tmlst}")
    print(
        f"{output_domain_name_map[csv_name]} after loop unrolling timeouts {len(after_pp_tmlst)} cases:"
    )
    print(f"\t {after_pp_tmlst}")
    # crab = crab[~crab['job_name'].isin(tmlst)] # exclude timeout cases
    crab["#pp"] = crab[f"{csv_name}_pp_warn"] + crab[f"{csv_name}_pp_safe"]
    # crab['#pp'] = crab['#pp'].astype(int)
    crab["#opsem"] = crab[f"{csv_name}_opsem_warn"] + crab[f"{csv_name}_opsem_safe"]
    # crab['#opsem'] = crab['#opsem'].astype(int)
    return crab


def print_show_general_statistics():
    print_centered_title("BENCHMARKS", "*", 45)
    utvpi = compute_stats(
        pd.concat(
            [process_csv("utvpi", "aws"), process_csv("utvpi", "fire_dancer")],
            ignore_index=True,
        ),
        "utvpi",
    )
    template_tvpi = compute_stats(
        pd.concat(
            [
                process_csv("template_tvpi", "aws"),
                process_csv("template_tvpi", "fire_dancer"),
            ],
            ignore_index=True,
        ),
        "template_tvpi",
    )
    pk = compute_stats(
        pd.concat(
            [process_csv("pk", "aws"), process_csv("pk", "fire_dancer")],
            ignore_index=True,
        ),
        "pk",
    )

    # merge all csv files
    res = pd.merge(utvpi, template_tvpi, how="inner", on="job_name")
    res = pd.merge(res, pk, how="inner", on="job_name")
    print(f"\nThe total number of cases: {len(res)}\n")
    return res


def make_assertion_consistent(res):
    # need to make assertions consistent after loop unrolling
    # seahorn will skip assertions check when all assertions are proven
    # here we instead change seahorn itself, just simply sum the number of assertions after loop unrolling as proved
    for dom in ["template_tvpi", "pk"]:
        res[f"{dom}_opsem_safe"] = np.where(
            (res[f"{dom}_pp_warn"] == 0.0)
            & (res[f"{dom}_pp_safe"] > 0.0)
            & ((res[f"{dom}_opsem_warn"] >= 0.0) | (res[f"{dom}_opsem_safe"] >= 0.0)),
            res[f"utvpi_opsem_safe"] + res[f"utvpi_opsem_warn"],
            res[f"{dom}_opsem_safe"],
        )
    return res


def print_time_stats(df: pd.DataFrame, col: str, is_pp: bool):
    # Compute statistics using pandas
    avg = df[col].mean()
    std = df[col].std()
    mn = df[col].min()
    mx = df[col].max()

    # Format and print
    key = next((k for k in output_domain_name_map if k in col), None)
    unrolled = "before" if "pp" in col else "after"
    if key:
        print(f"{output_domain_name_map[key]} {unrolled} loop unrolling time results:")
    print(f"   avg: {avg:.1f}s, std: {std:.1f}s, min: {mn:.1f}s, max: {mx:.1f}s")


def get_performance_res(res):
    print_centered_title("PERFORMANCE", "*", 45)
    # excluding timeout cases
    perf = res[
        ~res["utvpi_opsem_to"] & ~res["template_tvpi_opsem_to"] & ~res["pk_opsem_to"]
    ]
    # pp time statistics
    print_time_stats(perf, "utvpi_pp_time", True)
    print_time_stats(perf, "utvpi_opsem_time", False)
    print_time_stats(perf, "template_tvpi_pp_time", True)
    print_time_stats(perf, "template_tvpi_opsem_time", False)
    print_time_stats(perf, "pk_pp_time", True)
    print_time_stats(perf, "pk_opsem_time", False)


###########################################
#              Plot functions             #
###########################################
def scatter_with_axis_break(df, cand, configs, range1, range2):
    low_start, low_end = range1
    low_end = int(math.ceil((low_end - low_start) / 5) * 5)
    high_start, high_end = range2

    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(
        2,
        2,
        height_ratios=[high_end - high_start, low_end - low_start],
        width_ratios=[low_end - low_start, high_end - high_start],
        hspace=0.1,
        wspace=0.1,
    )
    ax_top_left = fig.add_subplot(gs[0, 0])
    ax_bottom_left = fig.add_subplot(gs[1, 0], sharex=ax_top_left)
    ax_top_right = fig.add_subplot(gs[0, 1], sharey=ax_top_left)
    ax_bottom_right = fig.add_subplot(
        gs[1, 1], sharex=ax_top_right, sharey=ax_bottom_left
    )

    # Set axis limits for each subplot segment
    ax_bottom_left.set_xlim(low_start, low_end)
    ax_bottom_left.set_ylim(low_start, low_end)
    ax_top_left.set_xlim(low_start, low_end)
    ax_top_left.set_ylim(high_start, high_end)
    ax_bottom_right.set_xlim(high_start, high_end)
    ax_bottom_right.set_ylim(low_start, low_end)
    ax_top_right.set_xlim(high_start, high_end)
    ax_top_right.set_ylim(high_start, high_end)

    def set_spine_as_grey(ax, edges):
        for edge in edges:
            sp = ax.spines[edge]
            sp.set_color("gray")
            sp.set_linestyle("--")
            sp.set_alpha(0.7)

    # Hide inner spines
    set_spine_as_grey(ax_bottom_left, ["right", "top"])
    set_spine_as_grey(ax_top_left, ["right", "bottom"])
    set_spine_as_grey(ax_bottom_right, ["left", "top"])
    set_spine_as_grey(ax_top_right, ["left", "bottom"])

    # Ticks (show only outer)
    ax_top_left.tick_params(labelbottom=False, bottom=False)
    ax_top_right.tick_params(
        labelbottom=False, bottom=False, labelleft=False, left=False
    )
    ax_bottom_right.tick_params(labelleft=False, left=False)

    # Set ticks and grid for each segment
    bottom_step = int(low_end / 5)
    ax_bottom_left.set_xticks(list(range(low_start, low_end + 1, bottom_step)))
    ax_bottom_left.set_yticks(list(range(low_start, low_end + 1, bottom_step)))
    top_step = high_end - high_start
    ax_top_right.set_xticks(list(range(high_start, high_end + 1, top_step)))
    ax_top_left.set_yticks(list(range(high_start, high_end + 1, top_step)))

    for ax in [ax_bottom_left, ax_bottom_right, ax_top_left, ax_top_right]:
        ax.grid(True, linestyle="--", color="gray", alpha=0.7)

    # Prepare mask for each quadrant and plot points only in that segment

    # pack each subplot with its x‐ and y‐ranges
    quadrants = [
        (ax_bottom_left, low_start, low_end, low_start, low_end),
        (ax_top_left, low_start, low_end, high_start, high_end),
        (ax_bottom_right, high_start, high_end, low_start, low_end),
        (ax_top_right, high_start, high_end, high_start, high_end),
    ]
    for col, config in configs.items():
        x, y = df["template_tvpi" + "_" + col], df[cand + "_" + col]
        for ax, x0, x1, y0, y1 in quadrants:
            mask = (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)

            ax.scatter(
                x[mask],
                y[mask],
                color=config["color"],
                label=config["name"],
                marker=config["marker"],
                edgecolors="black" if config["marker"] == "o" else None,
                zorder=3,
                clip_on=False,
                alpha=0.8,
            )

    # Plot the red trend line (only for visible segments)
    # Lower segment: (low_start, low_start) to (low_end, low_end)
    ax_bottom_left.plot(
        [low_start, low_end], [low_start, low_end], color="red", linewidth=2, zorder=2
    )
    # Upper segment: (high_start, high_start) to (high_end, high_end)
    ax_top_right.plot(
        [high_start, high_end],
        [high_start, high_end],
        color="red",
        linewidth=2,
        zorder=2,
    )

    # Axis labels
    plt.rc("legend", fontsize=8, title_fontsize=12)
    fig.supxlabel("Template TVPI (seconds)", fontsize=16, y=0.06)
    fig.supylabel(f"{cand.upper()} (seconds)", fontsize=16, x=0.06)

    # Add a legend
    ax_bottom_left.legend(title="Instance", alignment="left")
    sns.move_legend(ax_bottom_left, "lower right")

    # Diagonal break markers (slanted lines) on the broken axes
    d = 0.015
    # X-axis break markers
    ax_bottom_left.plot(
        (1 - d, 1 + d),
        (-d, +d),
        mec="k",
        mew=1,
        transform=ax_bottom_left.transAxes,
        color="k",
        clip_on=False,
    )
    ax_bottom_right.plot(
        (-d, +d),
        (-d, +d),
        mec="k",
        mew=1,
        transform=ax_bottom_right.transAxes,
        color="k",
        clip_on=False,
    )
    # Y-axis break markers
    ax_bottom_left.plot(
        (-d, +d),
        (1 - d, 1 + d),
        mec="k",
        mew=1,
        transform=ax_bottom_left.transAxes,
        color="k",
        clip_on=False,
    )
    ax_top_left.plot(
        (-d, +d),
        (-d, +d),
        mec="k",
        mew=1,
        transform=ax_top_left.transAxes,
        color="k",
        clip_on=False,
    )

    # plt.tight_layout()
    # plt.show()


def dump_scatter_plot(time_plot):
    for cand, offset, sub in [("utvpi", 2, "a"), ("pk", 15, "b")]:
        fig = plt.figure(figsize=(15, 12))
        cols = [
            "template_tvpi_pp_time",
            "template_tvpi_opsem_time",
            f"{cand}_pp_time",
            f"{cand}_opsem_time",
        ]
        second_max = second_distinct_max(time_plot, cols)
        df_test = time_plot[cols]
        scatter_with_axis_break(
            df_test,
            cand,
            configurations,
            (0, int(math.ceil(second_max))),
            (TIMEOUT_THRESHOLD - offset, TIMEOUT_THRESHOLD),
        )
        plt.savefig(f"SP_{cand}.png")
        print(f"Figure 4{sub} is saved as SP_{cand}.png")


###########################################
#             Table functions             #
###########################################
def get_assert_rate(df):
    print("\n")
    print_centered_title("PRECISION", "*", 45)
    df = df.copy()
    # excluding timeout cases so that all domains have results
    df = df[~df["utvpi_opsem_to"] & ~df["template_tvpi_opsem_to"] & ~df["pk_opsem_to"]]
    records = []
    for repo_name, cats in categories.items():
        for c in cats:
            row = {"category": c}
            if c == "others":
                ft = df[df["repo"] == repo_name]
                prefixes = tuple(cats[:-1])  # all but "others"
                ftd = ft[~ft["job_name"].str.startswith(prefixes, na=False)]
            elif c == "total":
                ftd = df
            else:
                ft = df[df["repo"] == repo_name]
                ftd = ft[ft["job_name"].str.startswith(c, na=False)]
            for prefix in ["pp", "opsem"]:
                for dom in dom_names:
                    total = (
                        ftd[f"{dom}_{prefix}_warn"].astype(int)
                        + ftd[f"{dom}_{prefix}_safe"].astype(int)
                    ).sum()
                    proved = ftd[f"{dom}_{prefix}_safe"].astype(int).sum()
                    rate = int(round(proved / total * 100, 0)) if total else 0
                    rate = f"{rate}%"

                    row[f"{dom}_{prefix}_assert"] = total
                    row[f"{dom}_{prefix}_proved"] = proved
                    row[f"{dom}_{prefix}_%"] = rate
            records.append(row)

    # wrap in a single-element list so we get a one‐row DataFrame
    precision = pd.DataFrame(records)
    # add extra columns
    precision["empty1"] = ""
    precision["empty2"] = ""

    table = precision[
        [
            "category",
            "empty1",
            "utvpi_pp_assert",
            "utvpi_pp_%",
            "template_tvpi_pp_%",
            "pk_pp_%",
            "empty2",
            "utvpi_opsem_assert",
            "utvpi_opsem_%",
            "template_tvpi_opsem_%",
            "pk_opsem_%",
        ]
    ].copy()
    for col in [
        "utvpi_pp_%",
        "template_tvpi_pp_%",
        "pk_pp_%",
        "utvpi_opsem_%",
        "template_tvpi_opsem_%",
        "pk_opsem_%",
    ]:
        table[col] = table[col].str.replace("%", r"\%")
    table["category"] = table["category"].str.replace("_", r"\_")
    table.rename(
        columns={
            "empty1": "",
            "utvpi_pp_assert": "BTotal",
            "utvpi_pp_%": "BZones",
            "template_tvpi_pp_%": "BtTVPI",
            "pk_pp_%": "BPK",
            "empty2": "",
            "utvpi_opsem_assert": "ATotal",
            "utvpi_opsem_%": "AZones",
            "template_tvpi_opsem_%": "AtTVPI",
            "pk_opsem_%": "APK",
        },
        inplace=True,
    )
    print(table.to_latex(escape=False, index=False))
    print("\nPlease use a latex compiler to render the table.\n")
    return table


def get_PK_bottleneck(res):
    display_cols = [
        "job_name",
        "pk_pp_warn", "pk_pp_safe", "template_tvpi_pp_warn", "template_tvpi_pp_safe",
        "pk_opsem_warn", "pk_opsem_safe", "template_tvpi_opsem_warn", "template_tvpi_opsem_safe",
    ]
    for phase in ("pp", "opsem"):
        pk_safe = f"pk_{phase}_safe"
        tvpi_safe = f"template_tvpi_{phase}_safe"
        diff = res.loc[res[pk_safe] < res[tvpi_safe]].copy()
        print(f"Template TVPI can solve at {phase} level but PK cannot: {len(diff)}")
        diff["assert_diff"] = diff[tvpi_safe] - diff[pk_safe]
        print(f"Total assertion difference: {diff['assert_diff'].sum()}")
        # print(diff[display_cols])


if __name__ == "__main__":
    print_centered_title("STATISTICS", "=")
    res = print_show_general_statistics()
    res = make_assertion_consistent(res)
    get_performance_res(res)
    time_plot = res[
        [
            "job_name",
            "template_tvpi_pp_time",
            "template_tvpi_opsem_time",
            "utvpi_pp_time",
            "utvpi_opsem_time",
            "pk_pp_time",
            "pk_opsem_time",
        ]
    ]
    dump_scatter_plot(time_plot)
    table = get_assert_rate(res)
    get_PK_bottleneck(res)