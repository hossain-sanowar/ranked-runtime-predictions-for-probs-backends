import sys
from pathlib import Path

import pandas as pd
import os


def run():
    raw_dataframes = {}
    healy_dataframes = {}
    backend_names = ["kodkod", "prob", "z3"]

    # read csv-files
    for name in backend_names:
        raw_dataframes[name] = pd.read_csv(os.path.join(csv_src_path, "{}-raw-all.csv".format(name)))
        healy_dataframes[name] = pd.read_csv(os.path.join(csv_src_path, "{}-healy-all.csv".format(name)))
    runtime_types = [raw_dataframes, healy_dataframes]

    # initialize rank-dataframes
    rank_classif_raw_dataframe = raw_dataframes["kodkod"].copy()
    rank_classif_healy_dataframe = healy_dataframes["kodkod"].copy()
    rank_classif_dataframes = [rank_classif_raw_dataframe, rank_classif_healy_dataframe]

    # initialize RankingDistributions.txt
    distribution_result_file = open_file(result_path, "RankingDistributions.txt", mode="w")

    distribution_result_file.write("The first Rankings were generated using the raw costs.\n")
    distribution_result_file.write("The second Rankings were generated using the 'healy' costs.\n")
    distribution_result_file.write("a<b<c is representative for the order of costs of the respective backends.\n")

    # loop over all Label0s of the the three backends and find out their cost-order
    n = len(raw_dataframes["kodkod"]["Label0"])  # 597134
    i = -1
    for runtime_type in runtime_types:
        i += 1

        kodkod_label0 = runtime_type["kodkod"]["Label0"].values
        prob_label0 = runtime_type["prob"]["Label0"].values
        z3_label0 = runtime_type["z3"]["Label0"].values

        ranks_list = []

        print("Done reading.")

        # xyz stands for x<y<z
        kpz = 0
        kzp = 0
        pkz = 0
        pzk = 0
        zkp = 0
        zpk = 0

        for j in range(0, n):
            k = kodkod_label0[j]
            p = prob_label0[j]
            z = z3_label0[j]

            if k < p:
                if k < z:
                    if p < z:
                        kpz += 1
                        ranks_list.append("kpz")
                    elif z < p:
                        kzp += 1
                        ranks_list.append("kzp")
                elif z < k:
                    zkp += 1
                    ranks_list.append("zkp")
            elif p < k:
                if p < z:
                    if k < z:
                        pkz += 1
                        ranks_list.append("pkz")
                    elif z < k:
                        pzk += 1
                        ranks_list.append("pzk")
                elif z < p:
                    zpk += 1
                    ranks_list.append("zpk")

        # determines how many times at least two costs were equal in a ranking
        equal_runtime = n-(kpz+kzp+pkz+pzk+zkp+zpk)

        distribution_result_file.write("-----------------------\n")
        distribution_result_file.write("Rankings:\n")
        distribution_result_file.write("k<p<z: %d,(%f%%)\n" % (kpz, kpz/n))
        distribution_result_file.write("k<z<p: %d,(%f%%)\n" % (kzp, kzp/n))
        distribution_result_file.write("p<k<z: %d,(%f%%)\n" % (pkz, pkz/n))
        distribution_result_file.write("p<z<k: %d,(%f%%)\n" % (pzk, pzk/n))
        distribution_result_file.write("z<k<p: %d,(%f%%)\n" % (zkp, zkp/n))
        distribution_result_file.write("z<p<k: %d,(%f%%)\n" % (zpk, zpk/n))
        distribution_result_file.write("others: %d,(%f%%)\n" % (equal_runtime, equal_runtime/n))
        distribution_result_file.write("#k is fastest: %d,(%f%%)\n" % (kzp+kpz, (kzp+kpz)/n))
        distribution_result_file.write("#p is fastest: %d,(%f%%)\n" % (pkz+pzk, (pkz+pzk)/n))
        distribution_result_file.write("#z is fastest: %d,(%f%%)\n" % (zkp+zpk, (zkp+zpk)/n))

        rank_classif_dataframes[i]["Label0"] = ranks_list

        # print rankings of the backend-costs to the respective csv
        if i == 0:
            rank_classif_dataframes[i].to_csv(os.path.join(result_path, "best-ranks-raw.csv"), index=False)
        else:
            rank_classif_dataframes[i].to_csv(os.path.join(result_path, "best-ranks-healy.csv"), index=False)

        distribution_result_file.flush()

    distribution_result_file.write("-----------------------\n")
    distribution_result_file.close()


def open_file(file_dir, file_name, mode="w"):
    Path(file_dir).mkdir(parents=True, exist_ok=True)
    return open(file_dir+os.path.sep+file_name, mode=mode)


if __name__ == "__main__":
    # standard directories
    dirname = os.path.dirname
    csv_src_path = os.path.join(dirname(__file__), "data", "regression", "all")
    result_path = os.path.join(dirname(__file__), "results", "real_rankings", "local")

    # optional alternative directories
    if len(sys.argv) == 3:
        csv_src_path = sys.argv[1]
        result_path = sys.argv[2]
    run()
