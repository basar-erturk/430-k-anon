##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
import random
from enum import unique
import glob
import os
import sys
from copy import deepcopy
import numpy as np
import datetime
import copy

if sys.version_info[0] < 3 or sys.version_info[1] < 5:
    sys.stdout.write("Requires Python 3.x.\n")
    sys.exit(1)


##############################################################################
# Helper Functions                                                           #
# These functions are provided to you as starting points. They may help your #
# code remain structured and organized. But you are not required to use      #
# them. You can modify them or implement your own helper functions.          #
##############################################################################

def read_dataset(dataset_file: str):
    """ Read a dataset into a list and return.

    Args:
        dataset_file (str): path to the dataset file.

    Returns:
        list[dict]: a list of dataset rows.
    """
    result = []
    with open(dataset_file) as f:
        records = csv.DictReader(f)
        for row in records:
            result.append(row)
    return result


def write_dataset(dataset, dataset_file: str) -> bool:
    """ Writes a dataset to a csv file.

    Args:
        dataset: the data in list[dict] format
        dataset_file: str, the path to the csv file

    Returns:
        bool: True if succeeds.
    """
    assert len(dataset) > 0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True


def read_DGH(DGH_file: str):
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    # TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.

    f = open(DGH_file, "r")
    DGH_dict = {}
    for line in f:
        depth = 0
        for char in line:
            if char == "\t":
                depth += 1
            else:
                break
        if line[-1] == '\n':
            line = line[:-1]
        DGH_dict[line[depth:]] = depth

    DGH_as_list = list(DGH_dict)  # any, preschool, elementary
    DGH_depths_as_list = list(DGH_dict.values())  # 0 1 2 3
    for i in range(len(DGH_as_list)):
        k = 0
        c = 0
        # print(f"For {DGH_as_list[i]}: i+k+1:{DGH_depths_as_list[i+k+1]}")
        while i + c + 1 < len(DGH_depths_as_list) and i + k + 1 < len(DGH_depths_as_list) and DGH_depths_as_list[i] < \
                DGH_depths_as_list[i + c + 1]:
            if i + c + 2 == len(DGH_depths_as_list):
                # print("k++")
                k += 1
                break
            elif DGH_depths_as_list[i + c + 1] >= DGH_depths_as_list[i + c + 2]:
                # print(f"{DGH_as_list[i]}  k++ for {DGH_as_list[i+c+1]}")
                k += 1
            c += 1
        if k == 0:
            k += 1
        DGH_dict[DGH_as_list[i]] = (DGH_depths_as_list[i], k)

    return DGH_dict


def read_DGHs(DGH_folder: str) -> dict:
    """ Read all DGH files from a directory and put them into a dictionary.

    Args:
        DGH_folder (str): the path to the directory containing DGH files.

    Returns:
        dict: a dictionary where each key is attribute name and values
            are DGHs in your desired format.
    """
    DGHs = {}
    for DGH_file in glob.glob(DGH_folder + "/*.txt"):
        attribute_name = os.path.basename(DGH_file)[:-4]
        DGHs[attribute_name] = read_DGH(DGH_file)

    return DGHs


def find_ancestor(child, DGH):
    DGH_as_list = list(DGH)
    DGH_depths_as_list = list(DGH.values())

    i = DGH_as_list.index(child)
    k = 1
    while i - k > 0 and DGH_depths_as_list[i][0] <= DGH_depths_as_list[i - k][0]:
        k += 1

    if i - k < 0:
        return DGH_as_list[0]
    return DGH_as_list[i - k]

def calculate_dist(rec1, rec2, DGHs):

    raw_rec1 = rec1
    raw_rec2 = rec2

    # to avoid pass by reference (altering the dictionary)
    alter_rec1 = copy.deepcopy(rec1)
    alter_rec2 = copy.deepcopy(rec2)

    for attr in alter_rec1.keys():
        if attr not in DGHs.keys():
            continue
        # condition of two records being in the same ec (for that attr)
        while not alter_rec1[attr] == alter_rec2[attr]:
            # condition of depths of values being equal, meaning both need to go one level shallower
            if DGHs[attr][alter_rec1[attr]][0] == DGHs[attr][alter_rec2[attr]][0]:
                for record in [alter_rec1, alter_rec2]:
                    record[attr] = find_ancestor(record[attr], DGHs[attr])
            else:
                # all attributes are not the same depth, pull the deeper to the shallower one

                # go to ancestors until all depths are the same (equal to min depth)
                while not DGHs[attr][alter_rec1[attr]][0] == DGHs[attr][alter_rec2[attr]][0]:
                    if DGHs[attr][alter_rec1[attr]][0] > DGHs[attr][alter_rec2[attr]][0]:
                        alter_rec1[attr] = find_ancestor(alter_rec1[attr], DGHs[attr])
                    else:
                        alter_rec2[attr] = find_ancestor(alter_rec2[attr], DGHs[attr])

    ## althought it's bad practice to duplicate code, since we cannot change function signatures
    ## and cost_LM function takes file paths as parameter, duplicating that function so that it can work with data
    raw_dataset = [raw_rec1, raw_rec2]
    anonymized_dataset = [alter_rec1, alter_rec2]
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))

    # TODO: complete this function.
    LM_cost_raw = 0
    for row in raw_dataset:
        for field, value in row.items():
            if field not in DGHs.keys():
                continue
            LM_cost_raw += (1/len(DGHs)) * ((DGHs[field][value][1] - 1) / (list(DGHs[field].values())[0][1] - 1))

    LM_cost_anon = 0
    for row in anonymized_dataset:
        for field, value in row.items():
            if field not in DGHs.keys():
                continue
            LM_cost_anon += (1/len(DGHs)) * ((DGHs[field][value][1] - 1) / (list(DGHs[field].values())[0][1] - 1))

    return LM_cost_anon - LM_cost_raw



##############################################################################
# Mandatory Functions                                                        #
# You need to complete these functions without changing their parameters.    #
##############################################################################


def cost_MD(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Distortion Metric (MD) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.
    MD_cost_raw = 0
    for row in raw_dataset:
        for field, value in row.items():
            if field not in DGHs.keys():
                continue
            MD_cost_raw += DGHs[field][value][0]

    MD_cost_anon = 0
    for row in anonymized_dataset:
        for field, value in row.items():
            if field not in DGHs.keys():
                continue
            MD_cost_anon += DGHs[field][value][0]

    return MD_cost_raw - MD_cost_anon


def cost_LM(raw_dataset_file: str, anonymized_dataset_file: str,
            DGH_folder: str) -> float:
    """Calculate Loss Metric (LM) cost between two datasets.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        anonymized_dataset_file (str): the path to the anonymized dataset file.
        DGH_folder (str): the path to the DGH directory.

    Returns:
        float: the calculated cost.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    anonymized_dataset = read_dataset(anonymized_dataset_file)
    assert (len(raw_dataset) > 0 and len(raw_dataset) == len(anonymized_dataset)
            and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.
    LM_cost_raw = 0
    for row in raw_dataset:
        for field, value in row.items():
            if field not in DGHs.keys():
                continue
            LM_cost_raw += (1/len(DGHs)) * ((DGHs[field][value][1] - 1) / (list(DGHs[field].values())[0][1] - 1))

    LM_cost_anon = 0
    for row in anonymized_dataset:
        for field, value in row.items():
            if field not in DGHs.keys():
                continue
            LM_cost_anon += (1/len(DGHs)) * ((DGHs[field][value][1] - 1) / (list(DGHs[field].values())[0][1] - 1))

    return LM_cost_anon - LM_cost_raw


def random_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                      output_file: str, s: int):
    """ K-anonymization a dataset, given a set of DGHs and a k-anonymity param.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
        s (int): seed of the randomization function
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    for i in range(len(raw_dataset)):  ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s)  ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)

    # TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...

    n_cluster = D // k
    for i in range(n_cluster - 1):
        clusters.append(raw_dataset[i * k:(i + 1) * k])

    clusters.append(raw_dataset[(i + 1) * k:])

    for cluster in clusters:
        for attr in cluster[0].keys():
            if attr not in DGHs.keys():
                continue
            # condition of all values in an ec being equal
            while not all(cluster[i][attr] == cluster[0][attr] for i in range(len(cluster))):
                # condition of all depths of values being equal, meaning it needs to go one level shallower
                if all(DGHs[attr][cluster[i][attr]][0] == DGHs[attr][cluster[0][attr]][0] for i in range(len(cluster))):
                    for record in cluster:
                        record[attr] = find_ancestor(record[attr], DGHs[attr])
                else:
                    # all attributes are not the same depth, pull all to the shallowest one

                    # find the shallowest depth
                    min_depth = 999
                    for record in cluster:
                        depth = DGHs[attr][record[attr]][0]
                        if depth < min_depth:
                            min_depth = depth
                    # go to ancestors until all depths are the same (equal to min depth)
                    while not all(DGHs[attr][cluster[i][attr]][0] == min_depth for i in range(len(cluster))):
                        for record in cluster:
                            depth = DGHs[attr][record[attr]][0]
                            if depth > min_depth:
                                record[attr] = find_ancestor(record[attr], DGHs[attr])

    # for cluster in clusters:
    #     print(cluster)
    #     print("*"*50)


    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            anonymized_dataset[item['index']] = item
            del item['index']

    write_dataset(anonymized_dataset, output_file)


def clustering_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                          output_file: str):
    """ Clustering-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """
    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)


    # TODO: complete this function.

    #random.shuffle(raw_dataset)
    clusters=[]
    for i in range(len(raw_dataset)):  # initially make all records in the dataset unmarked
        raw_dataset[i]['marked'] = False

    j = -1
    while j < len(raw_dataset) and sum(map(lambda x: not x['marked'], raw_dataset)) >= 2*k:
        j += 1
        if raw_dataset[j]['marked']:
            continue

        raw_dataset[j]['marked'] = True

        min_dists = [(0,j+900) for j in range(k-1)] # distinct values for max function to work
        for later in range(1,len(raw_dataset)-j):

            if raw_dataset[j+later]['marked']:
                continue

            dist = calculate_dist(raw_dataset[j], raw_dataset[j+later], DGHs)
            dist_values = list(min_dists[i][1] for i in range(len(min_dists)))
            if max(dist_values) > dist:
                min_dists[dist_values.index(max(dist_values))] = (raw_dataset[j+later], dist, j+later)

        for record, dist, index in min_dists:
            raw_dataset[index]['marked'] = True

        cluster = [raw_dataset[j]] + list(min_dists[i][0] for i in range(len(min_dists)))
        clusters.append(cluster)


    clusters.append(list(filter(lambda x: not x['marked'], raw_dataset)))

    # same code written for random_anonymizer
    for cluster in clusters:
        for attr in cluster[0].keys():
            if attr not in DGHs.keys():
                continue
            # condition of all values in an ec being equal
            while not all(cluster[i][attr] == cluster[0][attr] for i in range(len(cluster))):
                # condition of all depths of values being equal, meaning it needs to go one level shallower
                if all(DGHs[attr][cluster[i][attr]][0] == DGHs[attr][cluster[0][attr]][0] for i in range(len(cluster))):
                    for record in cluster:
                        record[attr] = find_ancestor(record[attr], DGHs[attr])
                else:
                    # all attributes are not the same depth, pull all to the shallowest one

                    # find the shallowest depth
                    min_depth = 999
                    for record in cluster:
                        depth = DGHs[attr][record[attr]][0]
                        if depth < min_depth:
                            min_depth = depth
                    # go to ancestors until all depths are the same (equal to min depth)
                    while not all(DGHs[attr][cluster[i][attr]][0] == min_depth for i in range(len(cluster))):
                        for record in cluster:
                            depth = DGHs[attr][record[attr]][0]
                            if depth > min_depth:
                                record[attr] = find_ancestor(record[attr], DGHs[attr])


    anonymized_dataset = raw_dataset

    for cluster in clusters:  # restructure according to previous indexes
        for item in cluster:
            del item['marked']

    # Finally, write dataset to a file
    write_dataset(anonymized_dataset, output_file)


def bottomup_anonymizer(raw_dataset_file: str, DGH_folder: str, k: int,
                        output_file: str):
    """ Bottom up-based anonymization a dataset, given a set of DGHs.

    Args:
        raw_dataset_file (str): the path to the raw dataset file.
        DGH_folder (str): the path to the DGH directory.
        k (int): k-anonymity parameter.
        output_file (str): the path to the output dataset file.
    """

    raw_dataset = read_dataset(raw_dataset_file)
    DGHs = read_DGHs(DGH_folder)

    # TODO: complete this function.

    # Finally, write dataset to a file
    # write_dataset(anonymized_dataset, output_file)


# Command line argument handling and calling of respective anonymizer:

if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now()  ##
print(start_time)  ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])


function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(
            f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)

    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print(f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now()  ##
print(end_time)  ##
print(end_time - start_time)  ##

# Sample usage:
# python3 code.py clustering DGHs/ adult-hw1.csv result.csv 300 5
