##############################################################################
# This skeleton was created by Efehan Guner  (efehanguner21@ku.edu.tr)       #
# Note: requires Python 3.5+                                                 #
##############################################################################

import csv
from enum import unique
import glob
import os
import sys
from copy import deepcopy
import numpy as np
import datetime

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
    assert len(dataset)>0, "The anonymized dataset is empty."
    keys = dataset[0].keys()
    with open(dataset_file, 'w', newline='')  as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(dataset)
    return True



def read_DGH(DGH_file: str) -> dict:
    """ Reads one DGH file and returns in desired format.

    Args:
        DGH_file (str): the path to DGH file.
    """
    #TODO: complete this code so that a DGH file is read and returned
    # in your own desired format.
    dgh_dict = {}

    # determine depths of keys:
    with open(DGH_file) as f:
        for line in f:
            depth = 0
            for char in line:
                if char != "\t":
                    break
                depth += 1
            
            dgh_dict[line.strip()] = depth
        
        key_list = list(dgh_dict)
        depth_list = list(dgh_dict.values())

        for x in range(len(key_list)):
            leaf_flag = False
            current = x
            next = x + 1
            leaf_list = []

            if next < len(key_list) and depth_list[next] <= depth_list[current]:
                dgh_dict[key_list[current]] = {"Depth:": depth_list[current], "Leaves:": leaf_list, "Leaf Count:": len(leaf_list)}  
                continue

            while next < len(key_list):
                if (next) == len(key_list) - 1:
                    leaf_list.append(key_list[next])
                if depth_list[current] >= depth_list[next]:
                    break
                # leaf checkkkk
                if(next + 1) < len(key_list):
                    if depth_list[next] >= depth_list[next + 1]:
                        leaf_list.append(key_list[next])
                    
                    next += 1
                else:
                    break
                
            dgh_dict[key_list[current]] = {"Depth:": depth_list[current], "Leaves:": leaf_list, "Leaf Count:": len(leaf_list)}  

    return dgh_dict



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


# def QI-equalizer() yazabilirsin!

def find_ancestor(child, DGH): #değiştir
    DGH_as_list = list(DGH)
    DGH_depths_as_list = list(DGH.values())

    i = DGH_as_list.index(child)
    k = 1
    while i - k > 0 and DGH_depths_as_list[i]["Depth:"] <= DGH_depths_as_list[i - k]["Depth:"]:
        k += 1

    if i - k < 0:
        return DGH_as_list[0]
    return DGH_as_list[i - k]


def dist_calculator(topmost_record, rest_of_unmarked, DGHs):
    # dist between two records = the total LM cost of 
    # hypothetically placing those two records in one equivalence class (EC) with the minimum amount of generalization necessary.

    lm_cost_list = []

    wattr = 1 / len(DGHs)

    for i in range(len(rest_of_unmarked)):
        lm_T = 0
        lm_record = 0
        second_record = rest_of_unmarked[i]
        #print("second_record = %s\n" % second_record)
        for key in topmost_record.keys():
            lm_val = 0
            if topmost_record[key] != second_record[key] and key in DGHs.keys():
                dgh_dict = DGHs[key]
                key_list = list(dgh_dict)
                total_leaves = dgh_dict[key_list[0]]["Leaf Count:"]
                lm_val = (dgh_dict[second_record[key]]["Leaf Count:"] - 1) / (total_leaves - 1)
            
            lm_record += (lm_val * wattr)
            
        lm_T += lm_record

        lm_cost_list.append({"Index:": second_record["Index:"], "Cost:": lm_T})

    return lm_cost_list         




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
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    # Charge penalty to each instance that is generalized or suppressed
    # Sum over all records and all attributes

    penalty = 0

    for i in range(len(raw_dataset)):
        dict_list = raw_dataset[i]
        anon_list = anonymized_dataset[i]
        for key in dict_list.keys():
            if dict_list[key] != anon_list[key] and key in DGHs.keys():
                dgh_dict = DGHs[key]
                penalty += dgh_dict[dict_list[key]]["Depth:"] - dgh_dict[anon_list[key]]["Depth:"]
    
    return penalty



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
    assert(len(raw_dataset)>0 and len(raw_dataset) == len(anonymized_dataset)
        and len(raw_dataset[0]) == len(anonymized_dataset[0]))
    DGHs = read_DGHs(DGH_folder)

    #TODO: complete this function.
    # LM(val) = (# of descendant leaves of val − 1) / (total # of leaves in DGH − 1)
    # LM(record) = sum of all (wattr * LM(val))
    # LM(T) = sum of all LM(record)

    wattr = 1 / len(DGHs)
    lm_T = 0

    for i in range(len(raw_dataset)):
        lm_record = 0
        dict_list = raw_dataset[i]
        anon_list = anonymized_dataset[i]
        for key in dict_list.keys():
            lm_val = 0
            if dict_list[key] != anon_list[key]:
                dgh_dict = DGHs[key]
                key_list = list(dgh_dict)
                total_leaves = dgh_dict[key_list[0]]["Leaf Count:"]
                lm_val = (dgh_dict[anon_list[key]]["Leaf Count:"] - 1) / (total_leaves - 1)
            
            lm_record += (lm_val * wattr)
            
        lm_T += lm_record

    return lm_T         
            

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

    for i in range(len(raw_dataset)): ##set indexing to not lose original places of records
        raw_dataset[i]['index'] = i

    raw_dataset = np.array(raw_dataset)
    np.random.seed(s) ## to ensure consistency between runs
    np.random.shuffle(raw_dataset)  ##shuffle the dataset to randomize

    clusters = []

    D = len(raw_dataset)
    
    #TODO: START WRITING YOUR CODE HERE. Do not modify code in this function above this line.
    # Store your results in the list named "clusters". 
    # Order of the clusters is important. First cluster should be the first EC, second cluster second EC, ...
    
    # If the size of the dataset is a perfect multiple of k, you’ll have exactly |D|/k ECs.
    if D%k == 0:
        start_index = 0
        for x in range(int(D/k)): # number of total ECs = D/k
            cluster = []
            while len(cluster) != k:
                cluster.append(raw_dataset[start_index])
                start_index += 1
            
            # QI-wise equality!!!

            lowest_depth = 1000000
        
            column_equal = False

            for key in DGHs.keys():

                prev_attr = cluster[0][key]
                lowest_depth = 1000000
                row_same_depth = 0
                is_same_depth = False
                same_attr_no = 0

                while column_equal == False:    
                    row_count = 0
                    same_attr_no = 0

                    for row in cluster:
                        attr_value = row[key]
                        attr_depth = DGHs[key][attr_value]["Depth:"]

                        if is_same_depth == False:
                        
                            if attr_depth < lowest_depth:
                                lowest_depth = attr_depth
                            elif attr_depth == lowest_depth:
                                row_same_depth += 1  
                            else:
                                current_depth = attr_depth
                                while current_depth != lowest_depth:
                                    row[key] = find_ancestor(row[key], DGHs[key])
                                    current_depth = DGHs[key][row[key]]["Depth:"]

                            if row_same_depth == k:
                                is_same_depth = True
                            
                            prev_attr = row[key]
                        else:
                            if row_count != 0 and prev_attr == attr_value:
                                same_attr_no += 1
                            else:
                                row[key] = find_ancestor(row[key], DGHs[key])
                                prev_attr = row[key]

                        row_count += 1
                    if same_attr_no == k - 1:
                        column_equal = True
                    else:
                        column_equal = False

                column_equal = False

            clusters.append(cluster)
    # Otherwise, the very last EC needs to contain between k+1 and 2k-1 records, so that all records can end up belonging to an EC that contains at least k records.
    else:
        start_index = 0

        for x in range(int(D/k)): # number of total ECs = D/k
            cluster = []
            if x == int(D/k) - 1:
                k += int(D%k)

            while len(cluster) != k:
                cluster.append(raw_dataset[start_index])
                start_index += 1
            
            # QI-wise equality!!!

            lowest_depth = 1000000
        
            column_equal = False

            for key in DGHs.keys():

                prev_attr = cluster[0][key]
                lowest_depth = 1000000
                row_same_depth = 0
                is_same_depth = False
                same_attr_no = 0

                while column_equal == False:    
                    row_count = 0
                    same_attr_no = 0

                    for row in cluster:
                        attr_value = row[key]
                        attr_depth = DGHs[key][attr_value]["Depth:"]

                        if is_same_depth == False:
                        
                            if attr_depth < lowest_depth:
                                lowest_depth = attr_depth
                            elif attr_depth == lowest_depth:
                                row_same_depth += 1  
                            else:
                                current_depth = attr_depth
                                while current_depth != lowest_depth:
                                    row[key] = find_ancestor(row[key], DGHs[key])
                                    current_depth = DGHs[key][row[key]]["Depth:"]

                            if row_same_depth == k:
                                is_same_depth = True
                            
                            prev_attr = row[key]
                        else:
                            if row_count != 0 and prev_attr == attr_value:
                                same_attr_no += 1
                            else:
                                row[key] = find_ancestor(row[key], DGHs[key])
                                prev_attr = row[key]

                        row_count += 1
                    if same_attr_no == k - 1:
                        column_equal = True
                    else:
                        column_equal = False

                column_equal = False

            clusters.append(cluster)

    # END OF STUDENT'S CODE. Do not modify code in this function below this line.

    anonymized_dataset = [None] * D

    for cluster in clusters:        #restructure according to previous indexes
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

    D = len(raw_dataset)

    anonymized_dataset = [None] * D 
    #TODO: complete this function.

    # Initially, set all records in D as unmarked.

    row_index = 0
    for row in raw_dataset:
        row["Marked:"] = False
        row["Index:"] = row_index
        row_index += 1
    
    # While there exist at least 2*k unmarked records in D:
    unmarked_record_count = len(raw_dataset)
    current_record = {}

    ECs = []

    EC_list = []

    while unmarked_record_count >= 2*k:
        EC_list = []
        # Pick the topmost unmarked record (unmarked record with lowest index) 
        for row1 in raw_dataset:
            if row1["Marked:"] == False:
                current_record = row1
                break

        current_record["Marked:"] = True
        EC_list.append(current_record)

        lowest_dist_count = 0

        rest_of_unmarked = []

        for row2 in raw_dataset:
            if row2["Marked:"] == False:
                rest_of_unmarked.append(row2)

        dist_list = dist_calculator(current_record, rest_of_unmarked, DGHs)

        lowest_dist = float(10000000000)

        ind = 0

        while lowest_dist_count < k - 1:
            if dist_list[ind]["Cost:"] < lowest_dist and raw_dataset[dist_list[ind]["Index:"]]["Marked:"] == False:
                lowest_dist = dist_list[ind]["Cost:"]
                lowest_dist_count += 1
                raw_dataset[dist_list[ind]["Index:"]]["Marked:"] = True
                EC_list.append(raw_dataset[dist_list[ind]["Index:"]])            
    
            ind += 1

            if ind == len(dist_list):
                ind = 0
                lowest_dist = float(10000000000)

        # Construct a k-anonymous EC:

        lowest_depth = 1000000
    
        column_equal = False

        for key in DGHs.keys():

            prev_attr = EC_list[0][key]
            lowest_depth = 1000000
            row_same_depth = 0
            is_same_depth = False
            same_attr_no = 0

            while column_equal == False:    
                row_count = 0
                same_attr_no = 0

                for row3 in EC_list:
                    attr_value = row3[key]
                    attr_depth = DGHs[key][attr_value]["Depth:"]

                    if is_same_depth == False:
                    
                        if attr_depth < lowest_depth:
                            lowest_depth = attr_depth
                        elif attr_depth == lowest_depth:
                            row_same_depth += 1  
                        else:
                            current_depth = attr_depth
                            while current_depth != lowest_depth:
                                row3[key] = find_ancestor(row3[key], DGHs[key])
                                current_depth = DGHs[key][row3[key]]["Depth:"]

                        if row_same_depth == len(EC_list):
                            is_same_depth = True
                        
                        prev_attr = row3[key]
                    else:
                        if row_count != 0 and prev_attr == attr_value:
                            same_attr_no += 1
                        else:
                            row3[key] = find_ancestor(row3[key], DGHs[key])
                            prev_attr = row3[key]

                    row_count += 1
                if same_attr_no == len(EC_list) - 1:
                    column_equal = True
                else:
                    column_equal = False
            column_equal = False
        
        ECs.append(EC_list)
        unmarked_record_count -= len(EC_list)
        

    # Once less than 2*k unmarked records
    last_EC = []

    for row4 in raw_dataset:
        if row4["Marked:"] == False:
            last_EC.append(row4)

    lowest_depth2 = 1000000
    
    column_equal2 = False

    for key2 in DGHs.keys():

        prev_attr = last_EC[0][key2]
        lowest_depth2 = 1000000
        row_same_depth = 0
        is_same_depth = False
        same_attr_no = 0

        while column_equal2 == False:    
            row_count = 0
            same_attr_no = 0
            for row5 in last_EC:
                attr_value = row5[key2]
                attr_depth = DGHs[key2][attr_value]["Depth:"]

                if is_same_depth == False:
                
                    if attr_depth < lowest_depth2:
                        lowest_depth2 = attr_depth
                    elif attr_depth == lowest_depth2:
                        row_same_depth += 1  
                    else:
                        current_depth = attr_depth
                        while current_depth != lowest_depth2:
                            row5[key2] = find_ancestor(row5[key2], DGHs[key2])
                            current_depth = DGHs[key2][row5[key2]]["Depth:"]

                    if row_same_depth == len(last_EC):
                        is_same_depth = True
                    
                    prev_attr = row5[key2]
                else:
                    if row_count != 0 and prev_attr == attr_value:
                        same_attr_no += 1
                    else:
                        row5[key2] = find_ancestor(row5[key2], DGHs[key2])
                        prev_attr = row5[key2]

                row_count += 1
            if same_attr_no == len(last_EC) - 1:
                column_equal2 = True
            else:
                column_equal2 = False
        column_equal2 = False
    
    ECs.append(last_EC)

    for EC_num in ECs:        #restructure according to previous indexes
        for ECs_elem in EC_num:
            anonymized_dataset[ECs_elem["Index:"]] = ECs_elem
            del ECs_elem["Index:"]
            del ECs_elem["Marked:"]

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

    #TODO: complete this function.

    # Finally, write dataset to a file
    #write_dataset(anonymized_dataset, output_file)

# TESTTTTTTT

DGHsss = read_DGHs("DGHS/")

for key, value in DGHsss.items():
    print(key, ' : ', value)
    print("\n")
    

# Command line argument handling and calling of respective anonymizer:
if len(sys.argv) < 6:
    print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
    print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
    sys.exit(1)

algorithm = sys.argv[1]
if algorithm not in ['clustering', 'random', 'bottomup']:
    print("Invalid algorithm.")
    sys.exit(2)

start_time = datetime.datetime.now() ##
print(start_time) ##

dgh_path = sys.argv[2]
raw_file = sys.argv[3]
anonymized_file = sys.argv[4]
k = int(sys.argv[5])

function = eval(f"{algorithm}_anonymizer");
if function == random_anonymizer:
    if len(sys.argv) < 7:
        print(f"Usage: python3 {sys.argv[0]} algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)")
        print(f"\tWhere algorithm is one of [clustering, random, bottomup]")
        sys.exit(1)
        
    seed = int(sys.argv[6])
    function(raw_file, dgh_path, k, anonymized_file, seed)
else:    
    function(raw_file, dgh_path, k, anonymized_file)

cost_md = cost_MD(raw_file, anonymized_file, dgh_path)
cost_lm = cost_LM(raw_file, anonymized_file, dgh_path)
print (f"Results of {k}-anonimity:\n\tCost_MD: {cost_md}\n\tCost_LM: {cost_lm}\n")

end_time = datetime.datetime.now() ##
print(end_time) ##
print(end_time - start_time)  ##

# Sample usage:
# python3 skeleton.py clustering DGHs/ adult-hw1.csv result.csv 300 5
# python3 skeleton.py random DGHs/ adult-hw1.csv result.csv 300 5