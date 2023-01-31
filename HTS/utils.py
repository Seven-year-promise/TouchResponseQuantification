import csv

RAW_CLASSES = {"Wild type": 0,
               "GABAA pore blocker": 1,
               "vesicular ACh transport antagonist": 2,
               "nAChR orthosteric agonist": 3,
               "nAChR orthosteric antagonist": 4,
               "TRPV agonist": 5,
               "GABAA allosteric antagonist": 6,
               "RyR agonist": 7,
               "Na channel": 8,
               "complex II inhibitor": 9,
               "nAChR allosteric agonist": 10,
               "unknown-likely neurotoxin": 11
               }

def read_actions(action_path):
    action_with_compounds = {}
    for a in RAW_CLASSES.keys():
        action_with_compounds[a] = []
    action_with_compounds["Wild type"] = ["C0"]
    with open(action_path, "r") as a_f:
        reader_to_lines = []
        reader = csv.reader(a_f, delimiter=",")
        for j, l in enumerate(reader):
            #print(l[0])
            reader_to_lines.append(l[0])
        action_with_compounds[reader_to_lines[1]] = ["C"+str(i) for i in reader_to_lines[2:30]]
        action_with_compounds[reader_to_lines[30]] = ["C"+str(i) for i in reader_to_lines[31:43]]
        action_with_compounds[reader_to_lines[43]] = ["C"+str(i) for i in reader_to_lines[44:52]]
        action_with_compounds[reader_to_lines[52]] = ["C"+str(i) for i in reader_to_lines[53:63]]
        action_with_compounds[reader_to_lines[63]] = ["C"+str(i) for i in reader_to_lines[64:122]]
        action_with_compounds[reader_to_lines[122]] = ["C"+str(i) for i in reader_to_lines[123:144]]
        action_with_compounds[reader_to_lines[144]] = ["C"+str(i) for i in reader_to_lines[145:147]]
        action_with_compounds[reader_to_lines[147]] = ["C"+str(i) for i in reader_to_lines[148:150]]
        action_with_compounds[reader_to_lines[150]] = ["C"+str(i) for i in reader_to_lines[151:156]]
        action_with_compounds[reader_to_lines[156]] = ["C"+str(i) for i in reader_to_lines[157:159]]
        action_with_compounds[reader_to_lines[159]] = ["C"+str(i) for i in reader_to_lines[160:171]]
    print("compound action modes:", action_with_compounds)

    return action_with_compounds

def get_key(dict, value):
    for k, v in dict.items():
        #print(value, v, k)
        if value in v:
            return k
    return "None"