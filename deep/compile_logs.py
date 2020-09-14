import os, csv

ABSENT, CRASHED, INCOMPLETE, READY = range(4)

def check(name, pess=False):
    if name not in os.listdir("logs/"): return ABSENT, None
    if "bellman_update.csv" not in os.listdir(f"logs/{name}"): return CRASHED, None
    with open(f"logs/{name}/bellman_update.csv") as f:
        s = f.readlines()[-1].split(",")
    if s[2] != "250":
        return INCOMPLETE, None
    return READY, float(s[5] if not pess else s[6])

def check_imi(name):
    if name not in os.listdir("logs/"): return ABSENT, None
    if "imitation_policy.csv" not in os.listdir(f"logs/{name}"): return CRASHED, None
    with open(f"logs/{name}/imitation_policy.csv") as f:
        s = f.readlines()[-1].split(",")
    return READY, float(s[6])

if False:
    print("\n\n\nSTARTING -------\n")
    results = [["game","eps","algo","trial","score"]]
    for game in ["asterix", "breakout", "freeway", "space_invaders"]:
        for eps in ["000", "005", "010", "020", "033", "066", "100"]:
            for algo in ["", "imi_", "pess_"]:
                vals = []
                for trial in ["0", "1", "2"]:
                    fname = f"{algo}{game}_opt_eps{eps}--{trial}"
                    if algo== "imi_":status, val = check_imi(fname)
                    else:              status, val = check(fname, pess=algo == "pess_")
                    # print(fname, {ABSENT: "ABSENT", CRASHED: "CRASHED", INCOMPLETE: "INCOMPLETE", READY: val}[status])
                    if status == CRASHED: print(fname)
                    if val != None: vals.append(val)
                    if val != None: results.append([game, eps, {"":"naive","imi_":"imitation","pess_":"pessimistic"}[algo], trial, val])
                # if len(vals) == 3: print(algo + game, eps, sum(vals) / 3)

    with open("results/eps_compare.csv", "w") as f:
        s = "\n".join([",".join([str(x) for x in line]) for line in results])
        f.write(s)




print("\n\n\nSTARTING -------\n")
results = [["game","data","algo","trial","score"]]
for game in ["asterix", "breakout", "freeway", "space_invaders"]:
    for data in ["0010000", "0020000", "0050000", "0100000", "0200000", "0500000"]:
        for algo in ["", "imi_", "pess_"]:
            vals = []
            for trial in ["0", "1", "2"]:
                fname = f"{algo}{game}_opt_data{data}--{trial}"
                if algo== "imi_":status, val = check_imi(fname)
                else:              status, val = check(fname, pess=algo == "pess_")
                print(fname, {ABSENT: "ABSENT", CRASHED: "CRASHED", INCOMPLETE: "INCOMPLETE", READY: val}[status])
                if status == CRASHED: print(fname)
                if val != None: vals.append(val)
                if val != None: results.append([game, data, {"":"Naïve","imi_":"Imitation","pess_":"Proximal Pessimistic"}[algo], trial, val])
            # if len(vals) == 3: print(algo + game, eps, sum(vals) / 3)

with open("results/data_compare.csv", "w") as f:
    s = "\n".join([",".join([str(x) for x in line]) for line in results])
    f.write(s)