from pathlib import Path

import numpy as np
import os
import subprocess as sp
import pickle
import  matplotlib.pyplot as plt

path = "/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/run_benchmark.py"
path2 = "/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/run_nls_jssp.py"
path3 = "/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/data/JSSP/"

def vary_rtgs():
    mean_makespans = {}
    rtg_range = np.arange(0.3,1.75,0.05)
    os.chdir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/")
    for i in rtg_range:
        out = None
        cmd = 'python ' + path + ' -r ' + path2 + ' -d ' + path3 + ' -g jssp15x15 -p jssp -m nls -e eval_jssp --args env=jssp15x15_unf -n 200 -x 15x15/brisk-lake-1290.pt  -a True -f '+ str(round(i,2))
        try:
            print(os.getcwd())
            out = sp.run(cmd.split(),
                         universal_newlines=True,
                         capture_output=True,
                         check=True
                         )
            print(out.stdout)
        except sp.CalledProcessError as e:
            print(f"encountered error for call: {e.cmd}\n")
            print(e.stderr)
        makespan = float(out.stdout.split("makespan")[1])
        mean_makespans[i] = makespan

    lists = sorted(mean_makespans.items())

    x, y = zip(*lists)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.plot(x, y)
    plt.ylabel('Mean makespan of various rtgs on 100 instances (NLSDT)  ')
    plt.xlabel('returns_to_go')
    plt.savefig('plot_15x15.png')
    plt.show()

    with open('15x15_vary_rtg.pkl', 'wb') as file:
        pickle.dump(mean_makespans, file)

def get_dt_schedules():
    rtg_range = np.arange(0.8,1.25,0.05)
    dt_schedules = {}
    for i in rtg_range:
        dt_schedules[i] = {}
        for filename in os.scandir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/outputs_experiment/dt/30x20/"+str(i)):
            if os.path.splitext(filename)[1] == ".npy":
                sequence_data  = np.load(filename, allow_pickle=True)
                dt_schedules[i][filename.name[0:-4]] = sequence_data
    return dt_schedules

def get_nls_schedules():
    no_dt_schedules = {}
    for filename in os.scandir(
            "/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/outputs_experiment/no_dt/100_it_10ct/"):
        if os.path.splitext(filename)[1] == ".npy":
            sequence_data = np.load(filename, allow_pickle=True)
            no_dt_schedules[filename.name[0:-4]] = sequence_data
    return no_dt_schedules

def calculate_influence_of_rtg_on_levDistance(actions=False): #If actions is true distance will be calculated on actions not schedules
    schedule_index = 2 if actions else 1
    mean_distance = {}
    rtg_range = np.arange(0.8,1.25,0.05)
    dt_schedules = get_dt_schedules()
    nls_schedules = get_nls_schedules()
    os.chdir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/")
    for i in rtg_range:
        mean_distance[i] = 0
        for schedule_id in dt_schedules[i]:
            if schedule_id in nls_schedules:
                mean_distance[i] += levenshtein_distance(np.asarray(dt_schedules[i][schedule_id][schedule_index]).flatten(), np.asarray(nls_schedules[schedule_id][schedule_index]).flatten())
        mean_distance[i] = mean_distance[i]/len(nls_schedules)

    lists = sorted(mean_distance.items())
    x, y = zip(*lists)
    plt.plot(x, y)
    plt.ylabel('mean_levensthein')
    plt.xlabel('returns_to_go')
    plt.savefig('plot_levensthein_solar.png')
    plt.show()

def levenshtein_distance(list1, list2):
    m = len(list1)
    n = len(list2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i

    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if list1[i - 1] == list2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]

def start_time_distance(dt, nls):
    sum = 0
    for i in range(len(nls)):
        sum += abs(dt[i] - nls[i])
    return sum/len(nls)

def calculate_startTime_distance():
    mean_distance = {}
    rtg_range = np.arange(0.8,1.25,0.05)
    dt_start_times = get_dt_schedules()
    nls_start_times = get_nls_schedules()
    for i in rtg_range:
        mean_distance[i] = 0
        for schedule_id in dt_start_times[i]:
            if schedule_id in nls_start_times:
                mean_distance[i] += start_time_distance(dt_start_times[i][schedule_id][0], nls_start_times[schedule_id][0])
        mean_distance[i] = mean_distance[i] / len(nls_start_times)

    lists = sorted(mean_distance.items())
    x, y = zip(*lists)
    plt.plot(x, y)
    plt.ylabel('mean_levensthein')
    plt.xlabel('returns_to_go')
    plt.savefig('plot_stdist_solar-energy.png')
    plt.show()


def hyperparameter_models():
    os.chdir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/")
    for model_file in os.scandir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/lib/trained_models/15x15"):
        if os.path.splitext(model_file)[1] == ".pt":
            out = None
            cmd = 'python ' + path + ' -r ' + path2 + ' -d ' + path3 + f' -g jssp15x15/test -p jssp -m nls -e eval_jssp --args env=jssp15x15_unf -n 200 -x {model_file.name} -a True)'
            try:
                print(os.getcwd())
                out = sp.run(cmd.split(),
                             universal_newlines=True,
                             capture_output=True,
                             check=True
                             )
                print(out.stdout)
            except sp.CalledProcessError as e:
                print(f"encountered error for call: {e.cmd}\n")
                print(e.stderr)
            makespan = float(out.stdout.split("makespan")[1])
            print(f"{model_file}: {makespan}")


if __name__ == '__main__':
    vary_rtgs()
    #hyperparameter_models()
  # calculate_influence_of_rtg_on_levDistance(actions=False)
    #calculate_startTime_distance()