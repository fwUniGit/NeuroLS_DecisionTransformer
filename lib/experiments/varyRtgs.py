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
    rtg_range = np.arange(0.25,1.8,0.05)
    os.chdir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/")
    for i in rtg_range:
        out = None
        cmd = 'python ' + path + ' -r ' + path2 + ' -d ' + path3 + ' -g jssp30x20/test -p jssp -m nls -e eval_jssp --args env=jssp30x20_unf -n 200 -x 30x20/rural-valley-270-2158.pt  -a True -f '+ str(round(i,2))
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
    plt.title("Mean makespan of various rtgs on 100 instances")
    plt.ylabel('mean makespan')
    plt.xlabel('return-to-go factor')
    plt.savefig('plot_30x20.png')
    plt.show()

    with open('30x20_vary_rtg.pkl', 'wb') as file:
        pickle.dump(mean_makespans, file)

def get_dt_schedules(rtg_range):
    dt_schedules = {}
    for i in rtg_range:
        dt_schedules[i] = {}
        for filename in os.scandir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/outputs_experiment/dt/15x15_briskLake/"+str(round(i,2))):
            if os.path.splitext(filename)[1] == ".npy":
                sequence_data  = np.load(filename, allow_pickle=True)
                dt_schedules[i][filename.name[0:-4]] = sequence_data
    return dt_schedules

def get_nls_schedules():
    no_dt_schedules = {}
    for filename in os.scandir(
            "/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/outputs_experiment/no_dt/15x15-no_dt/"):
        if os.path.splitext(filename)[1] == ".npy":
            sequence_data = np.load(filename, allow_pickle=True)
            no_dt_schedules[filename.name[0:-4]] = sequence_data
    return no_dt_schedules

def calculate_influence_of_rtg_on_hamDistance(actions=False): #If actions is true distance will be calculated on actions not schedules
    schedule_index = 2 if actions else 1 # defines if the Hamming distance should be calculated for the machine sequences or for the action sequences
    dist_string= "selected actions" if actions else "machine sequences"
    mean_distance = {}
    rtg_range = np.arange(0.25,1.8,0.05)
    dt_schedules = get_dt_schedules(rtg_range)
    nls_schedules = get_nls_schedules()
    os.chdir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/newPlots")
    for i in rtg_range:
        mean_distance[i] = []
        for schedule_id in dt_schedules[i]:
            if schedule_id in nls_schedules:
                mean_distance[i].append( levenshtein_distance(np.asarray(dt_schedules[i][schedule_id][schedule_index]).flatten(), np.asarray(nls_schedules[schedule_id][schedule_index]).flatten()))
        #mean_distance[i] = mean_distance[i]/len(nls_schedules)

    lists = sorted(mean_distance.items())
    x, y = zip(*lists)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    fig, ax = plt.subplots()
    bp = ax.boxplot(mean_distance.values(), meanline=True, showmeans=True ,patch_artist=True, boxprops=dict(facecolor="lightblue",color="lightblue"))
    ax.set_xticklabels([round(x,2) for x in mean_distance.keys()])
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.grid(True)
    plt.title(f"Hamming distance of {dist_string} for various rtgs on 100 instances")
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], loc="upper right")
    plt.ylabel('Hamming distance')
    plt.xlabel('return-to-go factor')
    plt.savefig(f'Hamming{actions}_15x15.png')
    plt.show()
    print(mean_distance)

def levenshtein_distance(sequence1, sequence2):
    counter = 0
    for i in range(len(sequence1)):
        if sequence1[i] != sequence2[i]: counter += 1
    return counter
    # m = len(sequence1)
    # n = len(sequence2)
    # distance = np.zeros((m + 1, n+ 1))
    #
    # for s1 in range(m + 1):
    #     distance[s1][0] = s1
    # for s2 in range(n + 1):
    #     distance[0][s2] = s2
    #
    # for s1 in range(1, m + 1):
    #     for s2 in range(1, n + 1):
    #         if sequence1[s1 - 1] == sequence2[s2 - 1]:
    #             distance[s1][s2] = distance[s1 - 1][s2 - 1]
    #         else:
    #             distance[s1][s2] = 1 + min(distance[s1 - 1][s2], distance[s1][s2 - 1], distance[s1 - 1][s2 - 1])
    #
    # return distance[m][n]


def start_time_distance(dt, nls):
    sum = 0
    for i in range(len(nls)):
        sum += abs(dt[i]*100 - nls[i]*100)
    return sum/len(nls)

def calculate_startTime_distance(rtg_range):
    mean_distance = {}
    dt_start_times = get_dt_schedules(rtg_range)
    nls_start_times = get_nls_schedules()
    for i in rtg_range:
        mean_distance[i] = []
        for schedule_id in dt_start_times[i]:
            if schedule_id in nls_start_times:
                mean_distance[i].append(start_time_distance(dt_start_times[i][schedule_id][0], nls_start_times[schedule_id][0]))
     #   mean_distance[i] = mean_distance[i] / len(nls_start_times)

    lists = sorted(mean_distance.items())
    x, y = zip(*lists)
    plt.rcParams["figure.figsize"] = (7, 5)
    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    fig, ax = plt.subplots()
    bp = ax.boxplot(mean_distance.values(), meanline=True, showmeans=True, patch_artist=True,
                    boxprops=dict(facecolor="lightblue", color="lightblue"))
    ax.set_xticklabels([round(x, 2) for x in mean_distance.keys()])
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
    plt.grid(True)
    #plt.plot(x, y)
    plt.legend([bp['medians'][0], bp['means'][0]], ['median', 'mean'], loc="upper right")
    plt.title(" Start time distance for various rtgs on 100 instances")
    plt.ylabel('Start time distance')
    plt.xlabel('return-to-go factor')
    plt.savefig('std_15x15.png')
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


def evaluate_action_frequencys():
    dt_schedules = get_dt_schedules([1.0])[1.0]
    nls_schedules = get_nls_schedules()
    action_count_nls = np.zeros(10)
    action_count_nlsdt = np.zeros(10)

    counter = 0
    for schedule in nls_schedules:
        #if counter > 3: break;
        counter += 1
        for action_index in range(len(nls_schedules[schedule][2])):
            nls_selected_action = nls_schedules[schedule][2][action_index][0]
            action_count_nls[nls_selected_action] += 1
            if schedule in dt_schedules:
                nlsdt_selected_action = dt_schedules[schedule][2][action_index][0]
                action_count_nlsdt[nlsdt_selected_action] += 1
    plt.stairs(action_count_nls, np.arange(11), facecolor="tab:blue")
    plt.show()
    plt.stairs(action_count_nlsdt, np.arange(11), facecolor="tab:blue")
    plt.show()

if __name__ == '__main__':
    rtg_range = np.arange(0.25,1.8,0.05)
    #vary_rtgs()
    #hyperparameter_models()
    os.chdir("/mnt/c/Users/fabia/OneDrive/MyUni/Masterarbeit/NeuroLS_DecisionTransformer/newPlots")
    #calculate_influence_of_rtg_on_hamDistance(actions=True)
    calculate_influence_of_rtg_on_hamDistance(actions=False)
    #calculate_startTime_distance(rtg_range)
    #evaluate_action_frequencys()