#!/user/bin/env python

"""Evaluation script for the global path planner."""
# 脚本用途:用于评估全局路径规划器(global path planner)的效果与性能,
#          对一批 CommonRoad 场景逐个运行全局路径规划,统计成功率、耗时,并输出可视化 PDF。

# =========================
# 标准库导入
# =========================
# Standard imports
import os
import sys
import time
import random
import matplotlib.pyplot as plt

# =========================
# 第三方库导入
# =========================
import progressbar         # 进度条显示(对场景文件列表迭代时展示进度)
import argparse            # 命令行参数解析(例如 --all_scenarios)
from PyPDF2 import PdfFileMerger  # PDF 合并(将每个场景的路径图汇总为 summary.pdf)
from joblib import Parallel, delayed  # 多进程并行计算(基于 joblib 的并行接口)
import multiprocessing      # 获取 CPU 核心数(用于并行 n_jobs

# =========================
# CommonRoad 可视化与文件读取
# =========================
from commonroad.visualization.draw_dispatch_cr import draw_object  # 统一绘制接口:lanelet_network、planning_problem 等对象
from commonroad.common.file_reader import CommonRoadFileReader  # CommonRoad 场景文件读取器:读取 scenario 与 planning_problem_set

# 将项目根路径加入 sys.path,方便导入自定义模块
# module_path:当前脚本所在路径往上回溯三层,推测是仓库根目录或源码根目录
module_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(module_path)

# 导入自定义的全局路径规划器(基于 lanelet 的 planner)
from planner.GlobalPath.lanelet_based_planner import LaneletPathPlanner

__author__ = "Florian Pfab"
__email__ = "Florian.Pfab@tum.de"
__date__ = "23.05.2020"


def evaluate_scenario(filename):
    """
    Evaluate a given scenario.

    Args:
        filename (str): Filename of the scenarios to be evaluated.

    Returns:
        array: Containing alternating a bool if the goal was reached and a log msg with relevant facts about the global path planning process. Has 2 * number of planning problems for the scenario entries.

    """
    """
    Evaluate a given scenario.
    评估单个 CommonRoad 场景文件:对场景中的每个 planning problem 运行一次全局路径规划,
    并输出:
      1) 是否到达目标(goal_reached)
      2) 记录日志字符串(包含 benchmark_id、planning_problem_id、成功与否、规划时间、路径长度)
      3) 各阶段耗时统计字典(加载/规划/绘图/总耗时)

    Args:
        filename (str): Filename of the scenarios to be evaluated.
                       待评估的场景文件路径(CommonRoad XML 等)

    Returns:
        array:
            返回一个列表(return_array),其内容是按 planning problem 交错存放的三元组:
              [goal_reached, log_msg, time_dict, goal_reached, log_msg, time_dict, ...]
            因此每个 planning problem 对应 3 个元素,长度为 3 * planning_problem 数量。

    """
    # 下面三个变量用于统计耗时(单位:秒)
    planning_time = 0.0          # 全局路径搜索(规划)耗时
    plotting_time = 0.0          # 绘制并保存 PDF 耗时(累计)
    load_scenarios_time = 0.0    # 场景加载耗时(读取文件并解析)

    ev_time0 = time.time() # 单个场景评估的起始时间戳(用于统计 total)

    # -------------------------
    # 读取场景文件
    # -------------------------
    # CommonRoadFileReader(filename).open() 会返回:
    # scenario:道路与动态元素等
    # planning_problem_set:包含一个或多个 planning problem(不同起终点/目标约束)
    scenario, planning_problem_set = CommonRoadFileReader(filename).open()

    # 记录加载时间:当前时刻 - 评估起始时刻
    load_scenarios_time += time.time() - ev_time0

    # Create return array
    return_array = []

    # -------------------------
    # 遍历每一个 planning problem
    # -------------------------
    # planning_problem_set.planning_problem_dict 是字典:
    # key: planning_problem_id
    # value: planning_problem 对象
    # 这里通过 range(len(...)) 的方式遍历,并用 list(...) 取 value 列表
    for planning_problem_id in range(len(list(planning_problem_set.planning_problem_dict.values()))):
        planning_problem = list(planning_problem_set.planning_problem_dict.values())[planning_problem_id]
        # 尝试为该 planning problem 计算全局路径
        # Try to find the global path
        try:
            # -------------------------
            # 初始化路径规划器
            # -------------------------
            # LaneletPathPlanner:通常基于 lanelet_network 构图并在 lanelet 层面搜索
            path_planner = LaneletPathPlanner(scenario, planning_problem)

            start_time = time.time()
            # 执行全局路径搜索
            # 返回:
            # path:路径点序列(例如 Nx2 的 numpy 数组),或 None(失败)
            # path_length:路径长度(单位通常是米)
            path, path_length = path_planner.plan_global_path()

            planning_time = time.time() - start_time
            # 根据 path 是否为 None 判断是否成功到达目标
            if path is not None:
                goal_reached = True
            else:
                goal_reached = False
            # 组装日志信息(以分号分隔,最后换行)
            # 格式:
            # benchmark_id; planning_problem_id; goal_reached; planning_time; path_length
            log_msg = scenario.benchmark_id + '; ' + str(planning_problem.planning_problem_id) + '; ' + str(goal_reached) + '; ' + str(round(planning_time, 5)) + '; ' + str(round(path_length, 2)) + '\n'

        except Exception:
            goal_reached = False
            log_msg = scenario.benchmark_id + '; ' + '; FAILED with Error: ' + str(sys.exc_info()[0]) + '; Error-message: ' + str(sys.exc_info()[1]) + '\n'

        ev_time1 = time.time()

        # Plot and save path
        plt.figure(figsize=[15, 10])

        try:
            # 先设置标题:成功
            plt.title(scenario.benchmark_id + ' ; Planning Problem ID: ' + str(planning_problem.planning_problem_id) + " Succeeded")
            plt.axis('equal')
            # 绘制 lanelet 网络与 planning problem(起点/目标区域等)
            draw_object(scenario.lanelet_network)
            draw_object(planning_problem)
            # 将全局路径分成两段:
            # path_to_goal:到达目标前部分
            # path_after_goal:到达目标后延伸部分(如果存在)
            # 这样可以用实线/虚线区分
            path_to_goal, path_after_goal = path_planner.split_global_path(global_path=path)
            plt.plot(path_to_goal[:, 0], path_to_goal[:, 1], color='red', zorder=20)
            plt.plot(path_after_goal[:, 0], path_after_goal[:, 1], color='red', zorder=20, linestyle='--')
            plt.xlabel('x in m')
            plt.ylabel('y in m')
            # 保存为 PDF:成功目录
            # 文件名:benchmark_id__planning_problem_id.pdf
            plt.savefig('./results/path_plots/succeeded/' + scenario.benchmark_id + str('__') + str(planning_problem.planning_problem_id) + '.pdf')
            plt.close()

        except Exception:
            # 绘图过程出现异常(例如 path 为 None 导致 split/索引失败等)
            # 则输出失败图:只绘制地图与 planning problem,不画路径
            plt.title(scenario.benchmark_id + "; Planning Problem ID: " + str(planning_problem.planning_problem_id) + " Failed")
            plt.axis('equal')
            draw_object(scenario.lanelet_network)
            draw_object(planning_problem)
            # 保存到 failed 目录
            plt.savefig('./results/path_plots/failed/' + scenario.benchmark_id + str('__') + str(planning_problem.planning_problem_id) + '.pdf')
            plt.close()

        plotting_time += time.time() - ev_time1
        # -------------------------
        # 生成耗时统计字典
        # -------------------------
        # planning:本 planning problem 的规划耗时(最近一次计算)
        # plotting:累计绘图耗时(从该场景开始到当前 planning problem 为止)
        # load:加载场景耗时(通常只发生一次,但用变量存储)
        # total:从 ev_time0 到当前的总耗时(累计)
        time_dict = {
            "planning": planning_time,
            "plotting": plotting_time,
            "load": load_scenarios_time,
            "total": (time.time() - ev_time0)
        }
        # 按 “结果/日志/时间统计” 的顺序追加到 return_array
        return_array.append(goal_reached)
        return_array.append(log_msg)
        return_array.append(time_dict)

    return return_array


if __name__ == '__main__':
    # 脚本主入口:批量评估多个场景文件,并汇总统计、合并 PDF、输出评估报告
    # -------------------------
    # 总计时起点(整次运行)
    # -------------------------
    time0 = time.time()

    # -------------------------
    # 命令行参数解析
    # -------------------------
    # --all_scenarios: 是否评估全部场景
    # 默认 False:只随机抽 10 个场景评估,以缩短运行时间
    parser = argparse.ArgumentParser()
    parser.add_argument('--all_scenarios', type=bool, default=False)
    args = parser.parse_args()

    # 将工作目录切换到当前脚本所在目录
    # 这样相对路径(./results/...、../../../commonroad-scenarios/...)才按预期工作
    # Change the working directory to the directory of the evaluation script
    os.chdir(os.path.dirname(os.path.realpath(__file__)))

    # 场景目录:CommonRoad scenarios 的根路径(相对当前脚本位置)
    # Get the directory of the scenarios
    scenario_directory = "../../../commonroad-scenarios/scenarios/"

    filelist = []
    # -------------------------
    # 遍历场景目录,收集所有文件路径
    # -------------------------
    # Get all files in the directory and the subdirectories
    for root, dirs, files in os.walk(scenario_directory):
        for file in files:
            filelist.append(os.path.join(root, file))
    # -------------------------
    # 若不评估全量场景,则随机抽取 10 个文件
    # -------------------------
    # If not all scenarios should be evaluated, 10 random ones are evaluated
    if args.all_scenarios is False:
        new_filelist = []
        for i in range(10):
            new_filelist.append(random.choice(filelist))
        filelist = new_filelist
    # -------------------------
    # 结果目录结构
    # -------------------------
    result_directory = './results/'
    path_plots_directory = './results/path_plots/'
    path_plots_directory_succeeded = './results/path_plots/succeeded/'
    path_plots_directory_failed = './results/path_plots/failed/'

    # Create directories if they dont exist
    if not os.path.exists(result_directory):
        os.makedirs(result_directory)
    if not os.path.exists(path_plots_directory):
        os.makedirs(path_plots_directory)
    if not os.path.exists(path_plots_directory_succeeded):
        os.makedirs(path_plots_directory_succeeded)
    if not os.path.exists(path_plots_directory_failed):
        os.makedirs(path_plots_directory_failed)

    # -------------------------
    # 获取可用 CPU 核心数,用于并行
    # -------------------------
    # Get number of available cores
    num_cores = multiprocessing.cpu_count()
    print("Running on {} cores".format(num_cores))

    time_pre_multi = time.time()
    # -------------------------
    # 多进程并行评估所有场景文件
    # -------------------------
    # Parallel(n_jobs=num_cores):使用全部核心
    # delayed(evaluate_scenario)(i):将 evaluate_scenario 调用封装为可并行任务
    # progressbar.progressbar(filelist):在主进程显示进度条
    # Evaluate scenarios on all cores
    results = Parallel(n_jobs=num_cores)(delayed(evaluate_scenario)(i) for i in progressbar.progressbar(filelist))
    # -------------------------
    # 如果需要不使用多进程,可改用下面串行版本(原作者保留)
    # -------------------------
    # results = []
    # for file in filelist:
    #     results.append(evaluate_scenario(file))

    time_multi_main = time.time() - time_pre_multi
    # -------------------------
    # 汇总统计变量初始化
    # -------------------------
    goal_reached_list = []  # 每个 planning problem 是否成功(True/False)
    log_msgs = []           # 日志字符串列表(每个 planning problem 一条)
    n_scenarios = len(results)  # 评估的场景数(等于 filelist 的长度)
    n_planning_problems = 0     # planning problem 总数(跨所有场景累计)
    planning_time = 0.0         # 规划耗时累计
    load_scenarios_time = 0.0   # 加载耗时累计
    plotting_time = 0.0         # 绘图耗时累计
    multi_time = 0.0            # evaluate_scenario 返回的 total 累计(用于估算占比)
    # -------------------------
    # 遍历 results,将每个 planning problem 的结果拆出来汇总
    # -------------------------
    # 注意:evaluate_scenario 返回的是 [goal, log, time_dict] * N
    for scenario_counter in range(len(results)):
        for planning_problem_counter in range(int(len(results[scenario_counter]) / 3)):
            n_planning_problems += 1
            goal_reached_list.append(results[scenario_counter][0 + planning_problem_counter * 3])
            log_msgs.append(results[scenario_counter][1 + planning_problem_counter * 3])
            time_dict = results[scenario_counter][2 + planning_problem_counter * 3]
            planning_time += time_dict["planning"]
            plotting_time += time_dict["plotting"]
            load_scenarios_time += time_dict["load"]
            multi_time += time_dict["total"]
    # -------------------------
    # 统计成功/失败与成功率
    # -------------------------
    # Get number of succeeded/failed scenarios
    suc_count = goal_reached_list.count(True)         # 成功的 planning problem 数
    fail_count = goal_reached_list.count(False)       # 失败的 planning problem 数
    suc_rate = suc_count / n_planning_problems * 100  # 成功率(百分比)

    # -------------------------
    # 写入日志文件:results/result_logs.txt
    # -------------------------
    log_file = open(os.path.join(result_directory, 'result_logs.txt'), 'w')
    log_file.write("Benchmark ID; Planning Problem; Succeeded; Execution Time; Path Length\n")
    for msg in log_msgs:
        log_file.write(msg)
    log_file.write('\n%d scenarios containing %d planning problems were evaluated.\nSuccess-rate: %.2f %%' % (n_scenarios, n_planning_problems, suc_rate))
    log_file.close()
    # 控制台输出汇总
    print('Evaluation finished. %d scenarios containing %d planning problems were evaluated.\nSuccess-rate: %.2f %%' % (n_scenarios, n_planning_problems, suc_rate))

    time1 = time.time()

    # Merge PDFs
    # -------------------------
    # 合并成功路径图 PDF:summary_succeeded.pdf
    # -------------------------
    merger = PdfFileMerger()
    for pdf in os.listdir(path_plots_directory_succeeded):
        merger.append(os.path.join(path_plots_directory_succeeded, pdf))

    merger.write(os.path.join(result_directory, 'summary_succeeded.pdf'))
    merger.close()
    # -------------------------
    # 合并失败路径图 PDF:summary_failed.pdf
    # -------------------------
    merger = PdfFileMerger()
    for pdf in os.listdir(path_plots_directory_failed):
        merger.append(os.path.join(path_plots_directory_failed, pdf))

    merger.write(os.path.join(result_directory, 'summary_failed.pdf'))
    merger.close()
    # -------------------------
    # 合并结果目录下所有 PDF:summary.pdf
    # 包含 summary_succeeded.pdf、summary_failed.pdf、evaluation.pdf 等
    # -------------------------
    merger = PdfFileMerger()
    for file in os.listdir(result_directory):
        if file.endswith(".pdf"):
            merger.append(os.path.join(result_directory, file))

    merger.write(os.path.join(result_directory, 'summary.pdf'))
    merger.close()

    create_pdfs_time = time.time() - time1
    total_time = time.time() - time0

    # Calculate the time percentages of the different tasks
    percentage_create_pdfs = create_pdfs_time / total_time
    percentage_multi = time_multi_main / total_time
    percentage_planning = percentage_multi * (planning_time / multi_time)
    percentage_plotting = percentage_multi * (plotting_time / multi_time)
    percentage_load_scenarios = percentage_multi * (load_scenarios_time / multi_time)
    percentage_other_stuff = 1 - (percentage_create_pdfs + percentage_planning + percentage_plotting + percentage_load_scenarios)
    # -------------------------
    # 计算平均耗时(每个 planning problem 的平均值)
    # -------------------------
    # Get the average times for planning, plotting and loading
    av_planning_time = planning_time / n_planning_problems
    av_plotting_time = plotting_time / n_planning_problems
    av_load_time = load_scenarios_time / n_planning_problems

    # Plot the summary of the evaluation
    # -------------------------
    # 绘制评估汇总图:results/evaluation.pdf
    # 包括:
    # 1) 饼图:耗时占比
    # 2) 横向条形图:成功/失败比例(100% 中成功率/失败率)
    # 3) 表格:关键统计指标与平均耗时
    # -------------------------
    plt.subplot(221)
    plt.suptitle('Evaluation summary')
    plt.axis('equal')
    labels = 'Other stuff', 'Creating PDFs', 'Planning', 'Plotting', 'Loading Scenarios'
    sizes = [percentage_other_stuff, percentage_create_pdfs, percentage_planning, percentage_plotting, percentage_load_scenarios]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%')
    plt.subplot(222)
    # barh:绘制水平条形图
    # 这里用两个条形叠加显示:绿色成功 + 红色失败
    plt.barh(['Success rate', 'Success rate'], [suc_rate, 100 - suc_rate], left=[0, suc_rate], color=['g', 'r'], height=15)
    plt.text(suc_rate / 2, 0, str(round(suc_rate, 2)) + ' %', ha='center', va='center')
    ax = plt.gca()
    ax.set_xlim(0, 100)
    plt.axis('equal')
    plt.subplot(212)
    plt.axis('off')
    # 表格内容:指标名与数值
    table_text = [['Evaluated scenarios', n_scenarios],
                  ['Evaluated planning problems', n_planning_problems],
                  ['Failed planning problems', fail_count],
                  ['Solved planning problems', suc_count],
                  ['Success rate', str(round(suc_rate, 2)) + ' %'],
                  ['Evaluation Duration', str(round(total_time, 2)) + ' s'],
                  ['Average scenario loading time', str(round(av_load_time, 5)) + ' s'],
                  ['Average planning time', str(round(av_planning_time, 5)) + ' s'],
                  ['Average plotting time', str(round(av_plotting_time, 5)) + ' s']]
    # 将表格绘制到图上
    plt.table(table_text, cellLoc='left', loc='center')
    plt.subplots_adjust(wspace=1.0)
    # 保存汇总图为 PDF
    plt.savefig('./results/evaluation.pdf')
    plt.show()
