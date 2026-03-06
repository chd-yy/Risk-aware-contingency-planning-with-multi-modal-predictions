"""This module provides parent classes for easy evaluation of a planner."""
"""
这个模块提供“评测(evaluation)”相关的父类/工具类,
用来方便地评估一个 planner(规划器)在一个或一组 CommonRoad 场景上表现如何。

整体流程(非常重要,小白先看这段):
1) ScenarioEvaluator:评估“一个场景”
   - 读场景 -> 初始化 -> 仿真 -> 捕获成功/失败原因 -> 返回一个结果字典 return_dict
2) DatasetEvaluator:评估“一堆场景”
   - 获取场景列表 -> 逐个调用 ScenarioEvaluator.eval_scenario -> 收集结果
   - 支持单进程 or 多进程(multiprocessing)
3) EvalFilesGenerator:把 DatasetEvaluator 的结果写成文件
   - 统计日志 planner_statistic.log
   - 成功/失败场景列表 scenario_completion_list.json
   - harm.json(伤害汇总)
   - exec_timing.json(如果计时开启的话)
   - weights.json / settings.json
"""
import sys
import pathlib
import random
import multiprocessing
import time
import json
import git

import progressbar
import numpy as np

# 解释:
# - __name__ == "__main__" 表示“这是主程序入口”
# - pathlib.Path(__file__) 是当前文件路径
# - parents[1] 表示向上两级目录(取决于你的项目结构)
if __name__ == "__main__":
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

# 一些自定义异常:
# - ExecutionTimeoutError:执行超时
# - GoalReachedNotification:到达目标(可能还区分“是否在时间限制内到达”)
from commonroad_helper_functions.exceptions import (
    ExecutionTimeoutError,
    GoalReachedNotification,
)

# 你的 ScenarioHandler:负责读场景、创建 agent、推进仿真等
from beliefplanning.planner.plannertools.scenario_handler import ScenarioHandler

class ScenarioEvaluator(ScenarioHandler):
    """Generic class for evaluating a scenario with a planner."""
    """
    ScenarioEvaluator 继承 ScenarioHandler:
    - ScenarioHandler 负责“怎么跑场景”
    - ScenarioEvaluator 在此基础上负责“怎么评价结果并整理成 return_dict”
    """

    def eval_scenario(self, scenario_path):
        """
        评估一个场景(核心函数之一)。

        参数:
        - scenario_path:相对路径/文件名,例如 "xxx/yyy.xml"
          注意:它会拼到 self.path_to_scenarios 下面变成完整路径。

        返回:
        - return_dict(字典):包含 success、失败原因、执行时间、harm、平均速度等信息
        """
        """WIP."""
        # 每次评估一个场景前,先把计时器清零
        self.exec_timer.reset()
        # 拼接场景文件完整路径:path_to_scenarios / scenario_path
        self.scenario_path = self.path_to_scenarios.joinpath(scenario_path)
        print(f"(eval_scenario)self.scenario_path: {self.scenario_path}")
        # 记录开始时间,用于计算整个 eval_scenario 耗时
        start_time = time.time()
        # 用计时器记录 total 这个大步骤耗时
        with self.exec_timer.time_with_cm("total"):
            try:
                # 初始化:读场景、创建 ego agent、创建碰撞检测器等
                self._initialize()
                # 仿真:循环 time_step,让 agent.step() 推进
                self._simulate()
            except GoalReachedNotification as excp:
                # 如果抛出 GoalReachedNotification,说明达到目标了(属于成功)
                return_dict = {"success": True, "reason_for_failure": None}
                # 通过异常信息文本判断是否“在时间内到达”
                if "Goal reached but time exceeded!" in str(excp):
                    return_dict["reached_in_time"] = False
                elif "Goal reached in time!" in str(excp):
                    return_dict["reached_in_time"] = True
            except ExecutionTimeoutError as excp:
                # 如果执行超时:属于失败
                return_dict = {"success": False, "reason_for_failure": str(excp)}
            except NotImplementedError as excp:
                # 这类错误一般意味着:你有抽象方法没实现(比如 PlannerCreator.get_planner)
                # 这里选择直接抛出去,让调用者看到并修复代码
                raise excp
            except Exception as excp:
                # 兜底:其他所有异常都算失败
                # 为了方便定位,会打印:场景名 >>> 错误原因                
                import traceback
                traceback.print_exc()
                print(f"{scenario_path} >>> {str(excp)}")
                return_dict = {"success": False, "reason_for_failure": str(excp)}
                # 如果错误信息里包含 "Simulation",说明仿真超时或逻辑问题导致评测结果不可信
                # 这里直接终止整个程序(sys.exit)
                if "Simulation" in str(excp):
                    print(f"Stopping Evaluation, results not valid anymore due to simulation time out in {scenario_path}")
                    sys.exit()

            # TODO implement saving and animating scenario
            # self.postprocess()
        # ---------- 下面开始统一补充 return_dict 的其他字段(无论成功失败都会写) ----------

        # 记录场景路径(原始字符串)            
        return_dict["scenario_path"] = scenario_path
        # 记录总耗时(秒)
        return_dict["exec_time"] = time.time() - start_time
        # harm 是 ScenarioHandler 里维护的伤害统计字典
        return_dict["harm"] = self.harm

        # velocities:这里用 vel_list(速度列表)的平均值作为“平均速度”
        # 注意:如果 vel_list 为空,就返回 0        
        if not self.vel_list:
            return_dict["velocities"] = 0
        else:
            return_dict["velocities"] = np.mean(self.vel_list)
        # timesteps_agent:速度列表的长度,通常可以理解为 agent 实际走过的步数            
        return_dict["timesteps_agent"] = len(self.vel_list)
        # 如果启用了 timing(计时),把更细粒度的计时数据也放进返回字典        
        if self.timing_enabled:
            return_dict["exec_times_dict"] = self.exec_timer.get_timing_dict()

        return return_dict


class DatasetEvaluator:
    """Class for creating a chunk of scenarios with a scenario evaluator."""
    """
    DatasetEvaluator 用于评估“一组场景”(数据集)。

    它会:
    - 收集要评估的场景列表 scenario_list
    - 对每个场景调用 scenario_evaluator.eval_scenario
    - 把所有 return_dict 收集起来
    - 最后调用 EvalFilesGenerator 写结果文件
    """
    def __init__(
        self, scenario_evaluator, eval_directory, limit_scenarios=None, disable_mp=False
    ):
        """
        参数解释:
        - scenario_evaluator:一个 ScenarioEvaluator(或其子类)对象
        - eval_directory:输出评估文件的目录(Path 对象)
        - limit_scenarios:
            * int:随机抽取 N 个场景
            * list[str]:指定场景文件名列表(只跑这几个)
            * None:跑全部场景
        - disable_mp:是否禁用多进程(True=单进程,False=多进程)
        """
        """Documenation in progress.

        Args:
            scenario_evaluator ([type]): [description]
            limit_scenarios ([int, list, none], optional):
                    int -> number of randomly picked scenarios is evaluated.
                    list[str] -> list of given scenario names is evaluated
                    None -> All scenarios are evaluated.
            disable_mp (bool, optional): [description]. Defaults to False.
        """
        self.scenario_evaluator = scenario_evaluator
        # 把评估函数保存起来,方便在单进程/多进程里传递
        self.evaluation_function = self.scenario_evaluator.eval_scenario
        self.mp_disabled = disable_mp
        # 场景目录来自 scenario_evaluator
        self.path_to_scenarios = self.scenario_evaluator.path_to_scenarios
        # 构建要评估的场景列表
        self.scenario_list = self._get_scenario_list(limit_scenarios)
        # 评测总耗时
        self.evaluation_time = None
        # 存放每个场景的 return_dict
        self.return_dicts = []
        # 输出目录
        self.eval_directory = eval_directory
        # 文件生成器:负责把 return_dicts 写成各种文件
        self.eval_files_generator = EvalFilesGenerator(self)

    def _get_scenario_list(self, limit_scenarios):
        """
        根据 limit_scenarios 决定要跑哪些场景。

        返回:
        - scenario_index:list[str],每个元素是某个 xml 相对路径字符串
        """
        """Create the dict with scenarios to evaluate.

        Returns:
            dict: dict with {"scenario_path": <number of timesteps in scenario>}
        """
        if isinstance(limit_scenarios, list):
            # 如果用户传入的是 list:直接用指定场景(常用于 debug
            # Use debug scenarios
            scenario_index = limit_scenarios
            print("Using given scenarios!\n")
        else:
            # 否则:扫描目录下所有 xml 场景文件
            # Get all scenario names from scenario repository
            print("Reading scenario names from repo:")
            # self.path_to_scenarios.glob("**/*.xml") 会递归寻找所有 xml
            scenario_index = [
                str(child.relative_to(self.path_to_scenarios))
                for child in self.path_to_scenarios.glob("**/*.xml")
            ]
            print(f"{len(scenario_index)} scenarios found!\n")

            # Remove Blacklist from scenario index
            # 去掉黑名单中的场景(planner_creator.get_blacklist() 返回 list)
            scenario_index = [
                index
                for index in scenario_index
                if not any(
                    bl in index
                    for bl in self.scenario_evaluator.planner_creator.get_blacklist()
                )
            ]
            print(
                f"Remove blacklisted scenarios --> {len(scenario_index)} scenarios left\n"
            )

            # Check if the limit_scenarios key is a int
            # 如果 limit_scenarios 是 int:随机抽样 N 个场景
            if isinstance(limit_scenarios, int):
                scenario_index = random.sample(scenario_index, limit_scenarios)
                print(
                    f"Sampled {limit_scenarios} random scenarios from data set:\n{scenario_index}"
                )

        return scenario_index

    def eval_dataset(self):
        """
        评估整个数据集:
        - 遍历 scenario_list
        - 单进程或多进程执行
        - 记录总耗时
        - 生成评估文件
        """
        """WIP."""
        print("Evaluating Scenarios:")
        start_time = time.time()

        # progressbar 用来显示进度条
        with progressbar.ProgressBar(max_value=len(self.scenario_list)).start() as pbar:
            if self.mp_disabled:
                # Calculate single threaded
                self._loop_with_single_processing(pbar)
            else:
                # use multiprocessing
                self._loop_with_with_mulitprocessing(pbar)
        # 记录评测总耗时
        self.evaluation_time = time.time() - start_time
        # 写文件
        self.eval_files_generator.create_eval_files()

    def _loop_with_single_processing(self, pbar):
        """
        单进程循环评估每个场景:
        - 一个一个 eval_scenario
        - 收集 return_dict
        - 更新进度条
        """
        for scenario_path in self.scenario_list:
            return_dict = self.evaluation_function(scenario_path)
            self._process_return_dict(return_dict)
            pbar.update(pbar.value + 1)

    def _loop_with_with_mulitprocessing(self, pbar):
        """
        多进程版本(更快,但调试更麻烦):

        注意点(小白必须知道):
        - multiprocessing 会启动多个子进程
        - 某些对象(例如带 GPU、带句柄、不可 pickle 的对象)可能无法传递
        - Windows 和 Linux 的多进程行为不同(fork vs spawn)
        """

        # 这里写死 cpu_count=10(本来可以用 multiprocessing.cpu_count() 自动获取)
        cpu_count = 10  # multiprocessing.cpu_count()
        # create worker pool
        # 创建进程池 pool
        with multiprocessing.Pool(processes=cpu_count) as pool:
            # imap_unordered:无序返回结果,哪个进程先算完就先返回哪个
            # Use imap unordered to parelly eval the results
            for return_dict in pool.imap_unordered(
                self.evaluation_function, self.scenario_list
            ):
                self._process_return_dict(return_dict)
                pbar.update(pbar.value + 1)

    def _process_return_dict(self, return_dict):
        """WIP."""
        """
        处理每个场景返回结果。
        目前只是把它 append 进列表,后续可以在这里做更多统计/过滤。
        """
        self.return_dicts.append(return_dict)


class EvalFilesGenerator:
    """Helper class for a dataset evaluator to generate eval files."""
    """
    这个类负责:把 DatasetEvaluator 的结果写成各种文件。

    dataset_evaluator.return_dicts 是核心数据来源:
    每个 return_dict 里包含 success、exec_time、harm 等字段。
    """

    def __init__(self, dataset_evaluator):
        """Documentation in progress."""
        self.dataset_evaluator = dataset_evaluator

    @property
    def _number_processed(self):
        """已经处理了多少个场景(return_dict 个数)"""
        return len(self.dataset_evaluator.return_dicts)

    @property
    def _number_successful(self):
        """成功的场景数量(success=True)"""
        num = 0
        for return_dict in self.dataset_evaluator.return_dicts:
            if return_dict["success"]:
                num += 1
        return num

    @property
    def _number_successful_in_time(self):
        """成功且在时间内完成的场景数量(success=True 且 reached_in_time=True)"""
        num = 0
        for return_dict in self.dataset_evaluator.return_dicts:
            if return_dict["success"]:
                if return_dict["reached_in_time"]:
                    num += 1
        return num

    @property
    def _completion_rate(self):
        """完成率(百分比)= 成功数量 / 总数量 * 100"""
        return 100 * self._number_successful / self._number_processed

    @property
    def _in_time_completion_rate(self):
        """按时完成率(百分比)= 按时成功数量 / 总数量 * 100"""
        return 100 * self._number_successful_in_time / self._number_processed

    @property
    def _avg_exection_time_per_scenario(self):
        """所有场景的平均执行时间(秒)"""
        return np.mean(
            [
                return_dict["exec_time"]
                for return_dict in self.dataset_evaluator.return_dicts
            ]
        )

    @property
    def _avg_exection_time_per_successfull_scenario(self):
        """成功场景的平均执行时间(秒)"""
        if self._number_successful > 0:
            return np.mean(
                [
                    return_dict["exec_time"]
                    for return_dict in self.dataset_evaluator.return_dicts
                    if return_dict["success"] is True
                ]
            )
        else:
            return 0.0

    @property
    def _fail_log(self):
        """
        失败日志列表:
        - 格式:"<scenario_path> >>> <reason>"
        - 如果 reason 太长(>100),截断
        """
        fail_list = []
        for return_dict in self.dataset_evaluator.return_dicts:
            if return_dict["success"] is False:
                fail_reason = return_dict["reason_for_failure"]
                # Truncate error messages longer than 100 chars
                if len(fail_reason) > 100:
                    fail_reason = fail_reason[:100] + " ..."
                fail_list.append(return_dict["scenario_path"] + " >>> " + fail_reason)
        return fail_list

    # not used at the moment
    @property
    def _avg_velocity_driven(self):
        """
        平均速度(仅成功场景),未使用。
        注意:当前实现似乎有问题:取的是 return_dicts[0]['velocities']。
        """
        v = []
        for i in range(len(self.dataset_evaluator.return_dicts)):
            if self.dataset_evaluator.return_dicts[i]['success'] is True:
                v.append(self.dataset_evaluator.return_dicts[0]['velocities'])

        return np.mean(v)

    @property
    def _avg_velocity(self):
        """
        平均速度(成功用平均速度,失败记 0)
        注意:同样疑似有 bug:成功场景取 return_dicts[0] 而不是 return_dicts[i]。
        """
        v = []
        for i in range(len(self.dataset_evaluator.return_dicts)):
            if self.dataset_evaluator.return_dicts[i]['success'] is True:
                v.append(self.dataset_evaluator.return_dicts[0]['velocities'])
            else:
                v.append(0)

        return np.mean(v)

    @property
    def _avg_velocity_weighted(self):
        """
        加权平均速度:
        - 权重 w 是每个场景的 timesteps_agent
        - 速度 v 只统计成功场景
        注意:这里同样取 return_dicts[0]['velocities'](可能是 bug)
        """
        v = []
        for i in range(len(self.dataset_evaluator.return_dicts)):
            if self.dataset_evaluator.return_dicts[i]['success'] is True:
                v.append(self.dataset_evaluator.return_dicts[0]['velocities'])
        w = [
            return_dict["timesteps_agent"]
            for return_dict in self.dataset_evaluator.return_dicts
        ]
        s = np.sum(
            [
                return_dict["timesteps_agent"]
                for return_dict in self.dataset_evaluator.return_dicts
            ]
        )
        velocity = 0
        for x in range(len(v)):
            velocity += (v[x] * w[x]) / s
        return velocity

    def create_eval_files(self):
        """
        生成所有评估输出文件(按顺序):
        1) planner_statistic.log(总体统计)
        2) scenario_completion_list.json(成功/失败/无效列表)
        3) harm.json(伤害汇总)
        4) exec_timing.json(计时细节,若有)
        5) weights.json(模型权重)
        6) settings.json(运行设置)
        """
        """WIP."""
        self._create_eval_statistic()
        self._create_completion_list()
        self._create_harm_evaluation()
        self._create_eval_statistic_exec_times_dict()
        self._store_weights()
        self._store_settings()

    def _create_completion_list(self):
        """
        输出:scenario_completion_list.json
        结构:
        {
          "completed": [...],
          "failed": [...],
          "invalid": [...]
        }

        invalid 的判定逻辑:失败原因包含 "successor"
        (这可能表示 scenario 数据或 successor 关系有问题)
        """
        """WIP."""
        file_path = self.dataset_evaluator.eval_directory.joinpath(
            "scenario_completion_list"
        ).with_suffix(".json")

        eval_dict = {"completed": [], "failed": [], "invalid": []}
        for return_dict in self.dataset_evaluator.return_dicts:
            if return_dict["success"]:
                eval_dict["completed"].append(return_dict["scenario_path"])
            elif "successor" in return_dict["reason_for_failure"]:
                eval_dict["invalid"].append(return_dict["scenario_path"])
            else:
                eval_dict["failed"].append(return_dict["scenario_path"])

        with open(file_path, "w") as write_file:
            json.dump(eval_dict, write_file, indent=4)

    def _create_eval_statistic(self):
        """
        输出:planner_statistic.log
        内容包括:
        - 当前 git commit SHA(方便复现实验)
        - 评测数量、成功数量、完成率
        - 按时完成数量、按时完成率
        - 平均速度、加权平均速度
        - 总耗时、平均耗时
        - 失败日志列表
        """
        file_path = self.dataset_evaluator.eval_directory.joinpath(
            "planner_statistic"
        ).with_suffix(".log")
        repo = git.Repo(search_parent_directories=True)
        sha = repo.head.object.hexsha
        eval_list = [f"Commit: {sha}"]
        eval_list.append(f"Number of Evaluated Scenarios: {self._number_processed}")
        eval_list.append("")
        eval_list.append(
            f"Number of successfully driven scenarios: {self._number_successful}"
        )
        eval_list.append(f"Completion rate: {self._completion_rate:.2f} %")
        eval_list.append("")
        eval_list.append(
            f"Number of scenarios completed in time: {self._number_successful_in_time}"
        )
        eval_list.append(
            (f"In time completion rate: {self._in_time_completion_rate:.2f} %")
        )
        eval_list.append(f"Average velocity: {round(self._avg_velocity,2)} m/s")
        eval_list.append(
            f"Weighted average velocity: {round(self._avg_velocity_weighted,2)} m/s"
        )
        eval_list.append("\n")
        hours = int(self.dataset_evaluator.evaluation_time // 3600)
        mins = int((self.dataset_evaluator.evaluation_time % 3600) // 60)
        secs = int((self.dataset_evaluator.evaluation_time % 3600) % 60)
        eval_list.append(f"Execution time: {hours} h  {mins} min  {secs} sec")
        eval_list.append(
            f"Average execution time per scenario: {self._avg_exection_time_per_scenario:.2f} sec"
        )
        eval_list.append(
            "Average execution time per successfully completed scenario: "
            + f"{self._avg_exection_time_per_successfull_scenario:.2f} sec"
        )
        eval_list.append("\n")
        eval_list.append("Failure log:")
        eval_list.append("-------------------------------------------")
        eval_list.extend(self._fail_log)
        # 确保目录存在
        file_path.parent.mkdir(parents=True, exist_ok=True)
        # 写文件
        with open(file_path, "w") as file_obj:
            file_obj.write("\n".join(eval_list))

    def _create_eval_statistic_exec_times_dict(self):
        """
        输出:exec_timing.json(如果 return_dict 里包含 exec_times_dict)

        这个文件用于分析“程序都花时间在哪些步骤上”。
        它会:
        - 收集每个场景的 exec_times_dict(每个 key 对应多次耗时列表)
        - 合并成一个 dict:key -> 所有场景的耗时列表
        - 计算:总耗时占比、总耗时、调用次数、平均耗时
        - 再把 key 里带 "/" 的层级结构整理成嵌套字典,便于阅读
        """
        # Check if there are exec_times_dicts
        # 先检查第一个 return_dict 是否有 exec_times_dict
        if "exec_times_dict" in self.dataset_evaluator.return_dicts[0]:
            # Extract the dicts
            file_path = self.dataset_evaluator.eval_directory.joinpath(
                "exec_timing"
            ).with_suffix(".json")
            # 提取每个场景的 exec_times_dict
            exec_times_dicts = [
                return_dict["exec_times_dict"]
                for return_dict in self.dataset_evaluator.return_dicts
            ]
            # Merge list of dicts to dict of list
            # 1) 合并:list[dict] -> dict[list]
            # dict_merged[key] 是一个“列表”,里面装每个场景对应 key 的耗时列表
            dict_merged = {}
            for key in exec_times_dicts[0]:
                for exec_times_dict in exec_times_dicts:
                    if key in exec_times_dict:
                        if key not in dict_merged:
                            dict_merged[key] = []
                        dict_merged[key].append(exec_times_dict[key])

            # Merge the list of lists together
            # 2) 把 list of lists 展平:例如 [[0.1,0.2],[0.3]] -> [0.1,0.2,0.3]
            dict_merged = {
                key: [
                    inner_item for outer_item in list_item for inner_item in outer_item
                ]
                for key, list_item in dict_merged.items()
            }

            def eval_merged_dict(dict_merged):
                """
                把每个 key 的耗时列表统计成一句摘要字符串:
                - 占总耗时百分比
                - 总耗时
                - 调用次数
                - 平均每次耗时
                """
                out_dict = {}
                total_time = sum(dict_merged["total"])
                for key, item in dict_merged.items():
                    eval_list = []
                    eval_list.append(
                        f"Percentage from total: {100*sum(item)/total_time:.3f} %"
                    )
                    eval_list.append(f"Total time: {sum(item):.4f} s")
                    eval_list.append(f"Number of calls: {len(item)}")
                    eval_list.append(
                        f"Avg exec time per call: {sum(item)/len(item):.6f}"
                    )
                    out_dict[key] = " || ".join(eval_list)
                return out_dict

            evaluated_merged_dict = eval_merged_dict(dict_merged)

            def group_dict_recursive(input_rec_dict):
                """
                把 key 中包含 "/" 的层级结构变成嵌套 dict:
                例如:
                  "initialization/read scenario" -> {...}
                变成:
                  {"initialization": {"read scenario": "..."}}

                同时会检查一种“禁止情况”:
                不能既给 "super/stupid" 计时,又给 "super/stupid/example" 计时,
                因为父节点和子节点同时作为计时 key 会让层级结构冲突。
                """
                error_msg = (
                    "Do not use a label that has a parent used for timing\n\n"
                    + "Minimal example to reproduce this error:\n"
                    + """>>> with timer.time_with_cm("super/stupid"):\n"""
                    + ">>>     pass\n"
                    + """>>> with timer.time_with_cm("super/stupid/example"):\n"""
                    + ">>>     pass\n"
                )
                working_dict = {}
                for key, item in input_rec_dict.items():
                    split_key = key.split("/", 1)  # 只分割一次:前半层级 + 剩余部分
                    if len(split_key) == 1:
                        if split_key[0] in working_dict:
                            raise Exception(
                                error_msg + f"\nKey that threw error: {key}"
                            )
                        else:
                            working_dict[split_key[0]] = item
                    else:
                        if split_key[0] in working_dict:
                            try:
                                working_dict[split_key[0]][split_key[1]] = item
                            except TypeError as excp:
                                raise Exception(
                                    error_msg + f"\nKey that threw error: {key}"
                                ) from excp
                        else:
                            working_dict[split_key[0]] = {split_key[1]: item}
                # 递归把子 dict 再分层
                for key, item in working_dict.items():
                    if isinstance(item, dict):
                        working_dict[key] = group_dict_recursive(item)

                return working_dict

            grouped_dict = group_dict_recursive(evaluated_merged_dict)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "w") as file_obj:
                json.dump(grouped_dict, file_obj, indent=6)

    def _create_harm_evaluation(self):
        """
        输出:harm.json
        作用:把所有场景的 harm(伤害)累加起来,得到总 harm。

        注意:
        - harm 的字典结构在 ScenarioHandler 里定义过
        - 这里做的是简单求和(不是平均值)
        """
        harm = {
            "Ego": 0.0,
            "Unknown": 0.0,
            "Car": 0.0,
            "Truck": 0.0,
            "Bus": 0.0,
            "Bicycle": 0.0,
            "Pedestrian": 0.0,
            "Priority_vehicle": 0.0,
            "Parked_vehicle": 0.0,
            "Construction_zone": 0.0,
            "Train": 0.0,
            "Road_boundary": 0.0,
            "Motorcycle": 0.0,
            "Taxi": 0.0,
            "Building": 0.0,
            "Pillar": 0.0,
            "Median_strip": 0.0,
            "Total": 0.0,
        }
        # Check if collision is available
        # 检查 return_dict 是否包含 harm(正常应该有)      
        if "harm" in self.dataset_evaluator.return_dicts[0]:
            for return_dict in self.dataset_evaluator.return_dicts:
                for key in return_dict["harm"].keys():
                    harm[key] += return_dict["harm"][key]

        # Write harm dict to json
        file_path = self.dataset_evaluator.eval_directory.joinpath("harm").with_suffix(
            ".json"
        )

        with open(file_path, "w") as output:
            json.dump(harm, output, indent=6)

    def _store_weights(self):
        """
        输出:weights.json
        直接把 planner_creator.weights 保存下来,便于记录实验参数/复现。
        """
        file_path = self.dataset_evaluator.eval_directory.joinpath("weights").with_suffix(
            ".json"
        )
        weights = self.dataset_evaluator.scenario_evaluator.planner_creator.weights

        with open(file_path, "w") as output:
            json.dump(weights, output, indent=6)

    def _store_settings(self):
        """
        输出:settings.json
        把 planner_creator.settings 保存下来(运行配置)。

        注意一个细节:
        settings["frenet_settings"]["frenet_parameters"]["d_list"]
        可能是 numpy array(np.ndarray),json 不能直接序列化,所以要 .tolist() 转成普通 list。
        """
        file_path = self.dataset_evaluator.eval_directory.joinpath("settings").with_suffix(
            ".json"
        )
        settings = self.dataset_evaluator.scenario_evaluator.planner_creator.settings
        # 把 numpy array 转成 python list,才能 json.dump
        settings["frenet_settings"]["frenet_parameters"]["d_list"] = settings["frenet_settings"]["frenet_parameters"]["d_list"].tolist()

        with open(file_path, "w") as output:
            json.dump(settings, output, indent=6)
