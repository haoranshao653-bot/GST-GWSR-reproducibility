# GST-GWSR 完整代码：PyCharm运行说明

本包严格对应你最新贴出的GST算法要求。不是之前带空间分组、指数F或z-score回退的实验脚本。

## 1. 文件

- `run_gst_v2.py`：主入口。在PyCharm中运行此文件。
- `gst_v2.py`：新版局部任务、W/F/S、GP适配、种群迁移、cold probe及回退。
- `gst_gwsr_core.py`：现有项目核心代码的副本，提供GP参数、距离和预处理；不要单独运行它代替新版入口。
- `test_gst_v2.py`：单元测试（可选）。
- `requirements.txt`：依赖。

## 2. PyCharm操作

1. 解压本包，使用PyCharm打开解压目录；保留所有文件在同一目录。
2. 推荐创建Python 3.12虚拟环境。
3. 在PyCharm终端运行：

```text
python -m pip install -r requirements.txt
```

4. 打开 `run_gst_v2.py`，直接右键Run即可。默认读取原始数据：

```text
data/processed/heat_flow_stations.csv
```

默认输出位于工作目录下的 `results/GST_GWSR_ALL_GAUSSIAN_K25`。
已存在非空结果目录时程序会拒绝覆盖，请换一个输出目录。

也可在终端指定输入、输出和参数：

```text
python run_gst_v2.py --input "C:/你的路径/heat_flow_stations.csv" --output "C:/你的路径/GST_new_run" --alpha 0.5 --threshold 0.7 --seed 42
```

若要持久保存全部种群，加上 `--save-populations`。默认运行期间保留全部已完成种群，输出表保存每个任务的最佳公式。种群文件可能很大，joblib文件仅应从可信来源加载。

## 3. 固定逻辑

- 带宽由第25个最近训练样本的距离决定，全部训练样本用于高斯加权拟合。
- 权重归一化；零带宽以1e-12保护。
- 复用现有gplearn树、protected operators、随机初始化、竞赛选择、交叉、变异、复制。
- 原节点数复杂度惩罚系数为0.01；fitness越小越好。
- 取消80%样本抽取，每个个体按全部训练样本评价。
- W为归一化权重重叠；F为加权均值/标准差描述向量RMS距离的 `1/(1+d)`。
- S=alpha*W+(1-alpha)*F；按固定输入行顺序，只从全部已完成任务中选择单个最佳父任务。
- 无空间分组、无第三个地理评分、无Jaccard门槛。
- deepcopy并清除父fitness；子任务重新评分；无效/重复个体移除后用原随机树生成器补足种群。
- 短程搜索后与cold random probe的最佳惩罚fitness比较；必要时重新随机初始化完整搜索。
- fallback使用与Pure GWSR相同task的full_seed，保留失败原因、父任务、W/F/S。
- 预测接口不接收真实评价响应。主入口在两种方法全部完成后才计算评价指标。

默认参数：种群1000、完整搜索20代、短程5代、alpha=0.5、阈值0.7、种子42。
alpha和阈值仅是示例运行设置，本程序不自动调参，不声称它们最优。

## 4. 和旧代码相比必须注意

- 为符合指定Q目标，两种方法末代最佳个体均按惩罚fitness选择。旧版gplearn回归器默认按末代未惩罚误差输出。
- 复用现有算子，不新增精英保留或跨代hall-of-fame。完整20代含初始随机种群；短程5代在子任务重新评分后执行。
- 本版执行固定代数，不使用旧版raw MSE提前停止。日志记录实际完成的代数，probe及重新评分不记作进化代。
- 默认距离沿用旧项目经纬度欧氏距离。若决定用球面距离，添加 `--distance haversine_km`；不要混合报告两种设置。
- 预处理沿用原始80:20划分、训练集3IQR清洗、训练标准化、log响应变换。表达式在变换空间中，预测结果已逆变换。
- 第25邻居仅决定带宽，不代表有效样本量为25。
- 历史205个评价站点曾参与参数比较；本代码不使用它们调参，但也不能将它们变回从未使用过的测试集。

## 5. 输出

两个模型各自子目录：

- `task_log.csv`：逐任务父关系、相似度、迁移/回退、带宽、fitness、复杂度、种子、耗时、代数和评价次数。
- `expressions.csv`：变换空间局部表达式。
- `predictions.csv`：全部拟合结束后加入观测值、原单位预测值和残差。
- `task_weights.npz`：全部任务权重与训练/评价原始行编号。
- `populations.joblib`：仅指定 `--save-populations` 时保存。

总目录：

- `metrics_comparison.csv`：Pure/GST精度、耗时、fitness评价数、迁移情况对比。
- `prediction_comparison.png`：预测散点图。
- `run_summary.json`：完整参数、预处理参数、版本与结果汇总。

`n_fitness_evaluations` 为真正重新计算加权误差的次数，包含迁移准备、随机补充、probe、短程和fallback；同任务精确重复表达式可复用该任务评分。
`n_fitness_requests` 另计全部请求，包含重复命中。`n_tree_executions` 计训练数据上的实际树执行次数。
跨任务LRU缓存只保存float64预测，不保存fitness；新任务评分缓存清空。
缓存使用完整树结构和精确常数键，避免gplearn字符串中常数四舍五入引起错误命中。
为公平比较，Pure与GST各自使用新缓存。

## 6. 运行与测试状态

已通过11项小型合成数据单元测试。按用户要求，未运行本版真实数据完整实验。
因此还没有本版的预测精度、效率或迁移有效性结论。
你可直接正式运行；若报错，请发完整Traceback、Python版本与运行参数。

可选快速检查（不是论文实验）：

```text
python -m unittest test_gst_v2 -v
python run_gst_v2.py --smoke --output results/smoke_check
```

算法使用gplearn 0.4.3的内部接口，请不要随意升级该依赖。
当前顺序运行任务；评价适配器使用作用域内恢复的gplearn方法适配，不应同进程同时运行其他gplearn fit。
所有父种群保留在内存中；正式配置可能占用较多内存和较长时间。
