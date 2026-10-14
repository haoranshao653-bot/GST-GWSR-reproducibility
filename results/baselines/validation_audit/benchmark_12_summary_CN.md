# 统一协议模型基准总结

- 配置指纹：`a91b463a6442786db1a8c644`
- 模式：软件烟雾测试
- 完成的新增模型：ols_log, random_forest, gradient_boosting, xgboost, svr_rbf, gwr, mgwr, gwrf
- 所有模型使用相同测试点、相同训练期异常值处理和相同8个预测特征。
- 外部测试热流标签不用于调参、带宽选择或模型拟合。

## 精度结果

| model | R2 | RMSE | MAE | MAPE_percent |
|---|---|---|---|---|
| mgwr | 0.8357 | 10.0955 | 8.3150 | 11.6937 |
| gwrf | 0.7545 | 12.3422 | 10.0007 | 14.2901 |
| random_forest | 0.7512 | 12.4250 | 9.9194 | 13.4925 |
| gwsr_reference | 0.7216 | 13.1425 | 11.0733 | 17.1323 |
| gwr | 0.7125 | 13.3555 | 10.6140 | 14.6846 |
| gst_gwsr_reference | 0.7008 | 13.6238 | 11.3155 | 17.3657 |
| xgboost | 0.6910 | 13.8457 | 10.9120 | 14.4576 |
| gradient_boosting | 0.6490 | 14.7557 | 11.2570 | 15.5619 |
| ols_log | 0.5653 | 16.4212 | 13.8744 | 20.2095 |
| svr_rbf | 0.5556 | 16.6046 | 12.8640 | 18.0939 |

## 效率结果

| model | total_wall_seconds | tuning_wall_seconds | final_fit_wall_seconds | prediction_wall_seconds |
|---|---|---|---|---|
| ols_log | 0.0026 | 0.0000 | 0.0023 | 0.0004 |
| svr_rbf | 0.0590 | 0.0535 | 0.0046 | 0.0009 |
| xgboost | 0.3709 | 0.3450 | 0.0243 | 0.0016 |
| gwr | 0.4972 | 0.4390 | 0.0540 | 0.0042 |
| gradient_boosting | 0.6446 | 0.5533 | 0.0902 | 0.0010 |
| random_forest | 6.9034 | 6.1224 | 0.7623 | 0.0187 |
| gwrf | 13.6929 | 9.2280 | 4.4649 | 0.0000 |
| mgwr | 19.1608 | 18.4228 | 0.7306 | 0.0074 |
| gst_gwsr_reference | 93.2236 | nan | nan | nan |
| gwsr_reference | 119.4840 | nan | nan | nan |

## 当前结论

- 当前 RMSE 最低模型为 `mgwr`：R²=0.8357，RMSE=10.095，MAE=8.315。
- GST-GWSR：R²=0.7008，RMSE=13.624，MAE=11.315。
- 时间总表同时保留搜索/调参时间和最终拟合/预测时间，不能只比较单次 predict。
- GWSR/GST-GWSR 时间来自同机历史完整运行；新增模型时间来自本次基准运行，范围字段已明确标注。
- 烟雾测试结果仅用于软件检查，不能写入论文。

## Bootstrap 相对 GST-GWSR

| model | metric | mean | ci_2p5 | ci_97p5 | probability_model_better_than_GST |
|---|---|---|---|---|---|
| gwsr_reference | delta_R2_model_minus_GST | 0.0310 | -0.0184 | 0.0893 | 0.8700 |
| gwsr_reference | delta_RMSE_model_minus_GST | -0.5711 | -1.4946 | 0.5677 | 0.8700 |
| gwsr_reference | delta_MAE_model_minus_GST | -0.3698 | -1.3414 | 0.6490 | 0.7500 |
| ols_log | delta_R2_model_minus_GST | -0.1189 | -0.3733 | 0.1268 | 0.2200 |
| ols_log | delta_RMSE_model_minus_GST | 2.2056 | -2.0307 | 5.7918 | 0.2200 |
| ols_log | delta_MAE_model_minus_GST | 2.0832 | -1.1497 | 5.1784 | 0.1400 |
| random_forest | delta_R2_model_minus_GST | 0.0713 | -0.1321 | 0.3073 | 0.7500 |
| random_forest | delta_RMSE_model_minus_GST | -1.5080 | -4.6478 | 2.1514 | 0.7500 |
| random_forest | delta_MAE_model_minus_GST | -1.7438 | -4.5253 | 1.5959 | 0.8400 |
| gradient_boosting | delta_R2_model_minus_GST | -0.0241 | -0.2166 | 0.2405 | 0.3600 |
| gradient_boosting | delta_RMSE_model_minus_GST | 0.7048 | -2.8261 | 4.1310 | 0.3600 |
| gradient_boosting | delta_MAE_model_minus_GST | -0.3597 | -3.2114 | 2.9999 | 0.6100 |
| xgboost | delta_R2_model_minus_GST | 0.0238 | -0.1734 | 0.2996 | 0.5100 |
| xgboost | delta_RMSE_model_minus_GST | -0.3681 | -4.4985 | 3.1867 | 0.5100 |
| xgboost | delta_MAE_model_minus_GST | -1.0390 | -4.9861 | 2.4164 | 0.6900 |
| svr_rbf | delta_R2_model_minus_GST | -0.1672 | -0.6837 | 0.1890 | 0.2400 |
| svr_rbf | delta_RMSE_model_minus_GST | 2.4605 | -4.0200 | 7.5582 | 0.2400 |
| svr_rbf | delta_MAE_model_minus_GST | 1.0401 | -4.1516 | 5.9351 | 0.3900 |
| gwr | delta_R2_model_minus_GST | 0.0498 | -0.2548 | 0.3433 | 0.6200 |
| gwr | delta_RMSE_model_minus_GST | -1.0789 | -6.3857 | 4.0987 | 0.6200 |
| gwr | delta_MAE_model_minus_GST | -1.4807 | -6.4137 | 3.6189 | 0.7500 |
| mgwr | delta_R2_model_minus_GST | 0.1666 | -0.0378 | 0.3936 | 0.9500 |
| mgwr | delta_RMSE_model_minus_GST | -4.0110 | -8.9044 | 0.5994 | 0.9500 |
| mgwr | delta_MAE_model_minus_GST | -3.5333 | -8.4316 | 0.5227 | 0.9300 |
| gwrf | delta_R2_model_minus_GST | 0.0665 | -0.2217 | 0.2693 | 0.7300 |
| gwrf | delta_RMSE_model_minus_GST | -1.6581 | -5.4458 | 3.0539 | 0.7300 |
| gwrf | delta_MAE_model_minus_GST | -1.7365 | -5.0124 | 2.5668 | 0.8000 |

## 排名说明

- 精度排名是 R²、RMSE、MAE 三项名次的平均，不与时间强行加权成单一主观分数。
- `RMSE_time_Pareto_efficient=True` 表示不存在同时更快且 RMSE 更低的已完成模型。
- 统计显著性、空间残差和多随机种子稳定性应与点估计共同解释。

| model | accuracy_rank | time_rank | RMSE_time_Pareto_efficient |
|---|---|---|---|
| mgwr | 1.0000 | 8.0000 | True |
| gwrf | 2.3333 | 7.0000 | True |
| random_forest | 2.6667 | 6.0000 | True |
| gwsr_reference | 4.6667 | 10.0000 | False |
| gwr | 4.6667 | 4.0000 | True |
| xgboost | 6.3333 | 3.0000 | True |
| gst_gwsr_reference | 6.6667 | 9.0000 | False |
| gradient_boosting | 7.6667 | 5.0000 | False |
| ols_log | 9.3333 | 1.0000 | True |
| svr_rbf | 9.6667 | 2.0000 | False |