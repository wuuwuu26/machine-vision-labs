**实验课2：立体匹配与三维重建（MVSNet / DTU scan9）**  
基于 MVSNet 的多视图立体匹配（MVS）与三维重建实验，使用 DTU 数据集 scan9（49 视角）。  

> 仓库说明：为控制体积，DTU scan9 数据、BlendedMVS checkpoint 与融合 PLY 未提交；放置位置与下载方式见根目录 `README.zh-CN.md` 的“未纳入仓库的大文件”一节。  
   
 本报告在原始 MVSNet 流程基础上，实跑并分析了两个关键参数对结果的影响：  
- **深度估计阶段**：interval_scale（深度假设间隔）  
- **点云融合阶段**：num_consistent（几何一致性检查所需一致视图数）  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANUlEQVR4nO3OQQmAABRAsSfYxKK/kJXEkyE8WcGbCFuCLTOzVXsAAPzFsVZ3dX4cAQDgvesB/vEF9H9odtUAAAAASUVORK5CYII=)  
**1. 环境**  
| | |  
|-|-|  
| **项** | **说明** |   
| Python 环境 | conda 环境 **mvsnet3**（Python 3.8），可执行文件 /home/user/miniconda3/envs/mvsnet3/bin/python |   
| 深度学习框架 | nvidia-tensorflow 1.15.5+nv23.03（NVIDIA 维护的 TF1 CUDA 12 分支，支持 RTX 4070 的 sm_89 算力） |   
| 其他依赖 | numpy / matplotlib / opencv-python / scipy / python-docx（均在 mvsnet3 内） |   
| CUDA 工具链 | **离线 CUDA 12.4** 工具链，位于 /home/user/cuda124（含 nvcc、头文件、libcudart.so.12）。用于编译 fusibile 融合程序。CUDA 12.1 在 Ubuntu 24.04（gcc 13 / glibc 2.39）下前端不兼容（_Float32 未定义），故用 12.4。 |   
| 硬件 | NVIDIA GeForce RTX 4070 Laptop GPU（8 GB，算力 8.9 / sm_89） |   
   
***为什么不用官方环境*** *：MVSNet 官方要求 Python 2.7 + tensorflow-gpu 1.15（CUDA 10），其 cuBLAS 不含 sm_89 内核，无法在 RTX 4070 上做 GPU 运算。故新建 * *mvsnet3* *（Python 3.8 + NVIDIA TF1 分支），并对 MVSNet 代码做了少量兼容修改（* *basestring→str* *、* *xrange→range* *、* *scipy.misc→cv2* *、* *GroupNorm* * 整除等），网络结构与权重不变。*  
***GPU 限制*** *：阶段 1（深度估计）与 * *fusibile* * 融合都必须在有 GPU 的* ***宿主机*** *上运行；当前 WorkBuddy 沙箱未透传 * */dev/nvidia** *，只能跑阶段 2 的纯 CPU 部分（可视化 / 报告生成）。*  
***网络*** *：宿主机通过 * *10808* * 端口走代理（pip / apt 均适用）。原 * *install_fonts.sh* * / * *install_open3d.sh* * / * *99-cjk-songti.conf* * 等安装脚本已移出本目录；报告正文字体为 * ***Noto Serif CJK SC*** *，未安装时中文可能回退为无衬线字形。*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAABRAsSdYxKa/i8WMIR7ECt5E2BJsmZmt2gMA4C+Otbqr8+sJAACvXQ85PAYartXEogAAAABJRU5ErkJggg==)  
**2. 目录结构**  
rvclab2/  
 ├── lab2.docx                          # 最终实验报告（交付物，由 exp2_make_docx.py 生成）  
 ├── 实验课2-立体匹配与三维重建.pptx    # 课程原始任务说明  
 │  
 ├── MVSNet/                           # MVSNet 原版代码（TF1 仓库）  
 │   └── mvsnet/  
 │       ├── test.py                   # 阶段1入口：多视图深度估计（GPU）  
 │       ├── depthfusion.py            # ★ 原版融合入口：调 fusibile 二进制做概率滤波+几何融合  
 │       ├── preprocess.py             # PFM/PNG 读写、相机读写（load_pfm 需二进制句柄，已修正）  
 │       ├── model.py / homography_warping.py / loss.py / visualize.py / validate.py / train.py ...  
 │       └── cnn_wrapper/ tools/ doc/ README.md requirements.txt LICENSE  
 │  
 ├── fusibile/                        # ★ 原版 CUDA 点云融合程序（被 depthfusion.py 调用）  
 │   ├── fusibile.cu main.cpp *.h      # 源码  
 │   ├── CMakeLists.txt                # 已改为现代写法（project LANGUAGES CXX CUDA + CMAKE_CUDA_ARCHITECTURES 60 89）  
 │   ├── build/                        # 编译产物（含已编好的 fusibile 二进制）  
 │   ├── build_fusibile.sh             # 用离线 CUDA 12.4 工具链编译 fusibile  
 │   ├── run_depthfusion.sh            # ★ 跑原版流程：depthfusion.py --prob_threshold 0.8 → 产出 final3d_model.ply  
 │   └── run_param_sweep.sh            # ★ 报告 §3.3.4 真实数据来源：num_consistent∈{2,3,4,5} 原版融合扫描，输出 result_param_sweep/sweep_log.csv  
 │  
 ├── result_param_sweep/              # ★ 报告 §3.3.4 真实数据（由 run_param_sweep.sh 产出，须本机 GPU 跑）  
 │   ├── sweep_log.csv                 # 真实 fusibile 扫描结果：tag,prob,disp,num,points,bbox_x,bbox_y,bbox_z  
 │   └── nc{2,3,4,5}.ply               # 各 num_consistent 对应的真实融合点云（供复核 / CloudCompare 查看）  
 │  
 ├── run_stage1.sh                    # ★ 阶段1 入口：bash 跑 MVSNet 深度估计（需 GPU，宿主机运行）
 ├── tf_model_blendedmvs_200301/      # 预训练模型权重  
 │   └── tf_model_blendedmvs/3DCNNs/  
 │       └── model.ckpt-150000.{data,index,meta}   # BlendedMVS 预训练权重（ckpt_step=150000）  
 │  
 ├── dtu_test_scan9/                  # DTU 数据集 scan9  
 │   ├── scan9/  
 │   │   ├── images/                   # 49 视角参考图像（1600×1200）  
 │   │   ├── cams/                     # 相机内外参 + 每个视角的 .txt 标定（49 视角）  
 │   │   ├── pair.txt                  # 视角配对（最近邻选择，供代价体构建）  
 │   │   ├── depths_mvsnet/            # 阶段1输出：*.pfm 深度图 + 概率图（288×216，输入的 1/4）  
 │   │   │                             #   *_init.pfm / *_prob.pfm / *_prob_filtered.pfm（各 49 个）  
 │   │   ├── points_mvsnet/            # 阶段2输出（点云与中间结果）  
 │   │   │   ├── point_cloud_prob_filtered.ply  # ★ B 档：概率滤波后点云（融合前；报告图3左）  
 │   │   │   ├── consistencyCheck-*/final3d_model.ply  # ★ C 档：几何融合最终点云（报告图3右）  
 │   │   │   └── 2333__* /              # fusibile 的 gipuma 中间输入（disp.dmb / normals.dmb）  
 │   │   └── points_param_sweep/       # （旧）exp2_sweep_num_consistent.py 的 CPU 代理扫描输出（nc2..nc5），非报告 §3.3.4 数据来源  
 │  
 ├── exp2_depthfusion.py               # 生成 B 档 point_cloud_prob_filtered.ply（纯 numpy 反投影）  
 ├── exp2_visualize.py                 # 生成图1（输入/深度/概率）与图2（interval_scale 对比）  
 ├── exp2_sweep_num_consistent.py      # （旧）CPU 代理同构几何一致性融合的 num_consistent 扫描，仅供方法对照，非 §3.3.4 数据来源  
 ├── exp2_fig_num_consistent.py        # （旧）重画 CPU 代理的 num_consistent 参数影响图；报告图4 现由 exp2_make_docx.py 直接读真实 CSV 生成  
 ├── exp2_metrics_2tier.py             # 解析 PLY 计算两档点云真实指标（点数/包围盒/离群/点距std）  
 ├── exp2_compose_cc.py                # 生成图3：将两张 ccViewer 截图左右拼合为图3（CloudCompare 截图版）  
 ├── exp2_make_docx.py                 # ★ 生成实验报告 docx（报告内容内置于此脚本）  
 ├── exp2_page_estimate.py             # 估算 docx 页数（长度估算，非真实排版）  
 ├── view_pointcloud.py                # open3d 点云查看器（--all / --file / --render）  
 │  
 └── report_figures/                  # 报告用图与数据  
     ├── fig_inputs_depth_prob.png     # 图1：参考图 / 深度图 / 概率图  
     ├── fig_param_interval_scale.png  # 图2：不同 interval_scale 的深度图与分布  
     ├── fig_param_num_consistent.png  # 图4：num_consistent 对融合点云的影响（双面板）  
     ├── fig_pointcloud_cc.png         # 图3：融合前后两档点云对比（ccViewer 截图拼合）  
     ├── cc_prob_filtered.png          # 图3左源图：ccViewer 中 prob_filtered 点云截图  
     ├── cc_final3d_model.png          # 图3右源图：ccViewer 中 final3d_model 点云截图  
     └── param_nc_results.json         # （旧）exp2_sweep_num_consistent.py 的 CPU 代理数值结果，非报告 §3.3.4 数据来源  
   
***两档流程说明（与课件原版一致）*** *：课件原版 * *depthfusion.py* * 只产出两档——概率滤波后的深度（* **_prob_filtered.pfm* *）与几何一致性 + 多视融合后的最终点云（* *final3d_model.ply* *），* ***不含"朴素点云"档*** *。报告图 3 / §3.3 据此展示 B（概率滤波后，由 * *point_cloud_prob_filtered.ply* * 表示）+ C（融合后，由 * *final3d_model.ply* * 表示）两档。B 档由 * *exp2_depthfusion.py* * 反投影生成，C 档由 * *fusibile* * 生成。*  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANElEQVR4nO3OQQmAUBBAwSf8GGLWDWFDY3ixgjcRZhLMNjNHdQYAwF9cq1rV/vUEAIDX7gcRXAQ2s/16gwAAAABJRU5ErkJggg==)  
**如何运行 → 见 运行说明.md**

详细的运行步骤（fusibile 编译、阶段 1 深度估计、阶段 2 点云融合、num_consistent 参数扫描、可视化图与报告生成）已拆到独立文档 **运行说明.md**。环境要求见上方 §1。
**3. 关键结论速览**  
- **深度估计**：interval_scale=1.06（默认）使 128 个深度假设恰好覆盖场景深度范围，兼顾覆盖与分辨率；过小（0.8）导致远区被钳位、深度低估，过大（1.3）导致分辨率下降、远景块状量化。  
- **点云生成（原版两档）**：概率滤波剔除约 33% 低置信度像素（304.8 万 → 203.2 万有效点）；几何一致性 + 融合再降至 171.0 万（−15.9%）。融合后包围盒由 482×517×456 mm 收缩为 293×401×377 mm，离群点比例由 11.2% 降至 7.9%，点距 std 由 46.1 降至 42.9 mm。  
- **参数权衡**：num_consistent 越大，融合点数与完整性单调下降、包围盒体积收缩（具体数值见 §3.3.4 表格，由真实 fusibile 扫描 run_param_sweep.sh 给出），即被剔除的主要是远离表面的外点——体现「完整性 ↔ 可靠性」权衡。  
![](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAnEAAAACCAYAAAA3pIp+AAAABmJLR0QA/wD/AP+gvaeTAAAACXBIWXMAAA7EAAAOxAGVKw4bAAAANklEQVR4nO3OYQ1AABSAwc8mi5wvkwZyCKCAACr4Z7a7BLfMzFYdAQDwF+da3dX+9QQAgNeuB6feBdUJcyS2AAAAAElFTkSuQmCC)  
**4. 本版本已清理的内容**  
为保持目录只含当前版本所需文件，已删除以下非必需内容：  
- 临时/锁文件：.~lock.实验报告…docx#、.Trash-0/（沙箱回收站残留）。  
- 备份与缓存：fusibile/CMakeLists.txt.orig、各 __pycache__/、MVSNet/mvsnet/events.out.tfevents.*（上游 TensorBoard 日志）。  
- 被本报告取代的旧图/脚本：report_figures/fig_pointcloud_2tier.png 与 exp2_fig_pointcloud_2tier.py（matplotlib 3D 散点版图3），已替换为 CloudCompare 截图版 fig_pointcloud_cc.png（由 exp2_compose_cc.py 从 cc_prob_filtered.png / cc_final3d_model.png 拼合）。  
- 旧三档流程产物（与课件原版两档流程不符，且未写入报告）：points_mvsnet/point_cloud_naive.ply、point_cloud_fused.ply、stats.npy。  
- 外部 CUDA 冗余（位于 /home/user，非本目录）：CUDA 12.1 工具链与 cuda_12.4.0…linux.run 安装包已清理，仅保留可用的 /home/user/cuda124。  
- 中途尝试的 exp2_sweep_nc_faithful.py：试图逐像素复刻 fusibile.cu 的视差空间一致性检验，但 fusibile 运行时会归一化相机（有效焦距尺度与原字面读取不一致），其数值（≈1.5% 保留率）与原版不符，已删除；§3.3.4 现直接采用原版 fusibile 的真实扫描（fusibile/run_param_sweep.sh → result_param_sweep/sweep_log.csv），分析对照原版源码 fusibile.cu（融合核 206–253 行）与 depthfusion.py 的调用关系。  
