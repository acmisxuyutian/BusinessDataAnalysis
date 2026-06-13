# 基于大语言模型的学生用户画像构建与个性化推荐方法

本教程通过真实教育数据集，完整演示两大核心主题：

1. **用户画像构建**：传统规则方法 vs 大语言模型方法
2. **个性化推荐**：协同过滤算法 vs LLM+向量检索方法

每个主题都遵循相同的叙事逻辑：**传统方法基线 → 暴露缺陷 → LLM方法 → 对比验证**。

## 快速启动

### 1. 创建虚拟环境

```bash
conda create -n student-profile python=3.9 -y
conda activate student-profile
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置中文字体（Linux/WSL环境）

项目根目录已包含 `SimHei.ttf` 字体文件，Notebook会自动加载，无需额外配置。

> **Windows用户**：系统自带SimHei字体，无需额外配置。

### 4. 启动Jupyter

```bash
jupyter notebook "基于大语言模型的学生用户画像构建和个性化推荐方法.ipynb"
```

### 5. 配置通义千问API（可选）

如需使用真实的LLM画像生成和推荐（而非模拟输出），需配置API Key：

```bash
# Linux/Mac
export DASHSCOPE_API_KEY="your-api-key-here"

# Windows PowerShell
$env:DASHSCOPE_API_KEY="your-api-key-here"
```

API Key申请地址：https://dashscope.console.aliyun.com/

## 项目结构

```
数智教育数据集/
├── README.md                                        # 本文件
├── requirements.txt                                 # Python依赖包
├── SimHei.ttf                                       # 中文字体文件
├── 基于大语言模型的学生用户画像构建和个性化推荐方法.ipynb  # 主教程Notebook
├── output/                                          # 可视化图表输出
└── 数智教育数据集/                                   # 原始数据
    ├── 1_teacher.csv                                # 教师信息表
    ├── 2_student_info.csv                           # 学生信息表
    ├── 3_kaoqin.csv                                 # 考勤记录表
    ├── 4_kaoqintype.csv                             # 考勤类型表
    ├── 5_chengji.csv                                # 成绩记录表
    ├── 6_exam_type.csv                              # 考试类型表
    ├── 7_consumption.csv                            # 消费记录表
    └── 数据说明.txt                                  # 数据字段说明
```

## 教程内容

### 第一部分：画像构建

| 章节 | 内容 | 核心教学点 |
|------|------|------------|
| 第1章 | 项目概述与环境准备 | 数据集说明、通义千问API配置 |
| 第2章 | 数据加载与预处理 | 7表关联、异常值清洗、多维特征聚合 |
| 第3章 | 传统方法构建画像 | 5维度规则标签（学业/学科/考勤/消费/综合） |
| 第4章 | **传统方法缺陷** | 3个缺陷的代码实验证据（D1-D3） |
| 第5章 | **LLM方法构建画像** | Prompt设计、通义千问API调用、结构化输出 |
| 第6章 | 对比分析与可视化 | 并排对比、雷达图、关键词分析、总结表 |
| 第7章 | 画像构建部分总结 | 核心结论、局限性、最佳实践、教育学理论扩展 |

### 第二部分：个性化推荐

| 章节 | 内容 | 核心教学点 |
|------|------|------------|
| 第8章 | **协同过滤推荐算法** | User-based CF、Item-based CF、评分矩阵构建 |
| 第9章 | **协同过滤的缺陷** | 7个缺陷的代码实验证据（CF1-CF7） |
| 第10章 | 向量数据库基础 | Embedding、余弦相似度、Chroma快速上手 |
| 第11章 | **LLM+画像个性化推荐** | 资源库构建、语义检索、LLM推荐报告生成 |
| 第12章 | 推荐方法对比与总结 | 三种方法完整对比、推荐系统三阶段演进、流程回顾、扩展方向 |
| 第13章 | 参考文献与扩展学习 | 9篇前沿论文（KDD/ACL/SIGIR/CIKM等）、综述论文 |

## 画像构建：传统方法的3个缺陷

| 编号 | 缺陷 | 说明 |
|------|------|------|
| D1 | 维度孤立 | 各维度独立计算，无法交叉分析 |
| D2 | 语义缺失 | 仅输出结构化标签，丢失行为模式信息 |
| D3 | 输出不可读 | 标签拼接无法直接用于沟通 |

## 推荐系统：协同过滤的7个缺陷

| 编号 | 缺陷 | 说明 |
|------|------|------|
| CF1 | 冷启动问题 | 新学生无数据，推荐完全失效 |
| CF2 | 数据稀疏性 | 矩阵大量缺失，相似度不可靠 |
| CF3 | 推荐同质化 | 只推荐热门学科，缺乏多样性 |
| CF4 | 缺乏语义理解 | 只看数值，不理解"为什么"推荐 |
| CF5 | 无法利用画像 | 考勤、消费、行为模式被完全忽略 |
| CF6 | 无法生成推荐理由 | 只能输出"推荐学科X"，无法解释 |
| CF7 | 可扩展性差 | 学生数增长时，计算量O(n²)爆炸 |

## 技术栈

- **数据处理**：pandas, numpy
- **可视化**：matplotlib, seaborn, wordcloud
- **机器学习**：scikit-learn（协同过滤、PCA降维）
- **向量数据库**：Chroma（语义检索）
- **大语言模型**：通义千问（Qwen）via OpenAI兼容API
- **开发环境**：Jupyter Notebook

## 常见问题

### 中文字体显示为方块

Notebook会自动加载项目目录下的 `SimHei.ttf` 字体文件。如仍有问题：

```bash
python -c "import matplotlib.font_manager as fm; fm._load_fontmanager(try_read_cache=False); print([f.name for f in fm.fontManager.ttflist if 'Hei' in f.name])"
```

### API调用失败

- 检查API Key是否正确设置
- 检查网络连接
- Notebook会自动降级为模拟输出，不影响教程演示

### Chroma向量数据库

Chroma使用内存模式运行，无需启动服务器。首次运行时会自动下载Embedding模型。

## 参考文献

本教程第13章列出了9篇前沿论文作为扩展学习资料，涵盖：

- **用户画像构建**：KDD 2025、ACL 2024、CIKM 2025 等
- **基于LLM的推荐系统**：SIGIR 2025、ACL Findings 2025、NCA 2024 等
- **综述论文**：World Wide Web 2023、EMNLP Findings 2025

详见 Notebook 第13章。

## 许可说明

本教程仅用于教学演示目的。数据集已做脱敏处理。
