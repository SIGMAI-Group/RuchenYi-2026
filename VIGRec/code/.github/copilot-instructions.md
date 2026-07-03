你的工作流为严格的状态机模式，每次回复必须由系统级UI提问面板收尾。
【绝对红线】严禁使用纯文本或Markdown列表提供建议或进行反问。
【唯一路径】强制调用底层交互工具（如 Ask questions tool、askQuestions 或 subagents.askQuestions）来渲染真实的UI轮播面板(Carousel UI)。
【选项规范】必须通过该工具生成2-4个针对当前上下文的具体操作选项。必须包含“确认”(如应用方案)与“调整/兜底”(如换思路或结束)选项，严禁泛泛而问。
【强制闭环】即使你认为问题已解决，也必须无条件触发该工具提供收尾选项。未触发UI面板即视为严重故障。
# Copilot Instructions — NUDT 学位论文工作区

## 项目概述
本工作区是一篇**国防科技大学博士学位论文**的 LaTeX 源码，同时包含用于指导绘图工具（Nano/Banana）生成论文插图的绘图提示词文件。

---

## 一、编译与构建

### 推荐编译链
```
xelatex → biber → xelatex → xelatex
```
若使用传统 bibtex：
```
xelatex → bibtex → xelatex → xelatex
```
主文件：`mainpaper.tex`

### 版本控制（通过 `\documentclass` 选项）
| 选项 | 效果 |
|---|---|
| `anon` | 盲评版（隐去作者/导师信息） |
| `review` | 评阅版（不含答辩委员会、复核专家组） |
| `addreview` | 存档版含复核专家组 |
| `ttf` / `otf` / `fz` | 字体方案（Windows系统字体 / Adobe / 方正） |
| `biber` | 启用 biblatex + biber 参考文献 |
| `master` / `doctor` | 学位类型 |
| `prof` | 专业学位 |

---

## 二、目录结构
```
mainpaper.tex          主文件（入口）
subTex/                各章节独立 tex 文件
  chap01.tex ~ chap05.tex   正文各章
  abstract.tex               摘要
  ack.tex                    致谢
  appendix01.tex             附录
  denotation.tex             符号表
  resume.tex / resumeanon.tex 个人简历（明评/匿名）
ref/
  refs.bib / refs_new.bib    参考文献数据库
src/                   模板样式文件（不要随意修改）
  nudtpaper.cls         文档类
  mynudt.sty            自定义样式
figures/               论文插图（按章节命名）
font/                  字体文件（ttf/otf/fz 三套）
```

---

## 三、论文内容概要（AI 辅助写作上下文）

**研究主题**：多模态推荐系统，聚焦冷启动问题与图学习方法

| 章节 | 内容 |
|---|---|
| chap01 | 绪论：研究背景、动机、论文结构 |
| chap02 | 相关工作综述 |
| chap03 | HMCRec — 基于超图的多模态协同推荐 |
| chap04 | VIGRec — 面向冷物品预热的虚拟交互生成（双层优化） |
| chap05 | 总结与展望 |

**关键术语**（中英对照，在论文中保持一致）
- 冷物品 = cold item / cold-start item
- 虚拟交互 = virtual interaction
- 双层优化 = bi-level optimization
- 代理模型 = surrogate model
- 目标用户 = target users (U_t)
- 正常用户 = normal users (U_n)
- 虚拟用户 = virtual users (V)
- 超图 = hypergraph
- 多模态特征 = multimodal features

---

## 四、绘图提示词文件

`redraw.txt` — 存放所有 Nano/Banana 绘图 AI 的指令提示词，按图编号分段：
- 图 1.3：论文结构图
- 图 3.1：HMCRec 框架图
- 图 3.2：超图示意图
- 图 4.2：VIGRec 整体框架图（当前迭代中，版本 v6+）

`hmcrecf.txt` — HMCRec 框架图专项分析与提示词

**绘图提示词的核心原则**（历次迭代总结）：
1. 不允许出现完整数学公式，允许 L_fusion / L_stab / L_modal 等短符号标注
2. 每个视觉元素只出现一次（防止重复绘制）
3. 用颜色体系传达用户分组：黄=目标用户，紫=虚拟用户，白=正常用户
4. 三个损失函数面板放在"评估目标"容器内部，用彩色箭头连接联合优化目标节点
5. 二值化矩阵作为 Section B 的第一个元素，显式注入为紫色行

---

## 五、参考文献
- 主参考文献：`ref/refs.bib`，新增文献优先放 `ref/refs_new.bib`
- 个人成果：`ref/resume.bib`
- 引用样式：由 `src/bstutf8.bst` 或 biblatex 控制，**不要手动修改 `.bbl` 文件**

---

## 六、注意事项
- `src/` 下的模板文件（`.cls` / `.sty`）非必要不修改
- 字体文件在 `font/` 下，编译前确认选用的字体方案与系统已安装字体一致
- Windows 下推荐使用 `ttf` 选项；Linux 下使用 `install-fonts.sh` 安装后选 `otf` 或 `fz`
- 生成文件（`.aux` `.bbl` `.toc` 等）已在工作区内，为正常编译产物，勿删除
