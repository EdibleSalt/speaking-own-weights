% make_paper_figs.m  ——  论文数据图 (base MATLAB, 无需任何 toolbox)
% 数据硬编码自 results/data/**/summary.json + _analysis_report.md (权威值)。
% 输出: PNG(300dpi, markdown 用) + PDF(矢量, 论文用) 到 paper_figs/(ASCII 路径)。
% 标签用英文(避免字体缺字), 中文图注写在 .md 里。

outdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'paper_figs');
if ~exist(outdir,'dir'); mkdir(outdir); end
FS = 11;  % font size
close all; set(0,'DefaultFigureVisible','off');  % 离屏渲染, 避开 JOGL/OpenGL 显示锁; exportgraphics 仍正常存盘

%% Fig 1 —— 言说鸿沟: 内部可解码上界 vs 零样本可言说 (末两根=主力模型, 高亮)
% 探针上界 = 各模型非平凡 prompt(排除 P4 "{word}" 单词裸读的平凡上界)的最佳层 RSA。
% pythia-1.4b(0.629)与 OLMo-Instruct(0.679)是后续主力(untied), 高亮区分。
ceil8 = [0.442 0.546 0.452 0.580 0.516 0.665 0.629 0.679];   % 各模型 best 非-P4 prompt 的最佳层(口径统一)
names = {'Qwen2.5-0.5B','Qwen2.5-3B','Qwen3-1.7B','Llama3.2-3B','Gemma3-1B','SmolLM3-3B','pythia-1.4B','OLMo2-1B-Inst'};
zshot = 0.011;
f1 = figure('Color','w','Position',[100 100 770 380]);
hb = bar(ceil8,0.62,'FaceColor','flat'); hold on;
hb.CData = repmat([0.62 0.66 0.74],8,1);     % 非主力: 淡蓝灰
hb.CData(7,:) = [0.85 0.45 0.20];            % pythia-1.4b (主力, 橙)
hb.CData(8,:) = [0.20 0.45 0.70];            % OLMo2-1B-Inst (主力, 蓝)
yline(zshot,'--r','LineWidth',1.6);
text(0.6, zshot+0.035, sprintf('zero-shot verbalization \\approx %.2f', zshot), 'Color','r','FontSize',FS-1);
set(gca,'XTick',1:8,'XTickLabel',names,'FontSize',FS-1); xtickangle(30);
ylabel('RSA (probe-decoded vs true embedding)','FontSize',FS); ylim([0 0.80]);
title('Verbalization gap: internally decodable (\approx0.5) vs spoken (\approx0.01)','FontSize',FS);
text(7.5, 0.77, 'main models (untied)','Color',[0.2 0.2 0.2],'FontSize',FS-2,'HorizontalAlignment','center','FontWeight','bold');
grid on; box off; save_fig(f1,'fig1_gap',outdir);

%% Fig 2 —— 微调桥接 (跨 3 模型, held-out, 3-seed mean±std; 行=模型, 列=目标)
models2 = {'OLMo-Instruct','OLMo-base','pythia-1.4b'};
Y = [0.502 0.484 0.655;    % OLMo-Instruct: input-embed / unembedding / mid-hidden(L12)
     0.390 0.410 0.657;    % OLMo-base: mid-hidden(L12) 3-seed=0.657 (健康, 与 instruct 同档)
     0.226 0.317 0.021];   % pythia: mid-hidden(L12) 3-seed=0.021 (该层塌; 但末层 L24=0.475)
E = [0.011 0.034 0.023;
     0.030 0.120 0.028;
     0.013 0.142 0.011];
f2 = figure('Color','w','Position',[100 100 720 400]); hold on;
b = bar(Y,'grouped');
b(1).FaceColor=[0.20 0.45 0.70]; b(2).FaceColor=[0.85 0.55 0.25]; b(3).FaceColor=[0.45 0.65 0.45];
for k=1:size(Y,2)
    errorbar(b(k).XEndPoints, Y(:,k), E(:,k),'k','LineStyle','none','LineWidth',1.0,'CapSize',6);
end
% pythia 中间层 L12(网络中点)塌, 但其末层 L24 读得出 —— 紧挨着画一根 L24 柱(紫)表示, 防误读
dx = b(3).XEndPoints(3) - b(2).XEndPoints(3);
xL24 = b(3).XEndPoints(3) + dx;
hL24 = bar(xL24, 0.475, dx*0.9, 'FaceColor',[0.48 0.30 0.62]);
text(xL24, 0.498, '0.48', 'HorizontalAlignment','center','FontSize',FS-3,'Color',[0.40 0.25 0.55]);
text(xL24, 0.085, 'L24', 'HorizontalAlignment','center','FontSize',FS-3,'Color','w','Rotation',90);
xlim([0.5 xL24+0.45]);
yline(0.014,'--','random-label ctrl \approx 0','Color',[0.45 0.45 0.45],'LineWidth',1.3,'FontSize',FS-2);
set(gca,'XTick',1:3,'XTickLabel',models2,'FontSize',FS-1);
ylabel('held-out RSA','FontSize',FS); ylim([0 0.76]);
legend([b(1) b(2) b(3) hL24], {'input-embed','unembedding','mid-hidden (L12)','pythia mid (L24, best layer)'},'Location','northeast','FontSize',FS-2);
title('LoRA bridges the gap across models (held-out)','FontSize',FS);
grid on; box off; save_fig(f2,'fig2_bridge',outdir);

%% Fig 3 (old two-panel 'decompose') RETIRED: split into the layer-1 residual
%% figure and the layer-2 three-panel figure -> see make_fig_layer12.m
%% (outputs fig_g1_residual / fig_g2_semantic). Kept out of this script so
%% rerunning it cannot regenerate the superseded fig3_decompose.

%% Fig 5 (出现顺序) —— shared readout: 单 adapter 读两个正交空间, OLMo vs pythia (双 panel)
% 每个通道与"它自己的" swap 对照并列: input-embed↔swap_input_vs_lh, unembedding↔swap_output_vs_ie。
% 数据 = C6_mixed_target/<model>_r{50,70,90}_tag_s0/summary.json (tag prompt, seed 0)。
ratios = {'50:50','70:30','90:10'};
% 列: input-embed 读出 / 其 swap 对照 / unembedding 读出 / 其 swap 对照
Yo5 = [0.325 0.085 0.287 0.136;   % OLMo   50:50
       0.426 0.098 0.200 0.133;   %        70:30
       0.498 0.148 0.084 0.094];  %        90:10
Yp5 = [0.209 0.053 0.156 0.121;   % pythia 50:50
       0.202 0.054 0.031 0.104;   %        70:30
       0.267 0.120 0.070 0.184];  %        90:10
R5 = [0.502 0.484; 0.226 0.317];   % 单目标(非混合)FT 参照: input-embed / unembedding, 每模型一行 (来自 fig2 桥接)
cIE=[0.20 0.45 0.70]; cIEc=[0.68 0.78 0.89]; cLH=[0.85 0.45 0.20]; cLHc=[0.93 0.80 0.66];
f4 = figure('Color','w','Position',[100 100 1120 430]);
P5 = {Yo5,'(a) OLMo'; Yp5,'(b) pythia'};
for p=1:2
    subplot(1,2,p); hold on;
    bb = bar(P5{p,1},0.92,'grouped');
    bb(1).FaceColor=cIE; bb(2).FaceColor=cIEc; bb(3).FaceColor=cLH; bb(4).FaceColor=cLHc;
    if p==1   % OLMo: input-embed 蓝/标在线上方, unembedding 橙/标在线下方(同图例色)
        yline(R5(1,1),'--','Color',cIE,'LineWidth',1.5,'Label','input-embed single-target  0.50','FontSize',FS-4,'LabelHorizontalAlignment','left','LabelVerticalAlignment','top');
        yline(R5(1,2),'--','Color',cLH,'LineWidth',1.5,'Label','unembedding single-target  0.48','FontSize',FS-4,'LabelHorizontalAlignment','left','LabelVerticalAlignment','bottom');
    else      % pythia: 两条参照各自标注, 都摆在线上方
        yline(R5(2,2),'--','Color',cLH,'LineWidth',1.5,'Label','unembedding single-target  0.32','FontSize',FS-4,'LabelHorizontalAlignment','left','LabelVerticalAlignment','top');
        yline(R5(2,1),'--','Color',cIE,'LineWidth',1.5,'Label','input-embed single-target  0.23','FontSize',FS-4,'LabelHorizontalAlignment','left','LabelVerticalAlignment','top');
        legend(bb,{'input-embed readout','  its swap-control','unembedding readout','  its swap-control'},'Location','northeast','FontSize',FS-4);
    end
    set(gca,'XTick',1:3,'XTickLabel',ratios,'FontSize',FS-1);
    xlabel('training ratio  (input-embed : unembedding)','FontSize',FS-2);
    ylabel('held-out RSA','FontSize',FS-1); ylim([0 0.60]);
    title(P5{p,2},'FontSize',FS);
    grid on; box off;
end
sgtitle('Single adapter, two orthogonal target spaces (dashed = single-target / non-mixed FT ref)','FontSize',FS-1);
save_fig(f4,'fig5_shared',outdir);

disp('ALL FIGS DONE'); disp(outdir);

%% Fig 4 —— 跨模型 2x2 的两把尺子: (a) RSA  (b) 同词集识别率(+三效应拆解)
tgts = {'OLMo target','pythia target'};
sc = {'self','cross'; 'cross','self'};   % 行=target, 列=reader [OLMo reads, pythia reads]
f5 = figure('Color','w','Position',[100 100 1100 400]);
% (a) RSA 2x2
subplot(1,2,1); hold on;
Y5 = [0.502 0.419; 0.225 0.226];  E5 = [0.011 0.025; 0.029 0.013];
b5 = bar(Y5,'grouped'); b5(1).FaceColor=[0.30 0.50 0.40]; b5(2).FaceColor=[0.55 0.65 0.80];
for k=1:2
    errorbar(b5(k).XEndPoints, Y5(:,k), E5(:,k),'k','LineStyle','none','LineWidth',1.0,'CapSize',6);
    for r=1:2, text(b5(k).XEndPoints(r), Y5(r,k)+E5(r,k)+0.02, sc{r,k},'HorizontalAlignment','center','FontSize',FS-3,'Color',[0.35 0.35 0.35]); end
end
set(gca,'XTick',1:2,'XTickLabel',tgts,'FontSize',FS-2);
ylabel('held-out RSA','FontSize',FS-2); ylim([0 0.60]);
legend({'OLMo learns','pythia learns'},'Location','northeast','FontSize',FS-3);
title('(a) RSA: set by the target, not the learner','FontSize',FS-1);
grid on; box off;
% (b) 同词集 top-5 识别率 + 三效应拆解
subplot(1,2,2); hold on;
Yid = [0.440 0.275; 0.333 0.213];   % rows=target, cols=[OLMo reads, pythia reads]
bb = bar(Yid,'grouped'); bb(1).FaceColor=[0.30 0.50 0.40]; bb(2).FaceColor=[0.55 0.65 0.80];
for k=1:2
    for r=1:2, text(bb(k).XEndPoints(r), Yid(r,k)+0.012, sc{r,k},'HorizontalAlignment','center','FontSize',FS-3,'Color',[0.35 0.35 0.35]); end
end
set(gca,'XTick',1:2,'XTickLabel',tgts,'FontSize',FS-2);
ylabel('matched top-5 identification','FontSize',FS-2); ylim([0 0.55]);
legend({'OLMo learns','pythia learns'},'Location','northeast','FontSize',FS-3);
title({'(b) identification, decomposed:','model +0.14 > target +0.085 >> self +0.02'},'FontSize',FS-1);
grid on; box off;
sgtitle('Cross-model 2x2 under two metrics: the "self" advantage is small (\approx +0.02)','FontSize',FS-1);
save_fig(f5,'fig4_crossmodel',outdir);

%% Fig 6 —— 通用能力变化 (微调后 vs base, 五项标准任务)
tasks6 = {'lambada','hellaswag','piqa','arc-easy','winogrande'};
% 行=适配器, 列=任务 (Δ 百分点)
D6 = [ -3.7  -1.5  -0.2  -1.7  -0.3;    % input-embed
       -0.6  -1.6  -0.9  -0.4  -2.6;    % unembedding
      -14.6  -9.3  -3.2  -5.5  -3.8;    % high-capacity (strongest readout)
       -3.6  -2.6  -1.7  -5.4  -2.2];   % pythia input-embed
adapters6 = {'input-embed (OLMo)','unembedding (OLMo)','high-capacity (strongest readout)','input-embed (pythia)'};
cmap6 = [0.20 0.45 0.70; 0.85 0.55 0.25; 0.75 0.25 0.25; 0.45 0.65 0.45];
f6 = figure('Color','w','Position',[100 100 780 410]); hold on;
b6 = bar(D6','grouped');   % D6' = 5任务 × 4适配器 → groups=任务
for k=1:4; b6(k).FaceColor=cmap6(k,:); end
yline(0,'k','LineWidth',0.8);
set(gca,'XTick',1:5,'XTickLabel',tasks6,'FontSize',FS-1); xtickangle(12);
ylabel('accuracy change vs. base (percentage points)','FontSize',FS-1); ylim([-16 2]);
legend(adapters6,'Location','southeast','FontSize',FS-2);
title('General-capability change after fine-tuning (5 standard tasks)','FontSize',FS);
grid on; box off; save_fig(f6,'fig6_capability',outdir);

%% Fig 7 —— Ogden 850: 训练词在"大小+方向"上都集中 -> 泛化更弱
S7 = load(fullfile(outdir,'ogden_scatter.mat'));    % 范数 KDE (analyze_ogden_geometry.py)
D7 = load(fullfile(outdir,'ogden_direction.mat'));  % 方向: 单位向量两两余弦 KDE
f7 = figure('Color','w','Position',[100 100 1320 360]);
% (a) 大小: 范数 KDE
subplot(1,3,1); hold on;
area(S7.xg, S7.dens_o,'FaceColor',[0.85 0.45 0.25],'FaceAlpha',0.45,'EdgeColor',[0.78 0.36 0.16],'LineWidth',1.7);
plot(S7.xg, S7.dens_r,'-','Color',[0.20 0.45 0.70],'LineWidth',2.0);
plot(S7.xg, S7.dens_t,'--','Color',[0.45 0.45 0.45],'LineWidth',1.6);
xlabel('input-embedding L2 norm','FontSize',FS-2); ylabel('word density','FontSize',FS-2);
legend({'Ogden-850','random','broad test'},'Location','northeast','FontSize',FS-3);
title('(a) magnitude: a small-norm clump','FontSize',FS-1);
xlim([4 13]); ylim([0 max(S7.dens_o)*1.15]); grid on; box off;
% (b) 方向: 两两余弦 KDE (越偏右=方向越一致/单调)
subplot(1,3,2); hold on;
area(D7.xg_cos, D7.pc_Ogden,'FaceColor',[0.85 0.45 0.25],'FaceAlpha',0.45,'EdgeColor',[0.78 0.36 0.16],'LineWidth',1.7);
plot(D7.xg_cos, D7.pc_random,'-','Color',[0.20 0.45 0.70],'LineWidth',2.0);
plot(D7.xg_cos, D7.pc_vocab,'--','Color',[0.45 0.45 0.45],'LineWidth',1.6);
xlabel('within-set pairwise cosine','FontSize',FS-2); ylabel('pair density','FontSize',FS-2);
legend({'Ogden-850','random','full vocab'},'Location','northeast','FontSize',FS-3);
title('(b) direction: words more aligned','FontSize',FS-1);
xlim([-0.18 0.42]); grid on; box off;
% (c) 后果: held-out RSA
subplot(1,3,3);
Y7 = [0.192 0.501; 0.181 0.218];   % rows=OLMo,pythia ; cols=Ogden-trained / random-trained
b7 = bar(Y7,'grouped'); b7(1).FaceColor=[0.85 0.45 0.25]; b7(2).FaceColor=[0.20 0.45 0.70];
set(gca,'XTick',1:2,'XTickLabel',{'OLMo','pythia'},'FontSize',FS-2);
ylabel('held-out RSA','FontSize',FS-2); ylim([0 0.58]);
legend({'trained on Ogden-850','trained on random'},'Location','northeast','FontSize',FS-3);
title('(c) -> weaker generalization','FontSize',FS-1);
grid on; box off;
sgtitle('Ogden-850 is concentrated in both magnitude (a) and direction (b), so the readout generalizes worse (c)','FontSize',FS-1);
save_fig(f7,'fig7_ogden',outdir);

%% ---- local function (R2023b 支持脚本内局部函数) ----
function save_fig(f,name,outdir)
    exportgraphics(f, fullfile(outdir,[name '.png']), 'Resolution',300);
    exportgraphics(f, fullfile(outdir,[name '.pdf']), 'ContentType','vector');
end
