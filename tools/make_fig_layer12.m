% make_fig_layer12.m —— 拆分旧 fig3: 第一层残余独立成图 + 第二层三实验合并图
% 数值硬编码自 实验记录/G族_判决性实验.md + _analysis_report.md (权威值)。
% 风格同 make_paper_figs.m; 输出 PNG(300dpi)+PDF(矢量) 到 paper_figs/。
outdir = 'D:\ClaudeCode_workplace\llm_embedding_probe\paper_figs';
if ~exist(outdir,'dir'); mkdir(outdir); end
FS = 11;
close all; set(0,'DefaultFigureVisible','off');

%% Fig A —— 第一层: 切断激活后的干净子集残余 (原 fig3 panel a 独立)
res = [0.156 0.178 0.050 0.049]; rerr=[0.130 0.090 0.080 0.060];
labr= {'OLMo syn','OLMo def','pythia syn','pythia def'};
cr = [0.75 0.35 0.20; 0.85 0.55 0.25; 0.55 0.65 0.80; 0.70 0.78 0.88];
fA = figure('Color','w','Position',[100 100 560 380]); hold on;
for i=1:4; bar(i,res(i),0.6,'FaceColor',cr(i,:)); end
errorbar(1:4,res,rerr,'k','LineStyle','none','LineWidth',1.1,'CapSize',7);
yline(0.29,'--k','LineWidth',1.3);
text(1.35,0.315,'geometry baseline (OLMo 0.29)','FontSize',FS-2);
set(gca,'XTick',1:4,'XTickLabel',labr,'FontSize',FS-2); xtickangle(20);
ylabel('clean-subset residual RSA','FontSize',FS-1); ylim([-0.05 0.40]);
title('Residual after cutting the activation shortcut','FontSize',FS);
grid on; box off;
exportgraphics(fA, fullfile(outdir,'fig_g1_residual.png'), 'Resolution',300);
exportgraphics(fA, fullfile(outdir,'fig_g1_residual.pdf'), 'ContentType','vector');

%% Fig B —— 第二层: (a) 偏相关 (b) 双物理目标互证 (c) 无语义目标记忆不泛化
fB = figure('Color','w','Position',[100 100 1240 360]);
cO = [0.75 0.35 0.20]; cP = [0.55 0.65 0.80];
% (a) partial correlation, 3-seed
subplot(1,3,1); hold on;
pc = [0.605 0.200]; pcerr=[0.060 0.200];
bar(1,pc(1),0.5,'FaceColor',cO); bar(2,pc(2),0.5,'FaceColor',cP);
errorbar(1:2,pc,pcerr,'k','LineStyle','none','LineWidth',1.1,'CapSize',8);
set(gca,'XTick',1:2,'XTickLabel',{'OLMo','pythia'},'FontSize',FS-1);
ylabel('partial r (spoken, true norm | log freq)','FontSize',FS-2); ylim([-0.05 0.75]);
title('(a) more than frequency','FontSize',FS-1);
grid on; box off;
% (b) two physical readouts at the same level
subplot(1,3,2); hold on;
Yb = [0.776 0.797; 0.295 0.183];   % rows=OLMo,pythia; cols=L2 norm, recon err
bb = bar(Yb,'grouped'); bb(1).FaceColor=[0.85 0.55 0.25]; bb(2).FaceColor=[0.48 0.30 0.62];
set(gca,'XTick',1:2,'XTickLabel',{'OLMo','pythia'},'FontSize',FS-1);
ylabel('held-out correlation','FontSize',FS-2); ylim([0 0.95]);
legend({'L2 norm','PCA-trunc. recon. error'},'Location','northeast','FontSize',FS-3);
title('(b) reproduces across physical targets','FontSize',FS-1);
grid on; box off;
% (c) token-id bits: memorized on train, dead on held-out
subplot(1,3,3); hold on;
Yc = [0.558 -0.003; 0.372 0.004];  % rows=OLMo,pythia; cols=train, held-out
bc = bar(Yc,'grouped'); bc(1).FaceColor=[0.45 0.65 0.45]; bc(2).FaceColor=[0.60 0.60 0.60];
yline(0,'k','LineWidth',0.8);
set(gca,'XTick',1:2,'XTickLabel',{'OLMo','pythia'},'FontSize',FS-1);
ylabel('RSA','FontSize',FS-2); ylim([-0.1 0.68]);
legend({'training words','held-out words'},'Location','northeast','FontSize',FS-3);
title('(c) no semantics, no signal','FontSize',FS-1);
grid on; box off;
sgtitle('Layer two: the signal survives frequency removal (a), reappears under a second physical readout (b), and dies without semantics (c)','FontSize',FS-1);
exportgraphics(fB, fullfile(outdir,'fig_g2_semantic.png'), 'Resolution',300);
exportgraphics(fB, fullfile(outdir,'fig_g2_semantic.pdf'), 'ContentType','vector');
disp('LAYER12 FIGS DONE');
