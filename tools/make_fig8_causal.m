% make_fig8_causal.m  ——  图8: C7 因果扰动剂量反应 (base MATLAB, 无需 toolbox)
% 数据硬编码自 results/data/C7_causal_perturb/{olmo,pythia}_perturb.json 的 summary (权威值)。
% 风格与 make_paper_figs.m 一致; 输出 PNG(300dpi)+PDF(矢量) 到 paper_figs/(ASCII 路径)。
% 标签用英文(避免字体缺字), 中文图注写在 .md 里。

outdir = fullfile(fileparts(fileparts(mfilename('fullpath'))), 'paper_figs');
if ~exist(outdir,'dir'); mkdir(outdir); end
FS = 11;
close all; set(0,'DefaultFigureVisible','off');   % 离屏渲染

x  = [0 0.5 1 2];
Aa = {[0 0.616 1.239 1.160], [0 0.047 0.113 0.110]};   % (a) w in prompt: mean_along_by_scale
kA = {'k = 0.46  [0.40, 0.52]', 'k = 0.30  [0.21, 0.38]'};
N  = [57 64];
ttl  = {'(a) OLMo','(b) pythia'};
ymax = [1.45 0.145];
cA = [0.20 0.45 0.70];   % 蓝: w in prompt
cB = [0.85 0.45 0.20];   % 橙: synonym, w absent

f8 = figure('Color','w','Position',[100 100 900 380]);
for p = 1:2
    subplot(1,2,p); hold on;
    hA = plot(x, Aa{p}, '-o', 'Color',cA, 'LineWidth',2.0, 'MarkerFaceColor',cA, 'MarkerSize',6);
    hB = plot(x, [0 0 0 0], '-s', 'Color',cB, 'LineWidth',2.0, 'MarkerFaceColor',cB, 'MarkerSize',6);
    text(0.10, ymax(p)*0.93, kA{p}, 'Color',cA, 'FontSize',FS-1);
    text(1.95, ymax(p)*0.075, 'k = 0.000  [0, 0]  (exact zero)', 'Color',cB, 'FontSize',FS-1, 'HorizontalAlignment','right');
    xlabel('perturbation scale (\times\delta)','FontSize',FS-1);
    ylabel('spoken-vector displacement along \delta (PCA-32)','FontSize',FS-1);
    title(sprintf('%s   (n=%d, parse 100%%)',ttl{p},N(p)),'FontSize',FS);
    xlim([-0.08 2.1]); ylim([-0.07*ymax(p) ymax(p)]);
    if p==2
        legend([hA hB], {'(a) w in prompt','(b) synonym, w absent'}, 'Location','east','FontSize',FS-2);
    end
    grid on; box off;
end
sgtitle('Perturbing one embedding row: spoken values follow only when w is in the prompt','FontSize',FS);
exportgraphics(f8, fullfile(outdir,'fig8_causal.png'), 'Resolution',300);
exportgraphics(f8, fullfile(outdir,'fig8_causal.pdf'), 'ContentType','vector');
disp('FIG8 DONE'); disp(outdir);
