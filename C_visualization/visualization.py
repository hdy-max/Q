import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import matplotlib as mpl
mpl.font_manager._load_fontmanager(try_read_cache=False)
COLORS = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#9B59B6']


class WineVisualizer:
    """红酒全量图表生成"""

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.a_dir = os.path.join(os.path.dirname(self.base_dir),
                                  "A_feature_analysis", "output")
        self.b_dir = os.path.join(os.path.dirname(self.base_dir),
                                  "B_modeling", "output")
        self.output_dir = os.path.join(self.base_dir, "output", "charts")
        os.makedirs(self.output_dir, exist_ok=True)


        self.df = pd.read_csv(os.path.join(self.a_dir, "wine_clean.csv"),
                              encoding="utf-8-sig")
        self.df_num = pd.read_csv(os.path.join(self.a_dir, "wine_numeric.csv"),
                                  encoding="utf-8-sig")
        print(f"读取数据: {len(self.df)} 条")

    def _save(self, name):
        path = os.path.join(self.output_dir, name)
        plt.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"{name}")

    # ── 1. 单变量分布直方图 ──────────────────────────
    def plot_univariate_histograms(self):
        """11个特征 + 品质的分布直方图"""
        df = self.df_num.copy()
        features = ['fixed acidity', 'volatile acidity', 'citric acid',
                    'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

        fig, axes = plt.subplots(4, 3, figsize=(15, 14))
        axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            ax.hist(df[feat], bins=30, color='#3498DB', edgecolor='white', alpha=0.8)
            ax.set_title(feat, fontsize=11, fontweight='bold')
            ax.set_xlabel('')
            ax.set_ylabel('频数', fontsize=9)
            ax.grid(axis='y', alpha=0.3)

        # 品质分布
        ax = axes[11]
        ax.hist(df['quality'], bins=range(3, 10), color='#E74C3C',
                edgecolor='white', alpha=0.8, align='left')
        ax.set_title('quality (品质)', fontsize=11, fontweight='bold')
        ax.set_ylabel('频数', fontsize=9)
        ax.grid(axis='y', alpha=0.3)

        plt.suptitle('单变量分布直方图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save('01_univariate_histograms.png')

    # ── 2. 箱线图 ─────────────────────────────────────
    def plot_boxplots(self):
        """各特征的箱线图"""
        df = self.df_num.copy()
        features = ['fixed acidity', 'volatile acidity', 'citric acid',
                    'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol']

        fig, axes = plt.subplots(3, 4, figsize=(16, 10))
        axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            ax.boxplot(df[feat], patch_artist=True,
                       boxprops=dict(facecolor='#3498DB', alpha=0.6))
            ax.set_title(feat, fontsize=10, fontweight='bold')
            ax.grid(axis='y', alpha=0.2)

        axes[11].axis('off')
        plt.suptitle('各特征箱线图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save('02_boxplots.png')

    # ── 3. 酸度分析（对数尺度） ──────────────────────
    def plot_acidity_analysis(self):
        """酸度特征的对数尺度直方图 + 总酸度"""
        df = self.df_num.copy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))


        ax = axes[0]
        bins = 10 ** np.linspace(-2, 2, 50)
        ax.hist(df['fixed acidity'], bins=bins, edgecolor='k',
                label='Fixed Acidity', alpha=0.7, color='#E74C3C')
        ax.hist(df['volatile acidity'], bins=bins, edgecolor='k',
                label='Volatile Acidity', alpha=0.7, color='#3498DB')
        ax.hist(df['citric acid'], bins=bins, edgecolor='k',
                label='Citric Acid', alpha=0.7, color='#2ECC71')
        ax.set_xscale('log')
        ax.set_xlabel('Acid Concentration (g/dm³)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Histogram of Acid Concentration (log scale)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)


        ax = axes[1]
        if 'total acidity' in df.columns:
            ax.hist(df['total acidity'], bins=30, color='#9B59B6',
                    edgecolor='white', alpha=0.8)
            ax.set_xlabel('Total Acidity (g/dm³)', fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)
            ax.set_title('Total Acidity Distribution', fontsize=14, fontweight='bold')
            ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save('03_acidity_analysis.png')

    # ── 4. 甜度分析 ───────────────────────────────────
    def plot_sweetness(self):
        """甜度分类分布"""
        df = self.df.copy()
        if 'sweetness' not in df.columns:
            return

        fig, ax = plt.subplots(figsize=(10, 6))
        counts = df['sweetness'].value_counts()
        colors = ['#2ECC71', '#F39C12', '#E74C3C', '#9B59B6'][:len(counts)]
        bars = ax.bar(counts.index, counts.values, color=colors, edgecolor='white')

        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                    f'{val}', ha='center', fontsize=12, fontweight='bold')

        ax.set_xlabel('甜度分类', fontsize=13)
        ax.set_ylabel('数量', fontsize=13)
        ax.set_title('红酒甜度分类分布', fontsize=15, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        self._save('04_sweetness.png')

    # ── 5. 双变量：品质×特征 ──────────────────────────
    def plot_quality_vs_features(self):
        """品质与 top 特征的关系"""
        df = self.df_num.copy()


        corr = df.corr(numeric_only=True)['quality'].drop('quality').abs().sort_values(ascending=False)
        top3 = corr.head(3).index.tolist()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        for i, feat in enumerate(top3):
            ax = axes[i]
            for q in sorted(df['quality'].unique()):
                subset = df[df['quality'] == q]
                ax.hist(subset[feat], bins=15, alpha=0.6, label=f'品质{q}', edgecolor='white')
            ax.set_xlabel(feat, fontsize=12)
            ax.set_ylabel('频数', fontsize=12)
            ax.set_title(f'{feat} × 品质', fontsize=13, fontweight='bold')
            ax.legend(fontsize=8, loc='upper right')
            ax.grid(axis='y', alpha=0.2)

        plt.suptitle('品质与 Top 3 相关特征分布', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save('05_quality_vs_top3.png')

    # ── 6. 品质箱线图 ─────────────────────────────────
    def plot_quality_boxplots(self):
        """各特征按品质分组的箱线图"""
        df = self.df_num.copy()
        features = ['alcohol', 'volatile acidity', 'citric acid',
                    'sulphates', 'density', 'pH']

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()

        for i, feat in enumerate(features):
            ax = axes[i]
            df.boxplot(column=feat, by='quality', ax=ax, grid=False,
                       patch_artist=True, boxprops=dict(alpha=0.6))
            ax.set_title(feat, fontsize=12, fontweight='bold')
            ax.set_xlabel('品质评分')
            ax.set_ylabel('')

        plt.suptitle('各特征按品质分组的箱线图', fontsize=16, fontweight='bold')
        plt.tight_layout()
        self._save('06_quality_boxplots.png')

    # ── 7. 密度×酒精浓度 ──────────────────────────────
    def plot_density_alcohol(self):
        """密度与酒精浓度的关系"""
        df = self.df_num.copy()
        fig, ax = plt.subplots(figsize=(10, 7))

        scatter = ax.scatter(df['alcohol'], df['density'],
                            c=df['quality'], cmap='RdYlGn',
                            alpha=0.6, s=30, edgecolors='none')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('品质评分', fontsize=12)

        ax.set_xlabel('Alcohol (%vol)', fontsize=13)
        ax.set_ylabel('Density (g/cm³)', fontsize=13)
        ax.set_title('Density vs Alcohol Concentration', fontsize=15, fontweight='bold')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save('07_density_vs_alcohol.png')

    # ── 8. 酸性物质×pH ────────────────────────────────
    def plot_acidity_ph(self):
        """fixed acidity vs pH + citric acid vs pH"""
        df = self.df_num.copy()
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))


        ax = axes[0]
        sc1 = ax.scatter(df['fixed acidity'], df['pH'],
                        c=df['quality'], cmap='RdYlGn', alpha=0.6, s=25)
        plt.colorbar(sc1, ax=ax, label='品质')
        ax.set_xlabel('Fixed Acidity (g/dm³)', fontsize=12)
        ax.set_ylabel('pH', fontsize=12)
        ax.set_title('Fixed Acidity vs pH', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        ax = axes[1]
        sc2 = ax.scatter(df['citric acid'], df['pH'],
                        c=df['quality'], cmap='RdYlGn', alpha=0.6, s=25)
        plt.colorbar(sc2, ax=ax, label='品质')
        ax.set_xlabel('Citric Acid (g/dm³)', fontsize=12)
        ax.set_ylabel('pH', fontsize=12)
        ax.set_title('Citric Acid vs pH', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)

        plt.tight_layout()
        self._save('08_acidity_vs_ph.png')

    # ── 9. 多变量散点图 ───────────────────────────────
    def plot_multivariate(self):
        """酒精×挥发性酸×品质 三维关系"""
        df = self.df_num.copy()
        fig, ax = plt.subplots(figsize=(12, 8))

        for q in sorted(df['quality'].unique()):
            subset = df[df['quality'] == q]
            ax.scatter(subset['alcohol'], subset['volatile acidity'],
                      s=subset['citric acid'] * 100 + 10,
                      alpha=0.5, label=f'品质{q}', edgecolors='none')

        ax.set_xlabel('Alcohol (%vol)', fontsize=13)
        ax.set_ylabel('Volatile Acidity (g/dm³)', fontsize=13)
        ax.set_title('Alcohol × Volatile Acidity × Citric Acid (点大小)', fontsize=15, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        plt.tight_layout()
        self._save('09_multivariate_scatter.png')

    # ── 10. 相关性热力图 ──────────────────────────────
    def plot_correlation_heatmap(self):
        """特征相关性热力图"""
        df = self.df_num.copy()
        corr = df.corr(numeric_only=True)

        fig, ax = plt.subplots(figsize=(14, 11))
        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, vmin=-1, vmax=1, linewidths=0.5, ax=ax,
                    cbar_kws={'label': '相关系数'})
        ax.set_title('特征相关性热力图', fontsize=16, fontweight='bold', pad=20)
        plt.tight_layout()
        self._save('10_correlation_heatmap.png')

    # ── 11. 模型预测结果 ──────────────────────────────
    def plot_prediction_results(self):
        """预测 vs 实际值"""
        pred_path = os.path.join(self.b_dir, "lr_predictions.csv")
        if not os.path.exists(pred_path):
            return
        pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))


        ax = axes[0]
        ax.scatter(pred_df['actual'], pred_df['predicted'],
                  alpha=0.5, color='#3498DB', edgecolors='none')
        ax.plot([2, 9], [2, 9], 'r--', linewidth=2, label='完美预测')
        ax.set_xlabel('实际品质', fontsize=13)
        ax.set_ylabel('预测品质', fontsize=13)
        ax.set_title('线性回归：预测 vs 实际', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)


        ax = axes[1]
        errors = pred_df['predicted'] - pred_df['actual']
        ax.hist(errors, bins=20, color='#9B59B6', edgecolor='white', alpha=0.8)
        ax.axvline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('预测误差', fontsize=13)
        ax.set_ylabel('频数', fontsize=13)
        ax.set_title('预测误差分布', fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        self._save('11_prediction_results.png')

    # ── 12. 特征系数图 ────────────────────────────────
    def plot_coefficients(self):
        """线性回归特征系数"""
        coef_path = os.path.join(self.b_dir, "lr_coefficients.csv")
        if not os.path.exists(coef_path):
            return
        coef_df = pd.read_csv(coef_path, encoding="utf-8-sig")
        coef_df = coef_df.sort_values('coefficient')

        fig, ax = plt.subplots(figsize=(12, 7))
        colors = ['#E74C3C' if v >= 0 else '#3498DB' for v in coef_df['coefficient']]
        ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors, edgecolor='white')
        for i, (_, row) in enumerate(coef_df.iterrows()):
            ax.text(row['coefficient'] + 0.001 if row['coefficient'] >= 0
                    else row['coefficient'] - 0.01, i,
                    f'{row["coefficient"]:.4f}', va='center', fontsize=10)

        ax.set_xlabel('回归系数', fontsize=13)
        ax.set_ylabel('特征', fontsize=13)
        ax.set_title('线性回归特征系数（对品质的影响）', fontsize=15, fontweight='bold')
        ax.axvline(0, color='black', linestyle='-', linewidth=0.5)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        self._save('12_lr_coefficients.png')

    # ── 一键运行 ──────────────────────────────────────
    def run_all(self):
        print("\n" + "=" * 50)
        print("  C - 红酒可视化模块")
        print("=" * 50)

        methods = [
            ("单变量分布直方图", self.plot_univariate_histograms),
            ("箱线图", self.plot_boxplots),
            ("酸度分析", self.plot_acidity_analysis),
            ("甜度分析", self.plot_sweetness),
            ("品质×Top3特征", self.plot_quality_vs_features),
            ("品质箱线图", self.plot_quality_boxplots),
            ("密度×酒精浓度", self.plot_density_alcohol),
            ("酸性物质×pH", self.plot_acidity_ph),
            ("多变量散点图", self.plot_multivariate),
            ("相关性热力图", self.plot_correlation_heatmap),
            ("模型预测结果", self.plot_prediction_results),
            ("特征系数图", self.plot_coefficients),
        ]

        for name, method in methods:
            print(f"\n{name}...")
            try:
                method()
            except Exception as e:
                print(f"失败: {e}")

        print("\n" + "=" * 50)
        print("C 模块完成！")
        n = len([f for f in os.listdir(self.output_dir) if f.endswith('.png')])
        print(f"  共 {n} 张图表")
        print("=" * 50)


if __name__ == "__main__":
    wv = WineVisualizer()
    wv.run_all()
