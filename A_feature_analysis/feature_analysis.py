import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import os
import pandas as pd
import numpy as np

class WineFeatureAnalysis:
    """红酒特征分析与工程"""

    DATA_PATH = r"c:/Users/hdy/Downloads/winequality-red (1).csv"

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(self.base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        # 读取数据
        self.df = pd.read_csv(self.DATA_PATH, sep=';')
        print(f"读取数据: {len(self.df)} 条, {len(self.df.columns)} 列")
        print(f"   列名: {list(self.df.columns)}")

    # 1. 基本统计 
    def basic_statistics(self) -> dict:
        """输出基本统计信息"""
        df = self.df
        stats = {
            "描述统计": df.describe().round(3),
            "缺失值": df.isnull().sum().to_dict(),
            "品质分布": df['quality'].value_counts().sort_index().to_dict(),
            "品质占比": (df['quality'].value_counts(normalize=True).sort_index() * 100).round(1).to_dict(),
        }

        print("\n=== 描述统计 ===")
        print(stats["描述统计"])
        print(f"\n品质分布: {stats['品质分布']}")
        print(f"品质占比(%): {stats['品质占比']}")

        # 保存统计结果
        path = os.path.join(self.output_dir, "basic_statistics.csv")
        stats["描述统计"].to_csv(path, encoding="utf-8-sig")
        print(f"统计结果导出: {path}")
        return stats

    # 2. 特征工程 
    def feature_engineering(self) -> pd.DataFrame:
        """创建新特征：总酸度、对数变换、甜度分类"""
        df = self.df.copy()

        # 2a. 总酸度 = fixed acidity + volatile acidity + citric acid
        df['total acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']

        # 2b. 对酸度相关特征取对数（pH 已在对数尺度，不处理）
        acid_cols = ['fixed acidity', 'volatile acidity', 'citric acid',
                     'free sulfur dioxide', 'total sulfur dioxide', 'sulphates']
        for col in acid_cols:
            df[f'log_{col}'] = np.log10(df[col].clip(lower=0.001))

        # 2c. 甜度分类
        def sweetness_category(sugar):
            if sugar <= 4:
                return '干红'
            elif sugar <= 12:
                return '半干'
            elif sugar <= 45:
                return '半甜'
            else:
                return '甜'

        df['sweetness'] = df['residual sugar'].apply(sweetness_category)

        # 2d. 酸度分类
        def acidity_level(ph):
            if ph < 3.0:
                return '高酸'
            elif ph < 3.5:
                return '中酸'
            elif ph < 4.0:
                return '低酸'
            else:
                return '微酸'

        df['acidity_level'] = df['pH'].apply(acidity_level)

        # 2e. 酒精浓度分类
        df['alcohol_level'] = pd.cut(df['alcohol'],
                                      bins=[0, 9, 11, 13, 100],
                                      labels=['低', '中', '高', '极高'])

        # 2f. 品质分类
        df['quality_label'] = df['quality'].apply(
            lambda q: '差酒' if q <= 4 else ('好酒' if q >= 7 else '中等')
        )

        self.df = df
        print("特征工程完成")
        print(f"   总酸度: min={df['total acidity'].min():.2f}, max={df['total acidity'].max():.2f}")
        print(f"   甜度分布: {df['sweetness'].value_counts().to_dict()}")
        print(f"   品质标签分布: {df['quality_label'].value_counts().to_dict()}")
        return self.df

    # 3. 导出数据 
    def export_data(self):
        """导出清洗后数据"""
        path = os.path.join(self.output_dir, "wine_clean.csv")
        cols = ['fixed acidity', 'volatile acidity', 'citric acid',
                'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates',
                'alcohol', 'quality', 'total acidity',
                'sweetness', 'acidity_level', 'alcohol_level', 'quality_label']
        exist = [c for c in cols if c in self.df.columns]
        out = self.df[exist].copy()
        out.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"清洗数据导出: {path} ({len(out)} 条, {len(exist)} 列)")

        # 数值特征
        num_path = os.path.join(self.output_dir, "wine_numeric.csv")
        num_cols = ['fixed acidity', 'volatile acidity', 'citric acid',
                    'residual sugar', 'chlorides', 'free sulfur dioxide',
                    'total sulfur dioxide', 'density', 'pH', 'sulphates',
                    'alcohol', 'total acidity', 'quality']
        exist_num = [c for c in num_cols if c in self.df.columns]
        self.df[exist_num].to_csv(num_path, index=False, encoding="utf-8-sig")
        print(f"数值数据导出: {num_path}")

    # 运行 
    def run_all(self):
        print("\n" + "=" * 50)
        print("  A - 红酒特征分析模块")
        print("=" * 50)

        print("\n 1. 基本统计...")
        self.basic_statistics()

        print("\n 2. 特征工程...")
        self.feature_engineering()

        print("\n 3. 导出数据...")
        self.export_data()

        print("\n" + "=" * 50)
        print(" A 模块完成！")
        print("=" * 50)


if __name__ == "__main__":
    wa = WineFeatureAnalysis()
    wa.run_all()
