import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class WineModeling:

    def __init__(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.a_dir = os.path.join(self.base_dir,
                                  "A_feature_analysis", "output")
        self.output_dir = os.path.join(self.base_dir, "output")
        os.makedirs(self.output_dir, exist_ok=True)

        num_path = os.path.join(self.a_dir, "wine_numeric.csv")
        self.df = pd.read_csv(num_path, encoding="utf-8-sig")
        self.clean_path = os.path.join(self.a_dir, "wine_clean.csv")
        self.df_clean = pd.read_csv(self.clean_path, encoding="utf-8-sig")
        print(f"读取数值数据: {len(self.df)} 条, {len(self.df.columns)} 列")

    def correlation_analysis(self) -> pd.DataFrame:
        df = self.df.copy()
        corr = df.corr(numeric_only=True).round(4)
        path = os.path.join(self.output_dir, "correlation_matrix.csv")
        corr.to_csv(path, encoding="utf-8-sig")

        quality_corr = corr['quality'].drop('quality').sort_values(ascending=False)
        corr_path = os.path.join(self.output_dir, "corr_with_quality.csv")
        quality_corr.to_frame().to_csv(corr_path, encoding="utf-8-sig")

        print("\n=== 特征与品质的相关性（从高到低）===")
        for feat, val in quality_corr.items():
            print(f"  {feat:25s}: {val:+.4f}")

        self.quality_corr = quality_corr
        return corr

    def bivariate_analysis(self) -> dict:
        df = self.df_clean.copy()
        results = {}

        feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid',
                        'residual sugar', 'chlorides', 'free sulfur dioxide',
                        'total sulfur dioxide', 'density', 'pH', 'sulphates',
                        'alcohol', 'total acidity']
        exist = [c for c in feature_cols if c in df.columns]
        grouped = df.groupby('quality')[exist].mean().round(4)
        path = os.path.join(self.output_dir, "bivariate_by_quality.csv")
        grouped.to_csv(path, encoding="utf-8-sig")
        results['by_quality'] = grouped

        if 'quality_label' in df.columns:
            label_group = df.groupby('quality_label')[exist].mean().round(4)
            lbl_path = os.path.join(self.output_dir, "bivariate_by_label.csv")
            label_group.to_csv(lbl_path, encoding="utf-8-sig")
            results['by_label'] = label_group

        print("\n=== 按品质标签分组均值 ===")
        if 'by_label' in results:
            print(results['by_label'])

        return results

    def multivariate_analysis(self) -> pd.DataFrame:
        df = self.df_clean.copy()
        if 'quality_label' not in df.columns:
            return pd.DataFrame()

        top3 = ['alcohol', 'volatile acidity', 'citric acid']
        exist = [c for c in top3 if c in df.columns]

        multi = df[['quality_label'] + exist].copy()
        for col in exist:
            multi[f'{col}_bin'] = pd.qcut(multi[col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'],
                                          duplicates='drop')

        path = os.path.join(self.output_dir, "multivariate_analysis.csv")
        multi.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"多变量数据导出: {path}")
        return multi


    def _clip_outliers(self, df: pd.DataFrame, cols: list) -> pd.DataFrame:
        """IQR 异常值截断：将超出 3×IQR 的值截断到边界（温和策略，仅处理极端值）"""
        df = df.copy()
        total = 0
        for col in cols:
            Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            IQR = Q3 - Q1
            lo, hi = Q1 - 3.0 * IQR, Q3 + 3.0 * IQR
            n_low = (df[col] < lo).sum()
            n_high = (df[col] > hi).sum()
            total += n_low + n_high
            df[col] = df[col].clip(lo, hi)
        print(f"  IQR(×3) 异常值截断: {total} 个数据点被截断")
        return df

    def linear_regression_model(self) -> dict:
        df = self.df.copy()
        results = {}

        feature_cols = ['fixed acidity', 'volatile acidity', 'citric acid',
                        'residual sugar', 'chlorides', 'free sulfur dioxide',
                        'total sulfur dioxide', 'density', 'pH', 'sulphates',
                        'alcohol']
        exist = [c for c in feature_cols if c in df.columns]

        # ── 1. 异常值处理 ──
        df = self._clip_outliers(df, exist)

        X = df[exist].values
        y = df['quality'].values

        # ── 2. 分层划分 (stratify) ──
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # ── 3. 特征标准化 ──
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ── 4. Ridge 回归 (L2 正则化，抗噪声) ──
        model = Ridge(alpha=0.1)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_pred_round = np.round(y_pred).clip(0, 10).astype(int)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        acc = (y_pred_round == y_test).mean()

        results['mse'] = mse
        results['mae'] = mae
        results['r2'] = r2
        results['accuracy'] = acc
        results['model'] = model
        results['feature_cols'] = exist
        results['y_test'] = y_test
        results['y_pred'] = y_pred

        print(f"\n=== Ridge 回归模型评估（异常值截断 + 标准化 + 分层划分）===")
        print(f"  MSE:  {mse:.4f}")
        print(f"  MAE:  {mae:.4f}")
        print(f"  R²:   {r2:.4f}")
        print(f"  准确率(±0): {acc:.2%}")

        # ── 5. 各品质准确率 ──
        print("\n=== 各品质准确率 ===")
        for q in sorted(np.unique(y_test)):
            mask = y_test == q
            n = mask.sum()
            acc_q = (y_pred_round[mask] == y_test[mask]).mean()
            bar = "█" * int(acc_q * 20)
            print(f"  品质 {q} (n={n:>3}): {acc_q:.1%} {bar}")

        # ── 6. 交叉验证 ──
        X_all_scaled = scaler.fit_transform(X)
        cv_scores = cross_val_score(model, X_all_scaled, y, cv=5, scoring='r2')
        print(f"\n  5折CV R²: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

        # ── 7. 特征系数（标准化后可比较） ──
        coef_df = pd.DataFrame({
            'feature': exist,
            'coefficient': model.coef_.round(4)
        }).sort_values('coefficient', ascending=False)
        coef_path = os.path.join(self.output_dir, "lr_coefficients.csv")
        coef_df.to_csv(coef_path, index=False, encoding="utf-8-sig")

        print("\n=== 特征系数（标准化后，影响从大到小）===")
        for _, row in coef_df.iterrows():
            direction = "↑" if row['coefficient'] >= 0 else "↓"
            print(f"  {direction} {row['feature']:25s}: {row['coefficient']:+.4f}")

        # ── 8. 导出预测结果 ──
        pred_df = pd.DataFrame({
            'actual': y_test,
            'predicted': y_pred.round(2),
            'predicted_round': y_pred_round,
        })
        pred_path = os.path.join(self.output_dir, "lr_predictions.csv")
        pred_df.to_csv(pred_path, index=False, encoding="utf-8-sig")

        results['coef_df'] = coef_df
        results['pred_df'] = pred_df
        self.lr_results = results
        return results

    def run_all(self):
        print("\n" + "=" * 50)
        print("  B - 红酒建模分析模块")
        print("=" * 50)

        print("\n 1. 相关性分析...")
        self.correlation_analysis()

        print("\n 2. 双变量分析...")
        self.bivariate_analysis()

        print("\n 3. 多变量分析...")
        self.multivariate_analysis()

        print("\n 4. 线性回归预测模型...")
        self.linear_regression_model()

        print("\n" + "=" * 50)
        print("   B 模块完成！")
        print("=" * 50)


if __name__ == "__main__":
    wm = WineModeling()
    wm.run_all()
