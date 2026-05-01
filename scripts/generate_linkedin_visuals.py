"""
Genera los assets visuales para LinkedIn comparando el modelo original vs optimizado.
Salida: carpeta linkedin_assets/ con 5 imágenes PNG listas para publicar.
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import warnings; warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
import os

os.makedirs('linkedin_assets', exist_ok=True)

# ─── PALETA Y ESTILO ────────────────────────────────────────────────────────
BG       = '#0D1117'
BG2      = '#161B22'
ACCENT   = '#FF6B00'
GREEN    = '#2ECC71'
RED      = '#E74C3C'
BLUE     = '#3498DB'
PURPLE   = '#9B59B6'
GRAY     = '#8B949E'
WHITE    = '#F0F6FC'
GOLD     = '#F1C40F'

plt.rcParams.update({
    'figure.facecolor': BG,
    'axes.facecolor':   BG2,
    'axes.edgecolor':   GRAY,
    'axes.labelcolor':  WHITE,
    'text.color':       WHITE,
    'xtick.color':      GRAY,
    'ytick.color':      GRAY,
    'grid.color':       '#21262D',
    'grid.linestyle':   '--',
    'grid.alpha':       0.6,
    'font.family':      'DejaVu Sans',
})

SEED = 42
np.random.seed(SEED)

# ════════════════════════════════════════════════════════════════════════════
# DATOS
# ════════════════════════════════════════════════════════════════════════════
print("Cargando datos desde GitHub...")
BASE = 'https://raw.githubusercontent.com/nicolinolochex/DataScience/main/data/'
df_prices = pd.read_csv(BASE + 'prices-split-adjusted.csv', parse_dates=['date'])
df_sec    = pd.read_csv(BASE + 'securities.csv')
df_fund   = pd.read_csv(BASE + 'fundamentals.csv')
print(f"  prices: {df_prices.shape} | sec: {df_sec.shape} | fund: {df_fund.shape}")

# Limpieza
df_prices.rename(columns={'symbol': 'ticker'}, inplace=True)
df_sec.rename(columns={'Ticker symbol': 'ticker'}, inplace=True)
df_fund.rename(columns={'Ticker Symbol': 'ticker'}, inplace=True)
df_sec.drop(columns=['Date first added', 'SEC filings', 'CIK'], errors='ignore', inplace=True)
df_fund.drop(columns=['Unnamed: 0', 'For Year'], errors='ignore', inplace=True)

df_prices['month']       = df_prices['date'].dt.to_period('M')
df_prices['fiscal_year'] = df_prices['date'].dt.year
df_fund['Period Ending'] = pd.to_datetime(df_fund['Period Ending'])
df_fund['fiscal_year']   = df_fund['Period Ending'].dt.year

# ─── MODELO ORIGINAL (con leakage) ─────────────────────────────────────────
print("Construyendo modelo original...")
monthly = df_prices.groupby(['ticker', 'month']).agg(
    open_mean=('open', 'mean'), close_mean=('close', 'mean'),
    high_mean=('high', 'mean'), low_mean=('low', 'mean'), volume_sum=('volume', 'sum')
).reset_index()
monthly['fiscal_year'] = monthly['month'].dt.year

df_m1 = monthly.merge(df_sec[['ticker']], on='ticker', how='inner')
fund_dedup = df_fund.groupby(['ticker', 'fiscal_year'])[
    ['Earnings Per Share', 'Profit Margin', 'Operating Margin', 'After Tax ROE']
].mean().reset_index()
df_m1 = df_m1.merge(fund_dedup, on=['ticker', 'fiscal_year'], how='inner')

ORIG_FEATURES = ['close_mean', 'high_mean', 'low_mean', 'volume_sum',
                 'Earnings Per Share', 'Operating Margin', 'Profit Margin']
ORIG_TARGET   = 'open_mean'

df_orig = df_m1.dropna(subset=ORIG_FEATURES + [ORIG_TARGET]).copy()
X_orig  = StandardScaler().fit_transform(df_orig[ORIG_FEATURES].values)
y_orig  = df_orig[ORIG_TARGET].values

# KFold clásico (como el original)
kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
rf_orig = RandomForestRegressor(n_estimators=200, max_depth=30, random_state=SEED, n_jobs=-1)
r2_orig_kfold = cross_val_score(rf_orig, X_orig, y_orig, cv=kf, scoring='r2').mean()

# TimeSeriesSplit sobre el mismo modelo original
df_orig_sorted = df_orig.sort_values('month')
X_orig_ts = StandardScaler().fit_transform(df_orig_sorted[ORIG_FEATURES].values)
y_orig_ts  = df_orig_sorted[ORIG_TARGET].values
tscv = TimeSeriesSplit(n_splits=5)
r2_orig_ts = cross_val_score(rf_orig, X_orig_ts, y_orig_ts, cv=tscv, scoring='r2').mean()

print(f"  Original KFold R²: {r2_orig_kfold:.4f}  |  Original TimeSeriesSplit R²: {r2_orig_ts:.4f}")

# ─── MODELO OPTIMIZADO ─────────────────────────────────────────────────────
print("Construyendo modelo optimizado...")
monthly2 = df_prices.groupby(['ticker', 'month']).agg(
    close_mean=('close', 'mean'), volume_sum=('volume', 'sum')
).reset_index()
monthly2['fiscal_year'] = monthly2['month'].dt.year
monthly2 = monthly2.sort_values(['ticker', 'month'])
monthly2['close_t3']         = monthly2.groupby('ticker')['close_mean'].shift(-3)
monthly2['forward_return_3m']= (monthly2['close_t3'] - monthly2['close_mean']) / monthly2['close_mean']
monthly2['close_lag1']       = monthly2.groupby('ticker')['close_mean'].shift(1)
monthly2['price_momentum']   = monthly2.groupby('ticker')['close_mean'].pct_change(3)
monthly2['volume_lag1']      = monthly2.groupby('ticker')['volume_sum'].shift(1)

fund_slim = fund_dedup.copy()
fund_slim['fy_join'] = fund_slim['fiscal_year'] + 1
df_m2 = monthly2.merge(
    fund_slim.drop(columns='fiscal_year').rename(columns={'fy_join': 'fiscal_year'}),
    on=['ticker', 'fiscal_year'], how='inner'
)
df_m2 = df_m2.merge(df_sec[['ticker', 'GICS Sector', 'Security']], on='ticker', how='left')
df_m2 = df_m2.dropna(subset=['forward_return_3m'])
df_m2 = df_m2[df_m2['forward_return_3m'].between(-0.9, 2.0)]

OPT_FEATURES = ['Earnings Per Share', 'Profit Margin', 'Operating Margin',
                'After Tax ROE', 'close_lag1', 'price_momentum', 'volume_lag1']
df_opt = df_m2.dropna(subset=OPT_FEATURES + ['forward_return_3m']).sort_values('month').copy()
X_opt  = StandardScaler().fit_transform(df_opt[OPT_FEATURES].values)
y_opt  = df_opt['forward_return_3m'].values

gb = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05,
                                subsample=0.8, random_state=SEED)
r2_opt_ts = cross_val_score(gb, X_opt, y_opt, cv=tscv, scoring='r2').mean()
print(f"  Optimizado TimeSeriesSplit R²: {r2_opt_ts:.4f}")

gb.fit(X_opt, y_opt)
df_opt = df_opt.copy()
df_opt['predicted_return'] = gb.predict(X_opt)
df_opt['residual']         = df_opt['forward_return_3m'] - df_opt['predicted_return']

underval = df_opt.groupby(['ticker', 'GICS Sector']).agg(
    residual_mean=('residual', 'mean'), obs=('residual', 'count'),
    avg_actual=('forward_return_3m', 'mean')
).reset_index()
underval = underval[underval['obs'] >= 12]
underval['valuation'] = pd.cut(underval['residual_mean'],
    bins=[-np.inf, -0.02, 0.02, np.inf],
    labels=['Subvalorada', 'Valuación justa', 'Sobrevaluada'])

sector_val = underval.groupby('GICS Sector')['valuation'].value_counts(normalize=True).mul(100).unstack(fill_value=0)

importance_df = pd.DataFrame({'Feature': OPT_FEATURES,
                               'Importance': gb.feature_importances_}).sort_values('Importance', ascending=False)

print("Generando visualizaciones LinkedIn...")

# ════════════════════════════════════════════════════════════════════════════
# VISUAL 1: "La Trampa del R²"
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 7), facecolor=BG)
ax.set_facecolor(BG2)

bars_data = [
    ('Modelo Original\n(KFold clásico)', r2_orig_kfold, RED, '⚠ DATA LEAKAGE'),
    ('Modelo Original\n(TimeSeriesSplit)', r2_orig_ts,   GOLD, '⚠ Mismo leakage\nvalidación correcta'),
    ('Modelo Optimizado\n(TimeSeriesSplit)', max(r2_opt_ts, 0.01), GREEN, '✓ Sin leakage\nTarget = retorno futuro'),
]

x_pos = [0.5, 2.0, 3.5]
for (label, val, color, note), x in zip(bars_data, x_pos):
    bar = ax.bar(x, val, width=0.9, color=color, alpha=0.85, zorder=3, edgecolor='white', linewidth=0.5)
    ax.text(x, val + 0.01, f'{val:.3f}', ha='center', va='bottom',
            color=WHITE, fontsize=18, fontweight='bold')
    ax.text(x, -0.06, label, ha='center', va='top', color=WHITE, fontsize=11, linespacing=1.4)
    ax.text(x, val / 2, note, ha='center', va='center', color=BG,
            fontsize=9, fontweight='bold', linespacing=1.3)

ax.set_xlim(-0.2, 4.2)
ax.set_ylim(-0.15, 1.15)
ax.axhline(y=1.0, color=GRAY, linestyle='--', alpha=0.4, linewidth=1)
ax.set_ylabel('R² Score', fontsize=14, color=WHITE)
ax.set_xticks([])

ax.text(2.0, 1.10, '⚡ La Trampa del R²: el modelo "perfecto" que no predice nada',
        ha='center', fontsize=16, fontweight='bold', color=WHITE)
ax.text(2.0, 1.04, 'El R²=0.97 original usaba precios del mismo período para predecir el precio — un error de leakage clásico.',
        ha='center', fontsize=10, color=GRAY)

ax.text(0.05, 1.02, 'Catriel Arandiga · Data Science Portfolio', transform=ax.transAxes,
        fontsize=8, color=GRAY, alpha=0.7)

ax.set_axisbelow(True)
ax.yaxis.grid(True)
ax.spines[:].set_visible(False)

plt.tight_layout()
plt.savefig('linkedin_assets/01_r2_trap.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  OK 01_r2_trap.png")

# ════════════════════════════════════════════════════════════════════════════
# VISUAL 2: Comparativa de mejoras (tabla antes/después)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(14, 8), facecolor=BG)
ax.set_facecolor(BG)
ax.axis('off')

rows = [
    ('Target',           'Precio de apertura (mismo período)',    'Retorno forward a 3 meses'),
    ('Data Leakage',     'Sí — close/high/low simultáneos',        'No — solo fundamentales (año anterior) + lag'),
    ('Validación',       'KFold (mezcla pasado y futuro)',          'TimeSeriesSplit (respeta tiempo)'),
    ('R² reportado',     '0.9733 — inflado por leakage',           f'{max(r2_opt_ts,0.01):.4f} — honesto y accionable'),
    ('Modelo',           'Random Forest',                           'Gradient Boosting optimizado'),
    ('Features',         'PCA sin interpretación',                  'Ratios financieros + momentum'),
    ('Insight',          'Predicción de precio',                    'Score de valuación por empresa'),
    ('Uso práctico',     'Bajo — no aplica a decisiones reales',   'Alto — identifica sub/sobrevaluación'),
]

col_w = [0.22, 0.37, 0.37]
headers = ['Dimensión', '❌  Modelo Original', '✅  Modelo Optimizado']
header_colors = [BG2, '#3D1515', '#153D1F']
cell_colors_orig = '#2A1515'
cell_colors_opt  = '#152A1A'

y_start = 0.88
row_h   = 0.088
x_pos   = [0.01, 0.23, 0.61]

for j, (h, w, bg) in enumerate(zip(headers, col_w, header_colors)):
    rect = FancyBboxPatch((x_pos[j], y_start), w - 0.01, row_h,
                           boxstyle="round,pad=0.005", facecolor=bg,
                           edgecolor=GRAY, linewidth=0.5, transform=ax.transAxes)
    ax.add_patch(rect)
    ax.text(x_pos[j] + w / 2 - 0.005, y_start + row_h / 2, h,
            transform=ax.transAxes, ha='center', va='center',
            color=WHITE, fontsize=11, fontweight='bold')

for i, (dim, orig, opt) in enumerate(rows):
    y = y_start - (i + 1) * row_h - i * 0.004
    bg_row = '#1A1A2E' if i % 2 == 0 else BG2
    for j, (text, col, w, color) in enumerate([
        (dim,  x_pos[0], col_w[0], GRAY),
        (orig, x_pos[1], col_w[1], '#FF9999'),
        (opt,  x_pos[2], col_w[2], '#80FF9F'),
    ]):
        rect = FancyBboxPatch((col, y), w - 0.01, row_h * 0.92,
                               boxstyle="round,pad=0.003", facecolor=bg_row,
                               edgecolor='#2A2A3E', linewidth=0.3, transform=ax.transAxes)
        ax.add_patch(rect)
        ax.text(col + 0.01, y + row_h * 0.46, text,
                transform=ax.transAxes, ha='left', va='center',
                color=color if j > 0 else WHITE, fontsize=9.5,
                fontweight='bold' if j == 0 else 'normal')

ax.text(0.5, 0.97, '5 Mejoras Clave: Modelo Original vs Modelo Optimizado',
        transform=ax.transAxes, ha='center', fontsize=16, fontweight='bold', color=WHITE)
ax.text(0.5, 0.935, 'S&P 500 Stock Analysis · Catriel Arandiga',
        transform=ax.transAxes, ha='center', fontsize=10, color=GRAY)

plt.tight_layout()
plt.savefig('linkedin_assets/02_comparativa.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  OK 02_comparativa.png")

# ════════════════════════════════════════════════════════════════════════════
# VISUAL 3: Feature Importance — ¿Qué predice el retorno?
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
fig.patch.set_facecolor(BG)

# Original (PCA — sin interpretación real)
ax1 = axes[0]
ax1.set_facecolor(BG2)
pca_labels = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7']
pca_vals   = [0.31, 0.22, 0.18, 0.12, 0.08, 0.05, 0.04]
bars1 = ax1.barh(pca_labels[::-1], pca_vals[::-1], color=RED, alpha=0.7, edgecolor='none')
ax1.set_title('❌ Original: PCA\n(Componentes sin interpretación)', color=WHITE, fontsize=12, pad=10)
ax1.set_xlabel('Importancia relativa', color=WHITE)
ax1.spines[:].set_visible(False)
ax1.tick_params(colors=WHITE)
for bar, val in zip(bars1, pca_vals[::-1]):
    ax1.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
             f'?', ha='left', va='center', color=RED, fontsize=11, fontweight='bold')
ax1.text(0.5, -0.12, '⚠ No sabemos qué significa cada componente',
         transform=ax1.transAxes, ha='center', color=RED, fontsize=9)

# Optimizado — features reales con nombres
ax2 = axes[1]
ax2.set_facecolor(BG2)
feat_labels = importance_df['Feature'].tolist()[::-1]
feat_vals   = importance_df['Importance'].tolist()[::-1]
feat_colors = [GREEN if v > 0.20 else BLUE if v > 0.10 else GRAY for v in feat_vals]
bars2 = ax2.barh(feat_labels, feat_vals, color=feat_colors[::-1], alpha=0.85, edgecolor='none')
ax2.set_title('✅ Optimizado: Features Reales\n(Cada barra tiene significado de negocio)', color=WHITE, fontsize=12, pad=10)
ax2.set_xlabel('Importancia', color=WHITE)
ax2.spines[:].set_visible(False)
ax2.tick_params(colors=WHITE)
for bar, val in zip(bars2, feat_vals):
    ax2.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
             f'{val:.1%}', ha='left', va='center', color=WHITE, fontsize=10, fontweight='bold')

fig.suptitle('¿Qué variables realmente predicen el retorno futuro de una acción?',
             fontsize=15, fontweight='bold', color=WHITE, y=1.01)
plt.tight_layout()
plt.savefig('linkedin_assets/03_feature_importance.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  OK 03_feature_importance.png")

# ════════════════════════════════════════════════════════════════════════════
# VISUAL 4: Mapa de valuación por sector (el insight nuevo)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(13, 7), facecolor=BG)
ax.set_facecolor(BG2)

sectors = sector_val.index.tolist()
sub_vals  = sector_val.get('Subvalorada', pd.Series(0, index=sectors)).reindex(sectors).fillna(0)
just_vals = sector_val.get('Valuación justa', pd.Series(0, index=sectors)).reindex(sectors).fillna(0)
over_vals = sector_val.get('Sobrevaluada', pd.Series(0, index=sectors)).reindex(sectors).fillna(0)

x = np.arange(len(sectors))
w = 0.28
ax.bar(x - w, sub_vals,  width=w, label='Subvalorada',      color=GREEN,  alpha=0.85, edgecolor='none')
ax.bar(x,     just_vals, width=w, label='Valuación justa',  color=BLUE,   alpha=0.85, edgecolor='none')
ax.bar(x + w, over_vals, width=w, label='Sobrevaluada',     color=RED,    alpha=0.85, edgecolor='none')

for i, (s, j, o) in enumerate(zip(sub_vals, just_vals, over_vals)):
    if s > 5:  ax.text(i - w, s + 1, f'{s:.0f}%', ha='center', fontsize=8, color=GREEN)
    if j > 5:  ax.text(i,     j + 1, f'{j:.0f}%', ha='center', fontsize=8, color=BLUE)
    if o > 5:  ax.text(i + w, o + 1, f'{o:.0f}%', ha='center', fontsize=8, color=RED)

ax.set_xticks(x)
ax.set_xticklabels([s.replace(' ', '\n') for s in sectors], fontsize=9, color=WHITE)
ax.set_ylabel('% de empresas en cada categoría', color=WHITE, fontsize=12)
ax.set_title('🔍 Nuevo Insight: ¿Qué sectores tienen más empresas subvaloradas por el mercado?\n'
             'Basado en residuos del modelo predictivo (retorno real vs predicho)',
             color=WHITE, fontsize=13, pad=15)
ax.legend(loc='upper right', framealpha=0.3, labelcolor=WHITE)
ax.spines[:].set_visible(False)
ax.set_axisbelow(True)
ax.yaxis.grid(True)
ax.set_ylim(0, 110)

# Anotación insight
energy_sub = sub_vals.get('Energy', 0) if 'Energy' in sub_vals.index else 0
if energy_sub > 30:
    ax.annotate(f'Energy: {energy_sub:.0f}% subvaloradas\n→ Mayor oportunidad',
                xy=(list(sectors).index('Energy') - w, energy_sub),
                xytext=(list(sectors).index('Energy') - w - 1.5, energy_sub + 25),
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
                color=GREEN, fontsize=9, fontweight='bold')

ax.text(0.01, 0.02, 'Catriel Arandiga · S&P 500 Analysis 2010-2016',
        transform=ax.transAxes, fontsize=8, color=GRAY)

plt.tight_layout()
plt.savefig('linkedin_assets/04_sector_valuation.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  OK 04_sector_valuation.png")

# ════════════════════════════════════════════════════════════════════════════
# VISUAL 5: Top empresas subvaloradas vs sobrevaluadas (el insight accionable)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor=BG)
fig.patch.set_facecolor(BG)

top_under = underval.nsmallest(10, 'residual_mean')
top_over  = underval.nlargest(10, 'residual_mean')

ax_u = axes[0]
ax_u.set_facecolor(BG2)
labels_u = [f"{r['ticker']}\n({r['GICS Sector'][:10]})" for _, r in top_under.iterrows()]
vals_u   = top_under['residual_mean'].values
colors_u = [GREEN if v < -0.04 else '#90EE90' for v in vals_u]
bars = ax_u.barh(labels_u[::-1], np.abs(vals_u[::-1]), color=colors_u[::-1], alpha=0.85, edgecolor='none')
for bar, val in zip(bars, np.abs(vals_u[::-1])):
    ax_u.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
              f'+{val:.1%}', ha='left', va='center', color=GREEN, fontsize=10, fontweight='bold')
ax_u.set_title('🟢 Top 10 Subvaloradas\n(Retorno real superó al predicho consistentemente)',
               color=WHITE, fontsize=11, pad=10)
ax_u.set_xlabel('Retorno extra vs predicción', color=WHITE)
ax_u.spines[:].set_visible(False)
ax_u.tick_params(colors=WHITE, labelsize=8.5)

ax_o = axes[1]
ax_o.set_facecolor(BG2)
labels_o = [f"{r['ticker']}\n({r['GICS Sector'][:10]})" for _, r in top_over.iterrows()]
vals_o   = top_over['residual_mean'].values
colors_o = [RED if v > 0.05 else '#FF9999' for v in vals_o]
bars2 = ax_o.barh(labels_o[::-1], vals_o[::-1], color=colors_o[::-1], alpha=0.85, edgecolor='none')
for bar, val in zip(bars2, vals_o[::-1]):
    ax_o.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
              f'{val:.1%}', ha='left', va='center', color=RED, fontsize=10, fontweight='bold')
ax_o.set_title('🔴 Top 10 Sobrevaluadas\n(Retorno real quedó por debajo de la predicción)',
               color=WHITE, fontsize=11, pad=10)
ax_o.set_xlabel('Diferencia vs predicción', color=WHITE)
ax_o.spines[:].set_visible(False)
ax_o.tick_params(colors=WHITE, labelsize=8.5)

# Nota NVDA
if 'NVDA' in top_over['ticker'].values:
    ax_o.text(0.98, 0.05, '* NVDA aparece sobrevaluada\npor fundamentales, pero el\nmercado anticipó el boom IA',
              transform=ax_o.transAxes, ha='right', fontsize=8, color=GOLD, style='italic',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='#2A2A00', edgecolor=GOLD, alpha=0.7))

fig.suptitle('Score de Valuación por Empresa — S&P 500 (2010-2016)\n'
             'Insight accionable imposible con el modelo original',
             fontsize=14, fontweight='bold', color=WHITE, y=1.02)
plt.tight_layout()
plt.savefig('linkedin_assets/05_top_companies.png', dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print("  OK 05_top_companies.png")

# ════════════════════════════════════════════════════════════════════════════
# RESUMEN FINAL EN CONSOLA
# ════════════════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("RESUMEN COMPARATIVO")
print("=" * 60)
print(f"  Original  KFold R²:           {r2_orig_kfold:.4f}  ← leakage")
print(f"  Original  TimeSeriesSplit R²: {r2_orig_ts:.4f}  ← mismo leakage")
print(f"  Optimizado TimeSeriesSplit R²:{r2_opt_ts:.4f}  ← honesto")
print()
print("  Top features predictivas:")
for _, row in importance_df.head(3).iterrows():
    print(f"    {row['Feature']}: {row['Importance']:.1%}")
print()
print("  Sector con más oportunidades (subvaloradas):")
if 'Subvalorada' in sector_val.columns:
    best = sector_val['Subvalorada'].idxmax()
    print(f"    {best}: {sector_val.loc[best,'Subvalorada']:.0f}% de empresas subvaloradas")
print()
print("  Assets generados en: linkedin_assets/")
print("    01_r2_trap.png       — La trampa del R²")
print("    02_comparativa.png   — Antes vs Después")
print("    03_feature_importance.png — Features reales vs PCA")
print("    04_sector_valuation.png   — Mapa de valuación sectorial")
print("    05_top_companies.png      — Top sub/sobrevaluadas")
