import dash
from dash import dcc, html, Input, Output, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ── DATA LOAD ─────────────────────────────────────────────────────────────────
df       = pd.read_csv('parts_master.csv')
demand   = pd.read_csv('demand_timeseries.csv', parse_dates=['Month'])
matrix   = pd.read_csv('abcxyz_matrix.csv')
kpi      = pd.read_csv('kpis.csv').iloc[0]

# ── DESIGN TOKENS ─────────────────────────────────────────────────────────────
NAVY     = '#0D1B2A'
STEEL    = '#1B3A5C'
ACCENT   = '#2196F3'
TEAL     = '#00BCD4'
AMBER    = '#FFC107'
DANGER   = '#EF5350'
SUCCESS  = '#66BB6A'
MUTED    = '#B0BEC5'
BG       = '#0A1628'
CARD_BG  = '#112240'
BORDER   = '#1E3A5F'
WHITE    = '#E8EDF2'
FONT     = "'Inter', 'Segoe UI', sans-serif"

STATUS_COLORS = {
    'Healthy':      SUCCESS,
    'Slow Mover':   AMBER,
    'Excess Stock': TEAL,
    'Dead Stock':   DANGER,
}
ABC_COLORS = {'A': ACCENT, 'B': TEAL, 'C': MUTED}
METHOD_COLORS = {'SBA': ACCENT, 'Croston': TEAL, 'Naive': AMBER}

# ── HELPERS ───────────────────────────────────────────────────────────────────
def card(children, style=None):
    base = {
        'background': CARD_BG,
        'border': f'1px solid {BORDER}',
        'borderRadius': '8px',
        'padding': '20px',
        'height': '100%',
    }
    if style:
        base.update(style)
    return html.Div(children, style=base)

def kpi_card(label, value, sub=None, color=ACCENT):
    return card([
        html.P(label, style={'color': MUTED, 'fontSize': '11px',
                             'fontWeight': '600', 'letterSpacing': '1.2px',
                             'textTransform': 'uppercase', 'margin': '0 0 6px'}),
        html.H2(value, style={'color': color, 'fontSize': '28px',
                              'fontWeight': '700', 'margin': '0 0 4px',
                              'fontFamily': "'JetBrains Mono', monospace"}),
        html.P(sub or '', style={'color': MUTED, 'fontSize': '11px', 'margin': 0}),
    ], style={'padding': '16px 20px'})

def section_header(title, sub=''):
    return html.Div([
        html.H4(title, style={'color': WHITE, 'fontWeight': '700',
                               'margin': '0 0 2px', 'fontSize': '15px'}),
        html.P(sub, style={'color': MUTED, 'fontSize': '12px', 'margin': 0}),
    ], style={'marginBottom': '14px'})

# ── CHART BUILDERS ─────────────────────────────────────────────────────────────
def make_working_capital_bar(dff):
    g = dff.groupby('Category')['Working_Capital'].sum().reset_index()
    g = g.sort_values('Working_Capital', ascending=True)
    fig = go.Figure(go.Bar(
        x=g['Working_Capital'], y=g['Category'],
        orientation='h',
        marker=dict(
            color=g['Working_Capital'],
            colorscale=[[0, STEEL], [1, ACCENT]],
            showscale=False,
        ),
        text=[f'${v/1e3:.0f}K' for v in g['Working_Capital']],
        textposition='outside',
        textfont=dict(color=WHITE, size=11),
        hovertemplate='<b>%{y}</b><br>Working Capital: $%{x:,.0f}<extra></extra>',
    ))
    fig.update_layout(**chart_layout('Working Capital by Category ($)'))
    fig.update_xaxes(showticklabels=False, showgrid=False)
    fig.update_yaxes(tickfont=dict(size=11, color=MUTED))
    return fig

def make_abcxyz_heatmap(dff):
    abc_order = ['A', 'B', 'C']
    xyz_order = ['X', 'Y', 'Z']
    m = matrix.copy()
    pivot = m.pivot(index='ABC', columns='XYZ', values='Part_Count').reindex(
        index=abc_order, columns=xyz_order).fillna(0)
    strategies_pivot = m.pivot(index='ABC', columns='XYZ', values='Strategy').reindex(
        index=abc_order, columns=xyz_order).fillna('')

    hover = [[f'<b>{abc_order[r]}{xyz_order[c]}</b><br>Parts: {int(pivot.iloc[r,c])}<br>'
              f'Strategy: {strategies_pivot.iloc[r,c]}'
              for c in range(3)] for r in range(3)]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=xyz_order, y=abc_order,
        colorscale=[[0, STEEL], [0.5, ACCENT], [1, '#64B5F6']],
        text=pivot.values.astype(int),
        texttemplate='<b>%{text}</b>',
        textfont=dict(size=18, color=WHITE),
        hovertext=hover,
        hovertemplate='%{hovertext}<extra></extra>',
        showscale=False,
    ))
    fig.update_layout(**chart_layout('ABC-XYZ Classification Matrix'))
    fig.update_xaxes(title='Demand Variability (XYZ)', tickfont=dict(size=13, color=WHITE))
    fig.update_yaxes(title='Inventory Value (ABC)', tickfont=dict(size=13, color=WHITE),
                     autorange='reversed')
    return fig

def make_pareto(dff):
    g = dff.groupby('Category')['Annual_Value'].sum().reset_index()
    g = g.sort_values('Annual_Value', ascending=False)
    g['cum_pct'] = g['Annual_Value'].cumsum() / g['Annual_Value'].sum() * 100

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=g['Category'], y=g['Annual_Value'],
        name='Annual Value',
        marker_color=ACCENT,
        opacity=0.85,
        hovertemplate='<b>%{x}</b><br>Value: $%{y:,.0f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=g['Category'], y=g['cum_pct'],
        name='Cumulative %',
        yaxis='y2',
        line=dict(color=AMBER, width=2.5),
        mode='lines+markers',
        marker=dict(size=6, color=AMBER),
        hovertemplate='%{x}: %{y:.1f}%<extra></extra>',
    ))
    fig.add_hline(y=80, line_dash='dash', line_color=DANGER,
                  annotation_text='80%', annotation_font_color=DANGER,
                  annotation_position='right', yref='y2')

    layout = chart_layout('Pareto Analysis — Annual Inventory Value')
    layout['yaxis2'] = dict(
        title='Cumulative %', overlaying='y', side='right',
        range=[0, 110], tickformat='.0f',
        showgrid=False, tickfont=dict(color=AMBER),
        titlefont=dict(color=AMBER),
    )
    layout['legend'] = dict(x=0.02, y=0.98, font=dict(color=WHITE, size=11),
                            bgcolor='rgba(0,0,0,0)')
    fig.update_layout(**layout)
    fig.update_xaxes(tickangle=-30, tickfont=dict(size=10))
    return fig

def make_scatter(dff):
    fig = go.Figure()
    for status, color in STATUS_COLORS.items():
        sub = dff[dff['Stock_Status'] == status]
        fig.add_trace(go.Scatter(
            x=sub['Inventory_Turnover'],
            y=sub['Working_Capital'],
            mode='markers',
            name=status,
            marker=dict(color=color, size=5, opacity=0.65,
                        line=dict(width=0)),
            hovertemplate=(
                '<b>%{customdata[0]}</b><br>'
                'Category: %{customdata[1]}<br>'
                'Turnover: %{x:.2f}<br>'
                'Working Capital: $%{y:,.0f}<extra></extra>'
            ),
            customdata=sub[['Part_ID', 'Category']].values,
        ))
    fig.update_layout(**chart_layout('Dead Stock Analysis — Turnover vs Working Capital'))
    fig.update_xaxes(title='Inventory Turnover', range=[-0.2, 15])
    fig.update_yaxes(title='Working Capital ($)')
    return fig

def make_forecast_pie(dff):
    counts = dff['Recommended_Method'].value_counts().reset_index()
    counts.columns = ['Method', 'Count']
    colors = [METHOD_COLORS.get(m, MUTED) for m in counts['Method']]
    fig = go.Figure(go.Pie(
        labels=counts['Method'],
        values=counts['Count'],
        hole=0.55,
        marker=dict(colors=colors, line=dict(color=BG, width=2)),
        textinfo='label+percent',
        textfont=dict(size=12, color=WHITE),
        hovertemplate='<b>%{label}</b><br>Parts: %{value}<br>Share: %{percent}<extra></extra>',
    ))
    fig.add_annotation(
        text='<b>2,674</b><br><span style="font-size:10px">parts</span>',
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=14, color=WHITE),
    )
    fig.update_layout(**chart_layout('Recommended Forecast Method'))
    fig.update_layout(showlegend=False)
    return fig

def make_six_sigma_cpk(dff):
    fig = go.Figure()
    for cls, color in ABC_COLORS.items():
        sub = dff[dff['ABC_Class'] == cls]['Cpk'].clip(-1, 2.5)
        fig.add_trace(go.Violin(
            y=sub, name=f'Class {cls}',
            fillcolor=color, opacity=0.7,
            line_color=color,
            box_visible=True,
            meanline_visible=True,
            hoverinfo='y+name',
        ))
    fig.add_hline(y=1.33, line_dash='dash', line_color=AMBER,
                  annotation_text='Cpk = 1.33 (capable)',
                  annotation_font_color=AMBER,
                  annotation_position='right')
    fig.update_layout(**chart_layout('Process Capability (Cpk) by ABC Class'))
    fig.update_yaxes(title='Cpk', range=[-1.2, 2.8])
    fig.update_xaxes(title='ABC Class')
    return fig

def make_spc_chart(dff):
    sample = dff[dff['ABC_Class'] == 'A'].head(1)
    if sample.empty:
        sample = dff.head(1)
    pid = sample['Part_ID'].values[0]
    part_demand = demand[demand['Part_ID'] == pid].sort_values('Month')
    d = part_demand['Demand'].values
    mean_d = d.mean()
    std_d  = d.std() if d.std() > 0 else 1
    ucl = mean_d + 3 * std_d
    lcl = max(mean_d - 3 * std_d, 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=part_demand['Month'], y=d,
        mode='lines+markers',
        name='Demand', line=dict(color=ACCENT, width=1.5),
        marker=dict(size=4), hovertemplate='%{x|%b %Y}: %{y}<extra></extra>',
    ))
    for val, color, label in [(mean_d, SUCCESS, 'Mean'), (ucl, DANGER, 'UCL +3σ'), (lcl, AMBER, 'LCL -3σ')]:
        fig.add_hline(y=val, line_dash='dot', line_color=color,
                      annotation_text=f'{label}: {val:.1f}',
                      annotation_font_color=color,
                      annotation_position='right')
    fig.update_layout(**chart_layout(f'SPC Control Chart — Part {pid}'))
    fig.update_xaxes(title='Month')
    fig.update_yaxes(title='Units Demanded')
    return fig

def make_dmaic_table():
    dmaic_data = [
        ('Define',   'Problem',        'Dead stock worth $348K tied in parts with zero demand for 12+ months'),
        ('Define',   'Goal',           'Reduce dead-stock working capital by 30% through demand-driven policy'),
        ('Define',   'Scope',          '2,674 automotive spare parts, 51 months demand history'),
        ('Measure',  'Dead Stock',     '297 parts (11.1%) classified as Dead Stock — $348K trapped capital'),
        ('Measure',  'ABC Split',      '843A / 856B / 975C; 82.3% of parts are Z-class (high variability)'),
        ('Measure',  'Turnover',       'Average inventory turnover: 3.35× across the portfolio'),
        ('Measure',  'Forecast Error', 'MASE ranges 0.6–1.8 across methods; SBA wins for erratic/lumpy demand'),
        ('Analyze',  'Root Cause 1',   'Static reorder points ignore demand intermittency — triggers overstocking'),
        ('Analyze',  'Root Cause 2',   'No demand-class segmentation; all parts managed identically'),
        ('Analyze',  'Root Cause 3',   'EOQ violated for 89% of C-class parts — ordering cost assumptions wrong'),
        ('Improve',  'Policy Change',  'Apply ABC-XYZ segmented inventory policies (9 strategies mapped)'),
        ('Improve',  'Forecasting',    'Replace single-method forecasting with Croston/SBA for intermittent demand'),
        ('Improve',  'Newsvendor',     'Optimal order quantities identified — $653K annual savings potential'),
        ('Improve',  'Dead Stock',     'Flag parts with 12-month zero demand for review/liquidation'),
        ('Control',  'SPC',            'Control charts on A-class parts; trigger review when demand breaches UCL/LCL'),
        ('Control',  'KPIs',           'Monthly: turnover, dead-stock %, days-of-supply, MASE by class'),
        ('Control',  'Review Cadence', 'A-class: weekly review | B-class: monthly | C-class: quarterly'),
    ]
    phase_colors = {
        'Define': ACCENT, 'Measure': TEAL,
        'Analyze': AMBER, 'Improve': SUCCESS, 'Control': '#AB47BC',
    }
    rows = []
    for phase, step, finding in dmaic_data:
        rows.append(html.Tr([
            html.Td(phase, style={
                'color': phase_colors[phase], 'fontWeight': '700',
                'fontSize': '12px', 'padding': '8px 12px',
                'borderBottom': f'1px solid {BORDER}',
            }),
            html.Td(step, style={
                'color': MUTED, 'fontSize': '11px', 'fontWeight': '600',
                'padding': '8px 12px', 'borderBottom': f'1px solid {BORDER}',
            }),
            html.Td(finding, style={
                'color': WHITE, 'fontSize': '12px',
                'padding': '8px 12px', 'borderBottom': f'1px solid {BORDER}',
            }),
        ]))
    return html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={'color': MUTED, 'fontSize': '11px', 'fontWeight': '700',
                              'letterSpacing': '1px', 'textTransform': 'uppercase',
                              'padding': '10px 12px', 'borderBottom': f'1px solid {BORDER}'})
            for h in ['Phase', 'Step', 'Finding / Action']
        ])),
        html.Tbody(rows),
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

def chart_layout(title=''):
    return dict(
        title=dict(text=title, font=dict(color=WHITE, size=13, family=FONT),
                   x=0, xanchor='left', pad=dict(l=0)),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family=FONT, color=MUTED),
        margin=dict(l=10, r=10, t=36, b=10),
        xaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   tickfont=dict(size=11, color=MUTED)),
        yaxis=dict(gridcolor=BORDER, zerolinecolor=BORDER,
                   tickfont=dict(size=11, color=MUTED)),
        hoverlabel=dict(bgcolor=NAVY, font_size=12, font_family=FONT),
    )

# ── CATEGORIES FOR FILTER ─────────────────────────────────────────────────────
all_cats = sorted(df['Category'].unique())
cat_options = [{'label': 'All Categories', 'value': 'ALL'}] + \
              [{'label': c, 'value': c} for c in all_cats]

# ── LAYOUT ────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@700&display=swap',
    ],
    title='Inventory Intelligence | Akshada Karade',
)

SIDEBAR_STYLE = {
    'position': 'fixed', 'top': 0, 'left': 0, 'bottom': 0,
    'width': '210px', 'background': NAVY,
    'borderRight': f'1px solid {BORDER}',
    'padding': '0', 'zIndex': 100, 'overflowY': 'auto',
}
CONTENT_STYLE = {
    'marginLeft': '210px', 'background': BG,
    'minHeight': '100vh', 'padding': '24px',
}

nav_items = [
    ('overview',    '◈  Overview'),
    ('classification', '⊞  ABC-XYZ'),
    ('inventory',   '⊙  Inventory Policy'),
    ('forecasting', '〜  Forecasting'),
    ('sixsigma',    '⬡  Six Sigma'),
    ('dmaic',       '◎  DMAIC'),
]

sidebar = html.Div([
    html.Div([
        html.P('INVENTORY', style={'color': ACCENT, 'fontSize': '10px',
                                    'fontWeight': '700', 'letterSpacing': '2px',
                                    'margin': '0 0 2px'}),
        html.P('INTELLIGENCE', style={'color': WHITE, 'fontSize': '13px',
                                       'fontWeight': '700', 'margin': '0'}),
        html.P('Automotive Spare Parts', style={'color': MUTED, 'fontSize': '10px',
                                                 'margin': '4px 0 0'}),
    ], style={'padding': '20px 16px 16px', 'borderBottom': f'1px solid {BORDER}'}),

    html.Div([
        html.P('NAVIGATE', style={'color': MUTED, 'fontSize': '9px',
                                   'letterSpacing': '1.5px', 'fontWeight': '700',
                                   'margin': '16px 0 8px', 'padding': '0 16px'}),
        *[html.A(label, href=f'#{page}',
                 style={'display': 'block', 'color': MUTED, 'fontSize': '12px',
                        'padding': '8px 16px', 'textDecoration': 'none',
                        'fontWeight': '500', 'transition': 'color 0.2s'},
                 className='nav-link-custom')
          for page, label in nav_items],
    ]),

    html.Div([
        html.P('FILTER', style={'color': MUTED, 'fontSize': '9px',
                                 'letterSpacing': '1.5px', 'fontWeight': '700',
                                 'margin': '0 0 8px'}),
        dcc.Dropdown(
            id='cat-filter',
            options=cat_options,
            value='ALL',
            clearable=False,
            style={'fontSize': '11px'},
        ),
    ], style={'padding': '16px', 'borderTop': f'1px solid {BORDER}',
              'position': 'absolute', 'bottom': 0, 'left': 0, 'right': 0,
              'background': NAVY}),
], style=SIDEBAR_STYLE)

content = html.Div([

    # ── OVERVIEW ──────────────────────────────────────────────────────────────
    html.Div(id='overview', children=[
        html.Div([
            html.H2('Inventory Optimization & Working Capital Intelligence',
                    style={'color': WHITE, 'fontWeight': '700', 'fontSize': '20px',
                           'margin': '0 0 4px'}),
            html.P('Automotive Spare Parts · 2,674 SKUs · Hyndman Car Parts Dataset (51 months)',
                   style={'color': MUTED, 'fontSize': '12px', 'margin': '0 0 20px'}),
        ]),

        dbc.Row([
            dbc.Col(kpi_card('Total Parts', f'{kpi["Total_Parts"]:,.0f}',
                             '2,674 SKUs across 10 categories'), width=2),
            dbc.Col(kpi_card('Working Capital', f'${kpi["Total_Working_Capital"]/1e6:.2f}M',
                             'Total inventory investment', TEAL), width=2),
            dbc.Col(kpi_card('Avg Turnover', f'{kpi["Avg_Turnover"]:.2f}×',
                             'Annual inventory turnover', SUCCESS), width=2),
            dbc.Col(kpi_card('Dead Stock', f'${kpi["Dead_Stock_Value"]/1e3:.0f}K',
                             f'{kpi["Pct_Dead_Stock"]*100:.1f}% of parts', DANGER), width=2),
            dbc.Col(kpi_card('Newsvendor Savings', f'${kpi["Total_Newsvendor_Savings"]/1e3:.0f}K',
                             'Potential annual savings', AMBER), width=2),
            dbc.Col(kpi_card('Z-Class Parts', f'{kpi["Pct_Z_Class"]*100:.1f}%',
                             'High variability demand', '#AB47BC'), width=2),
        ], className='g-3 mb-3'),

        dbc.Row([
            dbc.Col(card([
                section_header('Working Capital by Category',
                               'Which categories hold the most tied-up cash'),
                dcc.Graph(id='bar-capital', config={'displayModeBar': False},
                          style={'height': '300px'}),
            ]), width=7),
            dbc.Col(card([
                section_header('Stock Status Distribution',
                               'Health of the overall inventory'),
                dcc.Graph(id='pie-status', config={'displayModeBar': False},
                          style={'height': '300px'}),
            ]), width=5),
        ], className='g-3'),
    ], style={'marginBottom': '40px'}),

    # ── ABC-XYZ ───────────────────────────────────────────────────────────────
    html.Div(id='classification', children=[
        html.H3('ABC-XYZ Classification', style={'color': WHITE, 'fontSize': '16px',
                                                   'fontWeight': '700', 'marginBottom': '16px',
                                                   'paddingTop': '8px',
                                                   'borderTop': f'1px solid {BORDER}'}),
        dbc.Row([
            dbc.Col(card([
                section_header('9-Box Segmentation Matrix',
                               'Part count per segment · hover for stocking strategy'),
                dcc.Graph(id='heatmap-matrix', config={'displayModeBar': False},
                          style={'height': '320px'}),
            ]), width=5),
            dbc.Col(card([
                section_header('Pareto Analysis — 80/20 Rule',
                               'Annual inventory value by category'),
                dcc.Graph(id='pareto', config={'displayModeBar': False},
                          style={'height': '320px'}),
            ]), width=7),
        ], className='g-3'),
    ], style={'marginBottom': '40px'}),

    # ── INVENTORY POLICY ──────────────────────────────────────────────────────
    html.Div(id='inventory', children=[
        html.H3('Inventory Policy', style={'color': WHITE, 'fontSize': '16px',
                                            'fontWeight': '700', 'marginBottom': '16px',
                                            'paddingTop': '8px',
                                            'borderTop': f'1px solid {BORDER}'}),
        dbc.Row([
            dbc.Col(card([
                section_header('Dead Stock Analysis',
                               'Turnover vs working capital — colored by stock status'),
                dcc.Graph(id='scatter-dead', config={'displayModeBar': False},
                          style={'height': '360px'}),
            ]), width=8),
            dbc.Col(card([
                section_header('EOQ Policy Summary', 'Key inventory metrics by class'),
                html.Div(id='policy-table', style={'marginTop': '8px'}),
            ]), width=4),
        ], className='g-3'),
    ], style={'marginBottom': '40px'}),

    # ── FORECASTING ───────────────────────────────────────────────────────────
    html.Div(id='forecasting', children=[
        html.H3('Forecasting & Demand', style={'color': WHITE, 'fontSize': '16px',
                                                'fontWeight': '700', 'marginBottom': '16px',
                                                'paddingTop': '8px',
                                                'borderTop': f'1px solid {BORDER}'}),
        dbc.Row([
            dbc.Col(card([
                section_header('Recommended Forecast Method',
                               'Per-part method selection based on demand class'),
                dcc.Graph(id='donut-forecast', config={'displayModeBar': False},
                          style={'height': '320px'}),
            ]), width=4),
            dbc.Col(card([
                section_header('MASE Distribution by Method',
                               'Lower MASE = better forecast accuracy'),
                dcc.Graph(id='mase-box', config={'displayModeBar': False},
                          style={'height': '320px'}),
            ]), width=8),
        ], className='g-3'),
    ], style={'marginBottom': '40px'}),

    # ── SIX SIGMA ─────────────────────────────────────────────────────────────
    html.Div(id='sixsigma', children=[
        html.H3('Six Sigma Analysis', style={'color': WHITE, 'fontSize': '16px',
                                              'fontWeight': '700', 'marginBottom': '16px',
                                              'paddingTop': '8px',
                                              'borderTop': f'1px solid {BORDER}'}),
        dbc.Row([
            dbc.Col(card([
                section_header('Process Capability (Cpk) by ABC Class',
                               'Cpk > 1.33 = capable process'),
                dcc.Graph(id='violin-cpk', config={'displayModeBar': False},
                          style={'height': '320px'}),
            ]), width=6),
            dbc.Col(card([
                section_header('SPC Control Chart',
                               'Demand monitoring — Class A part sample'),
                dcc.Graph(id='spc-chart', config={'displayModeBar': False},
                          style={'height': '320px'}),
            ]), width=6),
        ], className='g-3'),
    ], style={'marginBottom': '40px'}),

    # ── DMAIC ─────────────────────────────────────────────────────────────────
    html.Div(id='dmaic', children=[
        html.H3('DMAIC Framework', style={'color': WHITE, 'fontSize': '16px',
                                           'fontWeight': '700', 'marginBottom': '16px',
                                           'paddingTop': '8px',
                                           'borderTop': f'1px solid {BORDER}'}),
        card([
            section_header('Define · Measure · Analyze · Improve · Control',
                           'All findings anchored to actual data from this analysis'),
            html.Div(id='dmaic-table'),
        ]),
    ], style={'marginBottom': '40px'}),

    # Footer
    html.Div([
        html.P([
            'Built by ', html.Strong('Akshada Karade', style={'color': ACCENT}),
            ' · MS Engineering Management, UMass Amherst · ',
            'Hyndman Car Parts Dataset · Python + Plotly Dash',
        ], style={'color': MUTED, 'fontSize': '11px', 'textAlign': 'center',
                  'margin': '0', 'padding': '16px 0'}),
    ], style={'borderTop': f'1px solid {BORDER}', 'marginTop': '16px'}),

], style=CONTENT_STYLE)

app.layout = html.Div([sidebar, content], style={
    'fontFamily': FONT, 'background': BG,
})

# ── CALLBACKS ─────────────────────────────────────────────────────────────────
@app.callback(
    Output('bar-capital',   'figure'),
    Output('pie-status',    'figure'),
    Output('heatmap-matrix','figure'),
    Output('pareto',        'figure'),
    Output('scatter-dead',  'figure'),
    Output('donut-forecast','figure'),
    Output('violin-cpk',    'figure'),
    Output('spc-chart',     'figure'),
    Output('policy-table',  'children'),
    Output('dmaic-table',   'children'),
    Output('mase-box',      'figure'),
    Input('cat-filter', 'value'),
)
def update_all(cat):
    dff = df if cat == 'ALL' else df[df['Category'] == cat]

    # ── bar
    bar = make_working_capital_bar(dff)

    # ── pie status
    sc = dff['Stock_Status'].value_counts().reset_index()
    sc.columns = ['Status', 'Count']
    pie_s = go.Figure(go.Pie(
        labels=sc['Status'], values=sc['Count'],
        hole=0.5,
        marker=dict(colors=[STATUS_COLORS.get(s, MUTED) for s in sc['Status']],
                    line=dict(color=BG, width=2)),
        textinfo='label+percent',
        textfont=dict(size=11, color=WHITE),
        hovertemplate='<b>%{label}</b><br>Parts: %{value}<extra></extra>',
    ))
    pie_s.update_layout(**chart_layout('Stock Status Distribution'))
    pie_s.update_layout(showlegend=False)

    # ── heatmap (always full matrix)
    heatmap = make_abcxyz_heatmap(dff)

    # ── pareto
    pareto = make_pareto(dff)

    # ── scatter
    scatter = make_scatter(dff)

    # ── donut forecast
    donut = make_forecast_pie(dff)

    # ── violin cpk
    violin = make_six_sigma_cpk(dff)

    # ── spc
    spc = make_spc_chart(dff)

    # ── policy table
    policy_rows = []
    for cls in ['A', 'B', 'C']:
        sub = dff[dff['ABC_Class'] == cls]
        if len(sub) == 0:
            continue
        policy_rows.append(html.Tr([
            html.Td(f'Class {cls}', style={'color': ABC_COLORS[cls], 'fontWeight': '700',
                                            'padding': '8px 10px', 'fontSize': '12px',
                                            'borderBottom': f'1px solid {BORDER}'}),
            html.Td(f'{len(sub):,}', style={'color': WHITE, 'padding': '8px 10px',
                                             'fontSize': '12px',
                                             'borderBottom': f'1px solid {BORDER}'}),
            html.Td(f'${sub["Working_Capital"].sum()/1e3:.0f}K',
                    style={'color': WHITE, 'padding': '8px 10px', 'fontSize': '12px',
                           'borderBottom': f'1px solid {BORDER}'}),
            html.Td(f'{sub["Inventory_Turnover"].mean():.1f}×',
                    style={'color': WHITE, 'padding': '8px 10px', 'fontSize': '12px',
                           'borderBottom': f'1px solid {BORDER}'}),
        ]))
    policy_tbl = html.Table([
        html.Thead(html.Tr([
            html.Th(h, style={'color': MUTED, 'fontSize': '10px', 'fontWeight': '700',
                              'letterSpacing': '1px', 'textTransform': 'uppercase',
                              'padding': '8px 10px', 'borderBottom': f'1px solid {BORDER}'})
            for h in ['Class', 'Parts', 'W. Capital', 'Turnover']
        ])),
        html.Tbody(policy_rows),
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

    # ── MASE box
    mase_fig = go.Figure()
    for m, color in METHOD_COLORS.items():
        sub = dff[dff['Recommended_Method'] == m]['MASE']
        mase_fig.add_trace(go.Box(
            y=sub, name=m, marker_color=color,
            boxmean=True, jitter=0.3, pointpos=-1.8,
            marker=dict(size=3, opacity=0.4),
            hovertemplate=f'{m}: %{{y:.2f}}<extra></extra>',
        ))
    mase_fig.add_hline(y=1.0, line_dash='dash', line_color=AMBER,
                       annotation_text='MASE = 1.0 (benchmark)',
                       annotation_font_color=AMBER, annotation_position='right')
    mase_fig.update_layout(**chart_layout('MASE by Forecast Method (lower = better)'))
    mase_fig.update_yaxes(title='MASE')

    # ── DMAIC
    dmaic = make_dmaic_table()

    return (bar, pie_s, heatmap, pareto, scatter, donut,
            violin, spc, policy_tbl, dmaic, mase_fig)


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8050)

server = app.server
