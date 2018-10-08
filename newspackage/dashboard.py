import dash_core_components as dcc
import dash_html_components as html
from newspackage import app
import plotly.plotly as py
import plotly.graph_objs as go
import numpy as np
import json
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pdb


app.layout = html.Div(style={'fontFamily': 'Sans-Serif'}, children=[
   html.H1('News Dashboard', style={
           'textAlign': 'center', 'margin': '48px 0', 'fontFamily': 'Sans-Serif'}),
   dcc.Tabs(id="tabs", children=[
       dcc.Tab(label='Home'),
       dcc.Tab(label='Topic 1'),
       dcc.Tab(label='Topic 2'),
       dcc.Tab(label='Topic 3'),
       dcc.Tab(label='Topic 4')]
,
       style={
   'width': '100%',
   'fontFamily': 'Sans-Serif',
   'margin-left': 'auto',
   'margin-right': 'auto',
},
       content_style={
       'borderLeft': '1px solid #d6d6d6',
       'borderRight': '1px solid #d6d6d6',
       'borderBottom': '1px solid #d6d6d6',
       'padding': '44px'
   },
       parent_style={
       'maxWidth': '1000px',
       'margin': '0 auto'
   }
   )
])
