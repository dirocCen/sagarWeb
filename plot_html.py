#!/usr/bin/env python
# coding: utf-8

# In[1]:


from vasprun import VaspRun, read_kpoints
import numpy as np
import re
import plotly.graph_objects as go
from plotly.offline import plot as pplot
# import plotly,json


def get_band_html(vasprun_file,kpts_file):
    vasp = VaspRun(vasprun_file)
    rec_lat = vasp.recip_lat()
    eigval_origin = vasp.read_eigenvals()[0]
    origin_kpt = vasp.read_kpoints()
    kpts = np.dot(origin_kpt,rec_lat)
    kpt_path = np.zeros((np.shape(kpts)[0],1))
    kpt_path[1:] = np.linalg.norm(kpts[1:]-kpts[:-1],axis=1).reshape((-1,1))
    kpt_path[1:] = np.cumsum(kpt_path[1:]).reshape((-1,1))
    eigval_shape = np.shape(eigval_origin)
    eigval = np.zeros((eigval_shape[0],eigval_shape[1]*2))
    for i in range(eigval_shape[1]):
        eigval[:,2*i:2*i+2] = eigval_origin[:,i,:]
    fermi,_,_ = vasp.read_dos()

    labels,high_kpts = read_kpoints(kpts_file)
    high_kpts = high_kpts @ rec_lat
    high_kpts_path = np.zeros((np.shape(high_kpts)[0],1))
    high_kpts_path[1:] = np.linalg.norm(high_kpts[1:]-high_kpts[:-1],axis=1).reshape((-1,1))
    high_kpts_path[1:] = np.cumsum(high_kpts_path[1:]).reshape((-1,1))


    fig = go.Figure()
    for ii in range(eigval_shape[1]):
        # import pdb; pdb.set_trace()
        fig.add_trace(go.Scatter(x=kpt_path.reshape((len(kpt_path),)), y=eigval[:,2*ii]-fermi,mode='lines',
                                 line=dict(color='blue', width=2)))

    annotations = []
    for i, label in enumerate(labels):
        annotations.append(
            go.Annotation(
                x=high_kpts_path[i][0], y=-5,
                xref="x1", yref="y1",
                text=label,
                xanchor="center", yanchor="top",
                showarrow=False
            )
        )
        fig.add_trace(go.Scatter(x=[high_kpts_path[i][0],high_kpts_path[i][0]], y=[-5,5],mode='lines',
                                 line=dict(color='black', width=1)))


    # In[4]:


    bandxaxis = go.XAxis(
        title="k-points",
        range=[0, kpt_path[-1]],
        showgrid=True,
        showline=True,
        ticks="",
        showticklabels=False,
        mirror=True,
        linewidth=2
    )
    bandyaxis = go.YAxis(
        title="$E - E_f \quad / \quad \\text{eV}$",
        range=[-5, 5],
        showgrid=True,
        showline=True,
        zeroline=True,
        mirror="ticks",
        ticks="inside",
        linewidth=2,
        tickwidth=2,
        zerolinewidth=2
    )

    bandlayout = go.Layout(
        title="Bands diagram",
        xaxis=bandxaxis,
        yaxis=bandyaxis,
        annotations=go.Annotations(annotations)
    )
    fig.update_layout(bandlayout)
    fig.update(layout_showlegend=False)
    # fig.show()

    # graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    # In[5]:


    # Get HTML representation of plotly.js and this figure
    plot_div = pplot(fig, output_type='div', include_plotlyjs=False)
    # Get id of html div element that looks like
    # <div id="301d22ab-bfba-4621-8f5d-dc4fd855bb33" ... >
    res = re.search('<div id="([^"]*)"', plot_div)
    div_id = res.groups()[0]

    # Build JavaScript callback for handling clicks
    # and opening the URL in the trace's customdata
    js_callback = """
    <script>
    var plot_element = document.getElementById("{div_id}");
    plot_element.on('plotly_click', function(data){{
        console.log(data);
        var point = data.points[0];
        if (point) {{
            console.log(point.customdata);
            window.open(point.customdata);
        }}
    }})
    </script>
    """.format(div_id=div_id)

    # Build HTML string
    html_str = """
    <html>
    <body>
    {plot_div}
    {js_callback}
    <script type="text/javascript" async
          src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.1/MathJax.js?config=TeX-MML-AM_SVG">
        </script>
    </body>
    </html>
    """.format(plot_div=plot_div, js_callback=js_callback)
    return html_str,plot_div

if __name__ == '__main__':
    html_str = get_band_html("/home/hecc/Desktop/vasprun.xml","/home/hecc/Desktop/band/band0/KPOINTS")
    with open("band.html","w") as f:
        f.write(html_str)
