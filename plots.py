from matplotlib import pyplot as plt
from math import sqrt

import plotly as py
import plotly.graph_objs as go

py.offline.init_notebook_mode(connected=True)

DEFAULT_COLOR = u'#1f77b4'

def get_rotation_matrix_2d(angle):
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c)))
    return R

def find_center(data):
    means = np.array([np.mean(col) for col in data.T])
    return means

def shift_point_to_point(data, to_, from_=None):
    if from_ is None:
        from_ = find_center(data)
    new_center = from_ - to_
    shifted_data = np.array([col - shift for col, shift in zip(data.T, new_center)]).T
    return shifted_data

def rotate_matrix_2d(X, y, angle):
    R = get_rotation_matrix_2d(angle)
    data = np.c_[X, y]

    data_rotated = np.dot(data, R)

    X_rot, y_rot = data_rotated[:, 0], data_rotated[:, 1]

    return X_rot, y_rot

def rotate_2d(X, y, angle, offset=None):
    data = np.c_[X, y]

    shifted_data = shift_point_to_point(data, np.array([0, 0]), offset)
    X_shift, y_shift = shifted_data[:, 0], shifted_data[:, 1]

    X_rot, y_rot = rotate_matrix_2d(X_shift, y_shift, angle)
    rotated_data = np.c_[X_rot, y_rot]

    if offset is None:
        offset = find_center(data)

    shifted_back_data = shift_point_to_point(rotated_data, offset, np.array([0, 0]))
    X_back, y_back = shifted_back_data[:, 0], shifted_back_data[:, 1]

    return X_back, y_back

def plot(data, aprox=None, xlim=None, ylim=None, figsize=None, pointer=None, name='Cool graph', xticks=None, yticks=None, rotate_angle=None, rotation_center=None, plot_type:'scatter, plot, logplot'='scatter', axes=None, color='r'):
    """
    Plots dataset: X, y and the aproximation
    :axes:
    pass fig, ax = plt.subplots(figsize=(10, 10))
    """

    AVALIBLE_TYPES = {'scatter', 'plot', 'logplot'}

    try:
        assert plot_type in AVALIBLE_TYPES
    except AssertionError:
        raise Exception('Choose one of:', AVALIBLE_TYPES)

    if axes is None:
        fig, axes = plt.subplots(figsize=figsize)


    if aprox is None:
        aprox = [None] * len(data)

    if type(data) not in (tuple, list):
        data = [data]

    for i, (split, f) in enumerate(zip(data, aprox)):
        X, y = split[:, 0], split[:, 1]

        prev_i = i - 1
        try:
            if prev_i >= 0:
                prev_data = data[prev_i]
                prev_X, prev_y = prev_data[-1, 0], prev_data[-1, 1]
                X, y = np.append(prev_X, X), np.append(prev_y, y)
        except IndexError:
            pass

        if plot_type == 'scatter':
            plotter = axes.scatter
        elif plot_type == 'plot':
            plotter = axes.plot
        elif plot_type == 'logplot':
            plotter = axes.semilogx

        if rotate_angle is not None:
            if rotation_center is None:
                rotation_center = find_center(np.c_[X, y])

            X_rot, y_rot = rotate_2d(X, y, rotate_angle, rotation_center)
            plotter(X_rot, y_rot, c=color)
        else:
            plotter(X, y, c=color)

        plt.title(name)

        if f is not None:
            X_aprox = np.linspace(np.min(X), np.max(X), 100)

            y_aprox = f(X_aprox)

            if rotate_angle is not None:
                X_aprox, y_aprox = rotate_2d(X_aprox, y_aprox, rotate_angle, rotation_center)

            axes.plot(X_aprox, y_aprox, c=DEFAULT_COLOR)

    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if pointer is not None:
        axes.axvline(pointer, c='r')
    if xticks is not None:
        plt.xticks(xticks)
    if yticks is not None:
        plt.yticks(yticks)

    if axes is None:
        plt.show()

def plot_3D(X, Y, Z, C=None, elev=None, azim=None, plot_type:'scatter or trisurf'='scatter', ax=None, figsize=None, cmap='Blues'):
    """
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
    """
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.tri import Triangulation

    AVALIBLE_TYPES = {'scatter', 'trisurf'}

    try:
        assert plot_type in AVALIBLE_TYPES
    except AssertionError:
        raise Exception('Choose one of:', AVALIBLE_TYPES)

    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = Axes3D(fig)

    if plot_type == 'scatter':
        ax.scatter(X, Y, Z, c=C)
    elif plot_type == 'trisurf':
        tri = Triangulation(np.ravel(X), np.ravel(Y))
        ax.plot_trisurf(X, Y, Z, triangles=tri.triangles, cmap=cmap)

    ax.view_init(elev=elev, azim=azim)

    plt.show()

def split_array(arr:'np.array', split_points:'[int]', split_ax=0):
    """Splits np.array on values in :split_ax: column using :split_points:"""
    splits = [arr]
    for point in split_points:
        l = splits[-1]
        idx = l[:, split_ax] <= point
        splits[-1] = l[idx]
        splits.append(l[~idx])
    return splits

def construct_layout(title='Cool graph', axis_names=None, figsize=None):
    if axis_names is not None:
        axis_names = dict(
            yaxis=dict(
                title=axis_names[1]
            ),
            xaxis=dict(
                title=axis_names[0]
            )
        )
    else:
        axis_names = {}

    if figsize is not None:
        fig_param = dict(
            width=figsize[0]*100,
            height=figsize[1]*100
        )
    else:
        fig_param = {}

    layout = go.Layout(
        title=title,
        **axis_names,
        **fig_param,
    )

    return layout

def handle_inputs(inputs, avalible_inputs):
    try:
        assert inputs in avalible_inputs
    except AssertionError:
        raise Exception('Choose one of:', avalible_inputs)

def seq_data(data):
    res = []
    for i, split in enumerate(data):
        X, y = split[:, 0], split[:, 1]
        try:
            prev_i = i - 1
            if prev_i >= 0:
                prev_data = data[prev_i]
                prev_X, prev_y = prev_data[-1, 0], prev_data[-1, 1]
                X, y = np.append(prev_X, X), np.append(prev_y, y)

            res.append(np.c_[X, y])
        except IndexError:
            pass
    return res

def plot_ly(data, aprox=None, title='Cool graph', trace_names=None, axis_names=None, rotate_angle=None, rotation_center=None, plot_type:'markers, lines, markers+lines'='markers', marker_obj=None, color='rgb(227,26,28)', return_traces=False, show_data=True):
    """
    Plots dataset: X, y and the aproximation
    :axes:
        fig, ax = plt.subplots(figsize=(10, 10))
    plotly:
        fig = go.Figure(data=data, layout=construct_layout())
        py.offline.iplot(fig)
    """

    handle_inputs(plot_type, {'markers', 'lines', 'markers+lines'})

    layout = construct_layout(title=title, axis_names=axis_names)

    if aprox is None:
        aprox = [None] * len(data)

    if trace_names is None:
        trace_names = ['data'] * len(data)

    if type(data) not in (tuple, list):
        data = [data]

    data = seq_data(data)

    traces = []
    for i, (split, f, name) in enumerate(zip(data, aprox, trace_names)):
        X, y = split[:, 0], split[:, 1]

        plot_data = dict(x=X, y=y)
        if rotate_angle is not None:
            if rotation_center is None:
                rotation_center = find_center(np.c_[X, y])

            X_rot, y_rot = rotate_2d(X, y, rotate_angle, rotation_center)
            plot_data = dict(x=X_rot, y=y_rot)

        if marker_obj is None:
            marker = dict(
                color=color,
            )

        trace = go.Scatter(
            name=name,
            mode=plot_type,
            x=plot_data['x'],
            y=plot_data['y'],
            marker=marker,
        )

        if f is not None:
            X_aprox = np.linspace(np.min(X), np.max(X), 100)
            y_aprox = f(X_aprox)

            if rotate_angle is not None:
                X_aprox, y_aprox = rotate_2d(X_aprox, y_aprox, rotate_angle, rotation_center)

            plot_data = dict(x=X_aprox, y=y_aprox)

            trace_line = go.Scatter(
                name=name+'_aproximation',
                mode='lines',
                x=plot_data['x'],
                y=plot_data['y'],
                line=dict(
                    color=DEFAULT_COLOR
                )
            )
            traces.append(trace_line)

        if show_data:
            traces.append(trace)

    if return_traces:
        return traces

    fig = go.Figure(data=traces, layout=layout)
    py.offline.iplot(fig)

def plot_ly_3D(X, Y, Z, C=None, figsize=(8, 8), axis_names=None, plot_type:'scatter or trisurf'='scatter', marker_size=5, return_traces=False, marker_obj=None, cmap='Viridis'):
    """
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10, 10))
        ax = Axes3D(fig)
        plotly:
            fig = go.Figure(data=data, layout=construct_layout())
            py.offline.iplot(fig)
    """
    from scipy.spatial import Delaunay
    import plotly.figure_factory as FF

    layout = construct_layout(axis_names, figsize=figsize)

    handle_inputs(plot_type, {'scatter', 'trisurf'})

    if marker_obj is None:
        marker = dict(
            color=C,
            colorscale=cmap,
            showscale=True,
            size=marker_size
        )
        if type(C) == str:
            marker['showscale'] = False
    else:
        marker = marker_obj

    if plot_type == 'scatter':
        trace = go.Scatter3d(
            mode='markers',
            x=X,
            y=Y,
            z=Z,
            marker=marker
        )
    elif plot_type == 'trisurf':

        u, v = np.meshgrid(X, Y)

        points2D = np.vstack([u, v]).T
        simplices = Delaunay(points2D).simplices

        fig = FF.create_trisurf(x=X, y=Y, z=Z, simplices=simplices, colormap=cmap)

        trace = go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale=cmap,
        )

    if return_traces:
        return trace

    if plot_type != 'trisurf':
        fig = go.Figure(data=[trace], layout=layout)

    py.offline.iplot(fig)