"""Plotting functions for visualizing distributions."""
from __future__ import division
import numpy as np
from scipy import stats
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import warnings

from six import string_types

try:
    import statsmodels.nonparametric.api as smnp
    _has_statsmodels = True
except ImportError:
    _has_statsmodels = False

from .utils import set_hls_values, iqr, _kde_support
from .palettes import color_palette, blend_palette
from .axisgrid import JointGrid


__all__ = ["distplot", "kdeplot", "rugplot", "jointplot"]


def _freedman_diaconis_bins(a):
    """Calculate number of hist bins using Freedman-Diaconis rule."""
    # From http://stats.stackexchange.com/questions/798/
    a = np.asarray(a)
    h = 2 * iqr(a) / (len(a) ** (1 / 3))
    # fall back to sqrt(a) bins if iqr is 0
    if h == 0:
        return int(np.sqrt(a.size))
    else:
        return int(np.ceil((a.max() - a.min()) / h))


def distplot(a, bins=None, hist=True, kde=True, rug=False, fit=None,
             hist_kws=None, kde_kws=None, rug_kws=None, fit_kws=None,
             color=None, vertical=False, norm_hist=False, axlabel=None,
             label=None, ax=None):
    """Flexibly plot a univariate distribution of observations.

    This function combines the matplotlib ``hist`` function (with automatic
    calculation of a good default bin size) with the seaborn :func:`kdeplot`
    and :func:`rugplot` functions. It can also fit ``scipy.stats``
    distributions and plot the estimated PDF over the data.

    Parameters
    ----------

    a : Series, 1d-array, or list.
        Observed data. If this is a Series object with a ``name`` attribute,
        the name will be used to label the data axis.
    bins : argument for matplotlib hist(), or None, optional
        Specification of hist bins, or None to use Freedman-Diaconis rule.
    hist : bool, optional
        Whether to plot a (normed) histogram.
    kde : bool, optional
        Whether to plot a gaussian kernel density estimate.
    rug : bool, optional
        Whether to draw a rugplot on the support axis.
    fit : random variable object, optional
        An object with `fit` method, returning a tuple that can be passed to a
        `pdf` method a positional arguments following an grid of values to
        evaluate the pdf on.
    {hist, kde, rug, fit}_kws : dictionaries, optional
        Keyword arguments for underlying plotting functions.
    color : matplotlib color, optional
        Color to plot everything but the fitted curve in.
    vertical : bool, optional
        If True, oberved values are on y-axis.
    norm_hist : bool, optional
        If True, the histogram height shows a density rather than a count.
        This is implied if a KDE or fitted density is plotted.
    axlabel : string, False, or None, optional
        Name for the support axis label. If None, will try to get it
        from a.namel if False, do not set a label.
    label : string, optional
        Legend label for the relevent component of the plot
    ax : matplotlib axis, optional
        if provided, plot on this axis

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.

    See Also
    --------
    kdeplot : Show a univariate or bivariate distribution with a kernel
              density estimate.
    rugplot : Draw small vertical lines to show each observation in a
              distribution.

    Examples
    --------

    Show a default plot with a kernel density estimate and histogram with bin
    size determined automatically with a reference rule:

    .. plot::
        :context: close-figs

        >>> import seaborn as sns, numpy as np
        >>> sns.set(rc={"figure.figsize": (8, 4)}); np.random.seed(0)
        >>> x = np.random.randn(100)
        >>> ax = sns.distplot(x)

    Use Pandas objects to get an informative axis label:

    .. plot::
        :context: close-figs

        >>> import pandas as pd
        >>> x = pd.Series(x, name="x variable")
        >>> ax = sns.distplot(x)

    Plot the distribution with a kenel density estimate and rug plot:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x, rug=True, hist=False)

    Plot the distribution with a histogram and maximum likelihood gaussian
    distribution fit:

    .. plot::
        :context: close-figs

        >>> from scipy.stats import norm
        >>> ax = sns.distplot(x, fit=norm, kde=False)

    Plot the distribution on the vertical axis:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x, vertical=True)

    Change the color of all the plot elements:

    .. plot::
        :context: close-figs

        >>> sns.set_color_codes()
        >>> ax = sns.distplot(x, color="y")

    Pass specific parameters to the underlying plot functions:

    .. plot::
        :context: close-figs

        >>> ax = sns.distplot(x, rug=True, rug_kws={"color": "g"},
        ...                   kde_kws={"color": "k", "lw": 3, "label": "KDE"},
        ...                   hist_kws={"histtype": "step", "linewidth": 3,
        ...                             "alpha": 1, "color": "g"})

    """
    if ax is None:
        ax = plt.gca()

    # Intelligently label the support axis
    label_ax = bool(axlabel)
    if axlabel is None and hasattr(a, "name"):
        axlabel = a.name
        if axlabel is not None:
            label_ax = True

    # Make a a 1-d array
    a = np.asarray(a).squeeze()

    # Decide if the hist is normed
    norm_hist = norm_hist or kde or (fit is not None)

    # Handle dictionary defaults
    if hist_kws is None:
        hist_kws = dict()
    if kde_kws is None:
        kde_kws = dict()
    if rug_kws is None:
        rug_kws = dict()
    if fit_kws is None:
        fit_kws = dict()

    # Get the color from the current color cycle
    if color is None:
        if vertical:
            line, = ax.plot(0, a.mean())
        else:
            line, = ax.plot(a.mean(), 0)
        color = line.get_color()
        line.remove()

    # Plug the label into the right kwarg dictionary
    if label is not None:
        if hist:
            hist_kws["label"] = label
        elif kde:
            kde_kws["label"] = label
        elif rug:
            rug_kws["label"] = label
        elif fit:
            fit_kws["label"] = label

    if hist:
        if bins is None:
            bins = min(_freedman_diaconis_bins(a), 50)
        hist_kws.setdefault("alpha", 0.4)
        hist_kws.setdefault("normed", norm_hist)
        orientation = "horizontal" if vertical else "vertical"
        hist_color = hist_kws.pop("color", color)
        ax.hist(a, bins, orientation=orientation,
                color=hist_color, **hist_kws)
        if hist_color != color:
            hist_kws["color"] = hist_color

    if kde:
        kde_color = kde_kws.pop("color", color)
        kdeplot(a, vertical=vertical, ax=ax, color=kde_color, **kde_kws)
        if kde_color != color:
            kde_kws["color"] = kde_color

    if rug:
        rug_color = rug_kws.pop("color", color)
        axis = "y" if vertical else "x"
        rugplot(a, axis=axis, ax=ax, color=rug_color, **rug_kws)
        if rug_color != color:
            rug_kws["color"] = rug_color

    if fit is not None:
        fit_color = fit_kws.pop("color", "#282828")
        gridsize = fit_kws.pop("gridsize", 200)
        cut = fit_kws.pop("cut", 3)
        clip = fit_kws.pop("clip", (-np.inf, np.inf))
        bw = stats.gaussian_kde(a).scotts_factor() * a.std(ddof=1)
        x = _kde_support(a, bw, gridsize, cut, clip)
        params = fit.fit(a)
        pdf = lambda x: fit.pdf(x, *params)
        y = pdf(x)
        if vertical:
            x, y = y, x
        ax.plot(x, y, color=fit_color, **fit_kws)
        if fit_color != "#282828":
            fit_kws["color"] = fit_color

    if label_ax:
        if vertical:
            ax.set_ylabel(axlabel)
        else:
            ax.set_xlabel(axlabel)

    return ax


def _univariate_kdeplot(data, shade, vertical, kernel, bw, gridsize, cut,
                        clip, legend, ax, cumulative=False, **kwargs):
    """Plot a univariate kernel density estimate on one of the axes."""

    # Sort out the clipping
    if clip is None:
        clip = (-np.inf, np.inf)

    # Calculate the KDE
    if _has_statsmodels:
        # Prefer using statsmodels for kernel flexibility
        x, y = _statsmodels_univariate_kde(data, kernel, bw,
                                           gridsize, cut, clip,
                                           cumulative=cumulative)
    else:
        # Fall back to scipy if missing statsmodels
        if kernel != "gau":
            kernel = "gau"
            msg = "Kernel other than `gau` requires statsmodels."
            warnings.warn(msg, UserWarning)
        if cumulative:
            raise ImportError("Cumulative distributions are currently"
                              "only implemented in statsmodels."
                              "Please install statsmodels.")
        x, y = _scipy_univariate_kde(data, bw, gridsize, cut, clip)

    # Make sure the density is nonnegative
    y = np.amax(np.c_[np.zeros_like(y), y], axis=1)

    # Flip the data if the plot should be on the y axis
    if vertical:
        x, y = y, x

    # Check if a label was specified in the call
    label = kwargs.pop("label", None)

    # Otherwise check if the data object has a name
    if label is None and hasattr(data, "name"):
        label = data.name

    # Decide if we're going to add a legend
    legend = label is not None and legend
    label = "_nolegend_" if label is None else label

    # Use the active color cycle to find the plot color
    line, = ax.plot(x, y, **kwargs)
    color = line.get_color()
    line.remove()
    kwargs.pop("color", None)

    # Draw the KDE plot and, optionally, shade
    ax.plot(x, y, color=color, label=label, **kwargs)
    alpha = kwargs.get("alpha", 0.25)
    if shade:
        if vertical:
            ax.fill_betweenx(y, 1e-12, x, facecolor=color, alpha=alpha)
        else:
            ax.fill_between(x, 1e-12, y, facecolor=color, alpha=alpha)

    # Draw the legend here
    if legend:
        ax.legend(loc="best")

    return ax


def _statsmodels_univariate_kde(data, kernel, bw, gridsize, cut, clip,
                                cumulative=False):
    """Compute a univariate kernel density estimate using statsmodels."""
    fft = kernel == "gau"
    kde = smnp.KDEUnivariate(data)
    kde.fit(kernel, bw, fft, gridsize=gridsize, cut=cut, clip=clip)
    if cumulative:
        grid, y = kde.support, kde.cdf
    else:
        grid, y = kde.support, kde.density
    return grid, y


def _scipy_univariate_kde(data, bw, gridsize, cut, clip):
    """Compute a univariate kernel density estimate using scipy."""
    try:
        kde = stats.gaussian_kde(data, bw_method=bw)
    except TypeError:
        kde = stats.gaussian_kde(data)
        if bw != "scott":  # scipy default
            msg = ("Ignoring bandwidth choice, "
                   "please upgrade scipy to use a different bandwidth.")
            warnings.warn(msg, UserWarning)
    if isinstance(bw, string_types):
        bw = "scotts" if bw == "scott" else bw
        bw = getattr(kde, "%s_factor" % bw)() * np.std(data)
    grid = _kde_support(data, bw, gridsize, cut, clip)
    y = kde(grid)
    return grid, y


def _bivariate_kdeplot(x, y, filled, fill_lowest,
                       kernel, bw, gridsize, cut, clip,
                       axlabel, ax, **kwargs):
    """Plot a joint KDE estimate as a bivariate contour plot."""
    # Determine the clipping
    if clip is None:
        clip = [(-np.inf, np.inf), (-np.inf, np.inf)]
    elif np.ndim(clip) == 1:
        clip = [clip, clip]

    # Calculate the KDE
    if _has_statsmodels:
        xx, yy, z = _statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip)
    else:
        xx, yy, z = _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip)

    # Plot the contours
    n_levels = kwargs.pop("n_levels", 10)
    cmap = kwargs.get("cmap", "BuGn" if filled else "BuGn_d")
    if isinstance(cmap, string_types):
        if cmap.endswith("_d"):
            pal = ["#333333"]
            pal.extend(color_palette(cmap.replace("_d", "_r"), 2))
            cmap = blend_palette(pal, as_cmap=True)
        else:
            cmap = mpl.cm.get_cmap(cmap)

    kwargs["cmap"] = cmap
    contour_func = ax.contourf if filled else ax.contour
    cset = contour_func(xx, yy, z, n_levels, **kwargs)
    if filled and not fill_lowest:
        cset.collections[0].set_alpha(0)
    kwargs["n_levels"] = n_levels

    # Label the axes
    if hasattr(x, "name") and axlabel:
        ax.set_xlabel(x.name)
    if hasattr(y, "name") and axlabel:
        ax.set_ylabel(y.name)

    return ax


def _statsmodels_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using statsmodels."""
    if isinstance(bw, string_types):
        bw_func = getattr(smnp.bandwidths, "bw_" + bw)
        x_bw = bw_func(x)
        y_bw = bw_func(y)
        bw = [x_bw, y_bw]
    elif np.isscalar(bw):
        bw = [bw, bw]

    if isinstance(x, pd.Series):
        x = x.values
    if isinstance(y, pd.Series):
        y = y.values

    kde = smnp.KDEMultivariate([x, y], "cc", bw)
    x_support = _kde_support(x, kde.bw[0], gridsize, cut, clip[0])
    y_support = _kde_support(y, kde.bw[1], gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde.pdf([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def _scipy_bivariate_kde(x, y, bw, gridsize, cut, clip):
    """Compute a bivariate kde using scipy."""
    data = np.c_[x, y]
    kde = stats.gaussian_kde(data.T)
    data_std = data.std(axis=0, ddof=1)
    if isinstance(bw, string_types):
        bw = "scotts" if bw == "scott" else bw
        bw_x = getattr(kde, "%s_factor" % bw)() * data_std[0]
        bw_y = getattr(kde, "%s_factor" % bw)() * data_std[1]
    elif np.isscalar(bw):
        bw_x, bw_y = bw, bw
    else:
        msg = ("Cannot specify a different bandwidth for each dimension "
               "with the scipy backend. You should install statsmodels.")
        raise ValueError(msg)
    x_support = _kde_support(data[:, 0], bw_x, gridsize, cut, clip[0])
    y_support = _kde_support(data[:, 1], bw_y, gridsize, cut, clip[1])
    xx, yy = np.meshgrid(x_support, y_support)
    z = kde([xx.ravel(), yy.ravel()]).reshape(xx.shape)
    return xx, yy, z


def kdeplot(data, data2=None, shade=False, vertical=False, kernel="gau",
            bw="scott", gridsize=100, cut=3, clip=None, legend=True,
            cumulative=False, shade_lowest=True, ax=None, **kwargs):
    """Fit and plot a univariate or bivariate kernel density estimate.

    Parameters
    ----------
    data : 1d array-like
        Input data.
    data2: 1d array-like, optional
        Second input data. If present, a bivariate KDE will be estimated.
    shade : bool, optional
        If True, shade in the area under the KDE curve (or draw with filled
        contours when data is bivariate).
    vertical : bool, optional
        If True, density is on x-axis.
    kernel : {'gau' | 'cos' | 'biw' | 'epa' | 'tri' | 'triw' }, optional
        Code for shape of kernel to fit with. Bivariate KDE can only use
        gaussian kernel.
    bw : {'scott' | 'silverman' | scalar | pair of scalars }, optional
        Name of reference method to determine kernel size, scalar factor,
        or scalar for each dimension of the bivariate plot.
    gridsize : int, optional
        Number of discrete points in the evaluation grid.
    cut : scalar, optional
        Draw the estimate to cut * bw from the extreme data points.
    clip : pair of scalars, or pair of pair of scalars, optional
        Lower and upper bounds for datapoints used to fit KDE. Can provide
        a pair of (low, high) bounds for bivariate plots.
    legend : bool, optional
        If True, add a legend or label the axes when possible.
    cumulative : bool, optional
        If True, draw the cumulative distribution estimated by the kde.
    shade_lowest : bool, optional
        If True, shade the lowest contour of a bivariate KDE plot. Not
        relevant when drawing a univariate plot or when ``shade=False``.
        Setting this to ``False`` can be useful when you want multiple
        densities on the same Axes.
    ax : matplotlib axis, optional
        Axis to plot on, otherwise uses current axis.
    kwargs : key, value pairings
        Other keyword arguments are passed to ``plt.plot()`` or
        ``plt.contour{f}`` depending on whether a univariate or bivariate
        plot is being drawn.

    Returns
    -------
    ax : matplotlib Axes
        Axes with plot.

    See Also
    --------
    distplot: Flexibly plot a univariate distribution of observations.
    jointplot: Plot a joint dataset with bivariate and marginal distributions.

    Examples
    --------

    Plot a basic univariate density:

    .. plot::
        :context: close-figs

        >>> import numpy as np; np.random.seed(10)
        >>> import seaborn as sns; sns.set(color_codes=True)
        >>> mean, cov = [0, 2], [(1, .5), (.5, 1)]
        >>> x, y = np.random.multivariate_normal(mean, cov, size=50).T
        >>> ax = sns.kdeplot(x)

    Shade under the density curve and use a different color:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x, shade=True, color="r")

    Plot a bivariate density:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x, y)

    Use filled contours:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x, y, shade=True)

    Use more contour levels and a different color palette:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x, y, n_levels=30, cmap="Purples_d")

    Use a narrower bandwith:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x, bw=.15)

    Plot the density on the vertical axis:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(y, vertical=True)

    Limit the density curve within the range of the data:

    .. plot::
        :context: close-figs

        >>> ax = sns.kdeplot(x, cut=0)

    Plot two shaded bivariate densities:

    .. plot::
        :context: close-figs

        >>> iris = sns.load_dataset("iris")
        >>> setosa = iris.loc[iris.species == "setosa"]
        >>> virginica = iris.loc[iris.species == "virginica"]
        >>> ax = sns.kdeplot(setosa.sepal_width, setosa.sepal_length,
        ...                  cmap="Reds", shade=True, shade_lowest=False)
        >>> ax = sns.kdeplot(virginica.sepal_width, virginica.sepal_length,
        ...                  cmap="Blues", shade=True, shade_lowest=False)

    """
    if ax is None:
        ax = plt.gca()

    data = data.astype(np.float64)
    if data2 is not None:
        data2 = data2.astype(np.float64)

    bivariate = False
    if isinstance(data, np.ndarray) and np.ndim(data) > 1:
        bivariate = True
        x, y = data.T
    elif isinstance(data, pd.DataFrame) and np.ndim(data) > 1:
        bivariate = True
        x = data.iloc[:, 0].values
        y = data.iloc[:, 1].values
    elif data2 is not None:
        bivariate = True
        x = data
        y = data2

    if bivariate and cumulative:
        raise TypeError("Cumulative distribution plots are not"
                        "supported for bivariate distributions.")
    if bivariate:
        ax = _bivariate_kdeplot(x, y, shade, shade_lowest,
                                kernel, bw, gridsize, cut, clip, legend,
                                ax, **kwargs)
    else:
        ax = _univariate_kdeplot(data, shade, vertical, kernel, bw,
                                 gridsize, cut, clip, legend, ax,
                                 cumulative=cumulative, **kwargs)

    return ax


def rugplot(a, height=.05, axis="x", ax=None, **kwargs):
    """Plot datapoints in an array as sticks on an axis.

    Parameters
    ----------
    a : vector
        1D array of observations.
    height : scalar, optional
        Height of ticks as proportion of the axis.
    axis : {'x' | 'y'}, optional
        Axis to draw rugplot on.
    ax : matplotlib axes, optional
        Axes to draw plot into; otherwise grabs current axes.
    kwargs : key, value pairings
        Other keyword arguments are passed to ``axvline`` or ``axhline``.

    Returns
    -------
    ax : matplotlib axes
        The Axes object with the plot on it.

    """
    if ax is None:
        ax = plt.gca()
    a = np.asarray(a)
    vertical = kwargs.pop("vertical", axis == "y")
    func = ax.axhline if vertical else ax.axvline
    kwargs.setdefault("linewidth", 1)
    for pt in a:
        func(pt, 0, height, **kwargs)

    return ax


def jointplot(x, y, data=None, kind="scatter", stat_func=stats.pearsonr,
              color=None, size=6, ratio=5, space=.2,
              dropna=True, xlim=None, ylim=None,
              joint_kws=None, marginal_kws=None, annot_kws=None, **kwargs):
    """Draw a plot of two variables with bivariate and univariate graphs.

    This function provides a convenient interface to the :class:`JointGrid`
    class, with several canned plot kinds. This is intended to be a fairly
    lightweight wrapper; if you need more flexibility, you should use
    :class:`JointGrid` directly.

    Parameters
    ----------
    x, y : strings or vectors
        Data or names of variables in ``data``.
    data : DataFrame, optional
        DataFrame when ``x`` and ``y`` are variable names.
    kind : { "scatter" | "reg" | "resid" | "kde" | "hex" }, optional
        Kind of plot to draw.
    stat_func : callable or None, optional
        Function used to calculate a statistic about the relationship and
        annotate the plot. Should map `x` and `y` either to a single value
        or to a (value, p) tuple. Set to ``None`` if you don't want to
        annotate the plot.
    color : matplotlib color, optional
        Color used for the plot elements.
    size : numeric, optional
        Size of the figure (it will be square).
    ratio : numeric, optional
        Ratio of joint axes size to marginal axes height.
    space : numeric, optional
        Space between the joint and marginal axes
    dropna : bool, optional
        If True, remove observations that are missing from ``x`` and ``y``.
    {x, y}lim : two-tuples, optional
        Axis limits to set before plotting.
    {joint, marginal, annot}_kws : dicts, optional
        Additional keyword arguments for the plot components.
    kwargs : key, value pairings
        Additional keyword arguments are passed to the function used to
        draw the plot on the joint Axes, superseding items in the
        ``joint_kws`` dictionary.

    Returns
    -------
    grid : :class:`JointGrid`
        :class:`JointGrid` object with the plot on it.

    See Also
    --------
    JointGrid : The Grid class used for drawing this plot. Use it directly if
                you need more flexibility.

    Examples
    --------

    Draw a scatterplot with marginal histograms:

    .. plot::
        :context: close-figs

        >>> import numpy as np, pandas as pd; np.random.seed(0)
        >>> import seaborn as sns; sns.set(style="white", color_codes=True)
        >>> tips = sns.load_dataset("tips")
        >>> g = sns.jointplot(x="total_bill", y="tip", data=tips)

    Add regression and kernel density fits:

    .. plot::
        :context: close-figs

        >>> g = sns.jointplot("total_bill", "tip", data=tips, kind="reg")

    Replace the scatterplot with a joint histogram using hexagonal bins:

    .. plot::
        :context: close-figs

        >>> g = sns.jointplot("total_bill", "tip", data=tips, kind="hex")

    Replace the scatterplots and histograms with density estimates and align
    the marginal Axes tightly with the joint Axes:

    .. plot::
        :context: close-figs

        >>> iris = sns.load_dataset("iris")
        >>> g = sns.jointplot("sepal_width", "petal_length", data=iris,
        ...                   kind="kde", space=0, color="g")

    Use a different statistic for the annotation:

    .. plot::
        :context: close-figs

        >>> from scipy.stats import spearmanr
        >>> g = sns.jointplot("size", "total_bill", data=tips,
        ...                   stat_func=spearmanr, color="m")

    Draw a scatterplot, then add a joint density estimate:

    .. plot::
        :context: close-figs

        >>> g = (sns.jointplot("sepal_length", "sepal_width",
        ...                    data=iris, color="k")
        ...         .plot_joint(sns.kdeplot, zorder=0, n_levels=6))

    Pass vectors in directly without using Pandas, then name the axes:

    .. plot::
        :context: close-figs

        >>> x, y = np.random.randn(2, 300)
        >>> g = (sns.jointplot(x, y, kind="hex", stat_func=None)
        ...         .set_axis_labels("x", "y"))

    Draw a smaller figure with more space devoted to the marginal plots:

    .. plot::
        :context: close-figs

        >>> g = sns.jointplot("total_bill", "tip", data=tips,
        ...                   size=5, ratio=3, color="g")

    Pass keyword arguments down to the underlying plots:

    .. plot::
        :context: close-figs

        >>> g = sns.jointplot("petal_length", "sepal_length", data=iris,
        ...                   marginal_kws=dict(bins=15, rug=True),
        ...                   annot_kws=dict(stat="r"),
        ...                   s=40, edgecolor="w", linewidth=1)

    """
    # Set up empty default kwarg dicts
    if joint_kws is None:
        joint_kws = {}
    joint_kws.update(kwargs)
    if marginal_kws is None:
        marginal_kws = {}
    if annot_kws is None:
        annot_kws = {}

    # Make a colormap based off the plot color
    if color is None:
        color = color_palette()[0]
    color_rgb = mpl.colors.colorConverter.to_rgb(color)
    colors = [set_hls_values(color_rgb, l=l) for l in np.linspace(1, 0, 12)]
    cmap = blend_palette(colors, as_cmap=True)

    # Initialize the JointGrid object
    grid = JointGrid(x, y, data, dropna=dropna,
                     size=size, ratio=ratio, space=space,
                     xlim=xlim, ylim=ylim)

    # Plot the data using the grid
    if kind == "scatter":

        joint_kws.setdefault("color", color)
        grid.plot_joint(plt.scatter, **joint_kws)

        marginal_kws.setdefault("kde", False)
        marginal_kws.setdefault("color", color)
        grid.plot_marginals(distplot, **marginal_kws)

    elif kind.startswith("hex"):

        x_bins = _freedman_diaconis_bins(grid.x)
        y_bins = _freedman_diaconis_bins(grid.y)
        gridsize = int(np.mean([x_bins, y_bins]))

        joint_kws.setdefault("gridsize", gridsize)
        joint_kws.setdefault("cmap", cmap)
        grid.plot_joint(plt.hexbin, **joint_kws)

        marginal_kws.setdefault("kde", False)
        marginal_kws.setdefault("color", color)
        grid.plot_marginals(distplot, **marginal_kws)

    elif kind.startswith("kde"):

        joint_kws.setdefault("shade", True)
        joint_kws.setdefault("cmap", cmap)
        grid.plot_joint(kdeplot, **joint_kws)

        marginal_kws.setdefault("shade", True)
        marginal_kws.setdefault("color", color)
        grid.plot_marginals(kdeplot, **marginal_kws)

    elif kind.startswith("reg"):

        from .linearmodels import regplot

        marginal_kws.setdefault("color", color)
        grid.plot_marginals(distplot, **marginal_kws)

        joint_kws.setdefault("color", color)
        grid.plot_joint(regplot, **joint_kws)

    elif kind.startswith("resid"):

        from .linearmodels import residplot

        joint_kws.setdefault("color", color)
        grid.plot_joint(residplot, **joint_kws)

        x, y = grid.ax_joint.collections[0].get_offsets().T
        marginal_kws.setdefault("color", color)
        marginal_kws.setdefault("kde", False)
        distplot(x, ax=grid.ax_marg_x, **marginal_kws)
        distplot(y, vertical=True, fit=stats.norm, ax=grid.ax_marg_y,
                 **marginal_kws)
        stat_func = None
    else:
        msg = "kind must be either 'scatter', 'reg', 'resid', 'kde', or 'hex'"
        raise ValueError(msg)

    if stat_func is not None:
        grid.annotate(stat_func, **annot_kws)

    return grid
