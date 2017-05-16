'''Plotting package including some convenience tools

10/27/2015, started by JYL @PSI
'''
from .core import scale
from matplotlib import figure as fig
from matplotlib import pyplot as plt
import numpy as np

def pplot(ax=None, axfs='large', lfs='x-large', tightlayout=True, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, xscl=None, yscl=None, position=None, xticks=None, yticks=None, xticklabels=None, yticklabels=None, spinewidth=2, skipline=False, **kwargs):
	'''Pretty up a plot for publication.

 Parameters
 ----------
 ax : matplotlib.axes.AxesSubplot, optional
   An axis to prettify.  Default is all axes in the current figure.
 axfs : string, float, or int, optional
   Axis tick label font size.
 lfs : string, float, or int, optional
   Axis label font size.
 tightlayout : bool, optional
   Run `plt.tight_layout`.
 xlabel, ylabel : string, optinonal
   The labels of x-axls and y-axis.
 title : string, optional
   The title of plot.
 xlim, ylim : two-element array-like, number, optional
   The limit of x-axis and y-axis.
 xscl, yscl : string, optional
   Specify the scale for corresponding axis ('log', 'linear', etc)
 position : four-element array-like, number, optional
   The position of plot.
 xticks, yticks : array-like, number, optional
   The major ticks along x-axis and y-axis.
 xticklabels, yticklabels : array-like str, optional
   Labels for major ticks in x and y
 spinewidth : number, optional
   The line width of axis spines.
 skipline : bool, optional
   If `True`, then program will not change the line width
 **kwargs
   Any line or marker property keyword.

 v1.0.0 : JYL @PSI, Nov, 2013
 v1.0.1 : JYL @PSI, Oct 14, 2014
   * Correct an omission with keyword `skipline` for the recursive
     call
   * Move all prettification for plot art into the control by keyword
     `skipline`
	'''

	if ax is None:
		for ax in plt.gcf().get_axes():
			pplot(ax, tightlayout=tightlayout, axfs=axfs, lfs=lfs, skipline=skipline, **kwargs)

	# for the axes
	plt.setp(ax.get_ymajorticklabels(), fontsize=axfs)
	plt.setp(ax.get_xmajorticklabels(), fontsize=axfs)
	plt.setp(list(ax.spines.values()), linewidth=spinewidth)
	if xticks is not None:
		ax.set_xticks(xticks)
	if yticks is not None:
		ax.set_yticks(yticks)
	if position is not None:
		ax.set_position(position)
	if xscl is not None:
		ax.set_xscale(xscl)
	if yscl is not None:
		ax.set_yscale(yscl)

	# tick labels
	if xticklabels is not None:
		ax.xaxis.set_ticklabels(xticklabels)
	if yticklabels is not None:
		ax.yaxis.set_ticklabels(yticklabels)

	# axis labes
	labels = (ax.xaxis.get_label(), ax.yaxis.get_label())
	plt.setp(labels, fontsize=lfs)

	# for labels
	if xlabel is not None:
		ax.set_xlabel(xlabel)
	if ylabel is not None:
		ax.set_ylabel(ylabel)
	if title is not None:
		ax.set_title(title)

	# for plotting ranges
	if xlim is not None:
		ax.set_xlim(xlim)
	if ylim is not None:
		ax.set_ylim(ylim)

	if not skipline:
		mew = kwargs.pop('markeredgewidth', kwargs.pop('mew', 1.25))
		ms = kwargs.pop('markersize', kwargs.pop('ms', 7.0))
		lw = kwargs.pop('linewidth', kwargs.pop('lw', 2.0))
		plt.setp(ax.get_lines(), mew=mew, ms=ms, lw=lw, **kwargs)

	# for plot markers, ticks
	lines = ax.xaxis.get_minorticklines() + ax.xaxis.get_majorticklines() + ax.yaxis.get_minorticklines() + ax.yaxis.get_majorticklines()
	plt.setp(lines, mew=1.75)

	# the frame
	plt.setp(ax.patch, lw=2.0)

	if hasattr(plt, "tight_layout") and tightlayout:
		plt.sca(ax)
		plt.tight_layout()


def savefigs(fname, figs=None):
	'''Save multiple figures to a PDF file

 Parameters
 ----------
 fname : str, or Python file-like object, or
   matplotlib.backends.backend_pdf.PdfPages object
 figs : array-like, matplotlib.figure.Figure object, optional
   Figure(s) to be saved.  Default is to save all active figures.

 Returns
 -------
 None

 v1.0.0 : JYL @PSI, Feb 14, 2014
	'''

	from matplotlib.backends.backend_pdf import PdfPages
	if not isinstance(fname, PdfPages):
		p = PdfPages(fname)
	else:
		p = fname
	if figs is None:
		figs = []
		for i in plt.get_fignums():
			plt.figure(i)
			figs.append(plt.gcf())
	if hasattr(figs,'__iter__'):
		for f in figs:
			p.savefig(f)
	else:
		p.savefig(figs)
	p.close()


def histogram(arr, *args, **kwargs):
	'''Plot histogram of input array

 arr : array-like
   Input array to be plotted
 cumulative : bool, optional
   Plot cumulative distribution
 inversed : bool, optional
   If `True`, then the cumulative distribution will be calculated from
   maximum to minimum.  Otherwise from minimum to maximum
 *args : arguments accepted by pyplot.step
 *all keywords accepted by numpy.histogram*
 *all keywords accepted by pylot.step

 v1.0.0 : 10/27/2015, JYL @PSI
	'''

	cumulative=kwargs.pop('cumulative', False)
	inversed = kwargs.pop('inversed', True)

	# numpy.histogram keywords
	bins = kwargs.pop('bins', 10)
	range = kwargs.pop('range', None)
	normed = kwargs.pop('normed', False)
	weights = kwargs.pop('weights', None)
	density = kwargs.pop('density', None)

	hist, bedges = np.histogram(arr, bins, range, normed, weights, density)
	if cumulative:
		if inversed:
			idx = slice(None, None, -1)
		else:
			idx = slice(None, None)
		hist = hist[idx].cumsum()[idx]
	plt.step(bedges[:-1], hist, *args, where='post', **kwargs)


def density(x, y, log=False, ax=None, **kwargs):
	'''Plot 2-D distribution density plot for input data.

 x, y : array-like, shape (N,)
   The x and y of the points to be plotted.
 log : bool, optional
   If `True`, then the density plot will be displayed in log grayscale
 *some keywords accepted by numpy.histogram2d*
   bins, range
 *some keywords accepted by imshow*
  cmap, interpolation, alpha, vmin, vmax
 *some keywords accepted by pplot*
  xticks, yticks, xticklabels, yticklabels

 v1.0.0 : 10/27/2015, JYL @PSI
 v1.0.1 : 1/12/2016, JYL @PSI
   Removed a bug that cause the density plot to be shifted when the
   bins size in x and y are different
	'''

	# numpy.histogram2d keywords
	bins = kwargs.pop('bins', 10)
	range = kwargs.pop('range', None)
	normed = kwargs.pop('normed', False)
	weights = kwargs.pop('weights', None)

	# Some pyplot.imshow keywords
	cmap = kwargs.pop('cmap', None)
	interpolation = kwargs.pop('interpolation', None)
	alpha = kwargs.pop('alpha', None)

	# Some pplot keywords
	xticks = kwargs.pop('xticks', None)
	yticks = kwargs.pop('yticks', None)
	xlabel = kwargs.pop('xlabel', None)
	ylabel = kwargs.pop('ylabel', None)

	hist, xedges, yedges = np.histogram2d(x, y, bins, range, normed, weights)
	hist = hist.T  # pyplot.imshow() displays images in row-major order.  Needs to transpose.
	if log:
		vmin = kwargs.pop('vmin', hist[hist>0].min())
	else:
		vmin = kwargs.pop('vmin', hist.min())
	vmax = kwargs.pop('vmax', hist.max())

	# Density plot axis
	xlim = [xedges.min(),xedges.max()]
	ylim = [yedges.min(),yedges.max()]
	ybs, xbs = hist.shape

	if ax is None:
		ax = plt.subplot(111)
	pplot(xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
	xt0 = ax.get_xticks()
	yt0 = ax.get_yticks()

	if log:
		hist = np.log10(np.clip(hist, a_min=vmin, a_max=vmax))
		vmax = np.log10(vmax)
		vmin = np.log10(vmin)
	im = ax.imshow(hist, aspect='auto', cmap=cmap, interpolation=interpolation, alpha=alpha, vmin=vmin, vmax=vmax)
	xtrans = lambda x: (float(x)-xlim[0])/(xlim[1]-xlim[0])*xbs-0.5
	ytrans = lambda x: (float(y)-ylim[0])/(ylim[1]-ylim[0])*ybs-0.5
	xt1 = [xtrans(x) for x in xt0]
	yt1 = [ytrans(y) for y in yt0]
	xticklabels = kwargs.pop('xticklabels', list(map(str, xt0)))
	yticklabels = kwargs.pop('yticklabels', list(map(str, yt0)))
	pplot(xlim=[-0.5,xbs-0.5], ylim=[-0.5,ybs-0.5], xticks=xt1, yticks=yt1, xticklabels=xticklabels, yticklabels=yticklabels, xlabel=xlabel, ylabel=ylabel)

	# Color scale axis
	cbar = plt.colorbar(im, ax=ax)
	ctk = cbar.ax.get_yticks()*(vmax-vmin)+vmin
	if log:
		ctk = 10**ctk
	ctk = [str(float('%.3g' % x)) for x in ctk]
	cbar.ax.set_yticklabels(ctk)


def imshow(*var, **kwargs):
	'''Improved version of matplotlib.pyplot.imshow

	xticks, yticks, xlabel, ylabel

	'''
	ax = kwargs.pop('ax', None)
	x = kwargs.pop('x', None)
	y = kwargs.pop('y', None)
	xticks = kwargs.pop('xticks', None)
	yticks = kwargs.pop('yticks', None)
	xticklabels = kwargs.pop('xticklabels', None)
	yticklabels = kwargs.pop('yticklabels', None)
	xlabel = kwargs.pop('xlabel', None)
	ylabel = kwargs.pop('ylabel', None)
	xlim = kwargs.pop('xlim', None)
	ylim = kwargs.pop('ylim', None)
	aspect = kwargs.pop('aspect', None)
	kwargs['aspect'] = 'auto'

	ys, xs = var[0].shape
	if xlim is None:
		if x is None:
			xlim = [0,xs]
		else:
			xlim = [x[0],x[-1]]
	if ylim is None:
		if y is None:
			ylim = [0,ys]
		else:
			ylim = [y[0],y[-1]]

	yb = (ylim[0]-y[-1])/y[0]
	xb = (xlim[1]-x[-1])/y[1]

	plt.clf()
	if ax is None:
		ax = plt.subplot(111)
	pplot(ax, xlim=xlim, ylim=ylim, xticks=xticks, yticks=yticks)
	xt0 = ax.get_xticks()
	yt0 = ax.get_yticks()
	plt.clf()

	im = plt.imshow(*var, **kwargs)
	ax = plt.gca()
	xt1 = scale(xt0, [0,xs])
	yt1 = scale(yt0, [0,ys])
	if xlim[0]>xlim[1]:
		xt1 = xt1[::-1]
	if ylim[0]>ylim[-1]:
		yt1 = yt1[::-1]

	if xticklabels is None: xticklabels = list(map(str, xt0))
	if yticklabels is None: yticklabels = list(map(str, yt0))
	pplot(ax,xlim=[0, xs], ylim=[0, ys], xticks=xt1, yticks=yt1, xticklabels=xticklabels, yticklabels=yticklabels, xlabel=xlabel, ylabel=ylabel)
