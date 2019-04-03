# Copyright 2016 Jeff Vierstra

import numpy as np

from matplotlib.ticker import MaxNLocator
from matplotlib.transforms import blended_transform_factory

from .. import load_data
from track import track, segment

class continuous_data_track(track):

	def __init__(self, interval, data=None, **kwargs):
		super(continuous_data_track, self).__init__(interval, **kwargs)
		self.data = data

	def format_axis(self, ax):
		super(continuous_data_track, self).format_axis(ax)

		ax.spines['bottom'].set_color('black')
		ax.spines['left'].set_color('black')

		locator = MaxNLocator(3, prune = 'upper')
		ax.yaxis.set(visible = True, major_locator = locator)

		if 'min' in self.options:
			ax.set_ylim(bottom = self.options['min'])

		if 'max' in self.options:
			ax.set_ylim(top = self.options['max'])

	def load_data(self, filepath, column=5, dtype=float):
		self.data = load_data(filepath, interval, [column], dtype)


	def density(self, vals, window_size = 150, step_size = 20):
		"""Smooth data in density windows"""
		
		pos = np.arange(0, len(vals)-window_size, step_size)
		s = np.zeros(len(pos))
		for i, j in enumerate(pos):
			s[i] = np.sum(vals[j:j+window_size])
		return pos + (window_size/2) + self.interval.start, s

	def simplify(self, step = 100):
		"""Simplify data points for fast plotting -- there is probably a better way of doing this..."""
		
		xx = []
		yy = []
		for i in range(step, len(self.data)-step, step):
			xx.append(self.interval.start + i)
			yy.append(np.amax(self.data[i-step:i+step]))
		return np.array(xx), np.array(yy)

	def step(self, vals, xaxis = False, step_interval = 0):
		"""Creates a step-style plot"""

		if xaxis and step_interval == 0:
			step_interval = abs(vals[1] - vals[0]) / 2.0
		step_vals = np.array(zip(vals - step_interval, vals + step_interval)).ravel()
		return step_vals

	def render(self, ax):

		if not self.data:
			raise Exception("No data loaded!")

		self.format_axis(ax)

		if 'density' in self.options:
			xx, yy = self.density(self.data)
			xs = self.step(xx, xaxis = True)
			ys = self.step(yy)
		else:
			xs = self.step(np.arange(self.interval.start, self.interval.end), xaxis = True)
			ys = self.step(self.data)


		if 'simplify' in self.options:
			xx, yy = self.simplify(step = self.options['simplify'])
			xs = self.step(xx, xaxis = True)
			ys = self.step(yy, xaxis = False)

		(ybot, ytop) = ax.get_ylim()
		ys[ys > ytop] = ytop
		ys[ys < ybot] = ybot

		if 'fill_between' in self.options:
			ax.fill_between(xs, np.ones(len(xs)) * self.options['fill_between'], ys, 
				facecolor = self.options.get('facecolor', 'blue'), 
				edgecolor = self.options['edgecolor'])
		else:
			ax.plot(xs, ys)

		# change to linecollection
		if 'clip_bar' in self.options:
			trans = blended_transform_factory(ax.transData, ax.transAxes)
			for x0, x1 in segment(self.data, ytop):
				ax.plot([x0 + self.interval.start, x1 + self.interval.start], [1, 1], color = self.options['clip_bar']['color'], lw = self.options['clip_bar']['lw'], transform = trans, clip_on = False)


