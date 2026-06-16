"""Domain decomposition helpers for spatial curriculum weights. / 空间课程权重使用的区域划分工具。"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from deepxde.geometry import Interval, Rectangle
import deepxde as dde

# Split sampled points into subdomains and keep their hierarchy. / 将已采样点划分为多个子域并记录层次关系。
class Pointset1D:
    def __init__(self, geom, num_subdomains):
        self.geom = geom
        self.num_subdomains = num_subdomains
        self.intervals = None
        self.distance_to_boundary = None
        self.split_points_num = [0 for _ in range(num_subdomains)]
        self.frame_points = None
        self.inside_anchors = None

    def split(self, savepath=None, col_points=None):
        l, r = self.geom.l, self.geom.r
        edges = np.linspace(l, r, self.num_subdomains + 1)
        intervals = [Interval(edges[i], edges[i + 1]) for i in range(len(edges) - 1)]
        self.intervals = intervals
        n = self.num_subdomains
        self.distance_to_boundary = [min(i, n - 1 - i) for i in range(n)]
        if savepath is not None:
            savepath.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 1.8))
            colors = plt.cm.tab10(np.linspace(0, 1, self.num_subdomains))
            if col_points is not None:
                for i, pts in enumerate(col_points):
                    ax.scatter(pts[:, 0], np.zeros_like(pts[:, 0]), s=1, color=colors[i], label=f"col part {i}")
            else:
                for i, interval in enumerate(self.intervals):
                    pts = interval.uniform_points(20, boundary=True)
                    ax.scatter(pts[:, 0], np.zeros_like(pts[:, 0]), marker="|", s=20, color=colors[i], label=f"part {i}")
            ax.set_yticks([])
            ax.set_xlabel("x (subdomain splits)")
            ax.set_title("Subdomain spans (1D)")
            fig.tight_layout()
            out = savepath / "subdomain_splits.png"
            fig.savefig(out)
            print(f"Saved subdomain splits plot to {out}\n")
            plt.close(fig)

            # Color intervals by distance to the outer boundary. / 按到外边界的距离给区间着色。
            fig_fill, ax_fill = plt.subplots(figsize=(8, 1.2))
            unique_dist = sorted(set(self.distance_to_boundary))
            cmap = plt.cm.get_cmap("tab10", len(unique_dist))
            for i, interval in enumerate(self.intervals):
                dist = self.distance_to_boundary[i]
                color = cmap(unique_dist.index(dist))
                ax_fill.fill_between([interval.l, interval.r], [0, 0], [1, 1], color=color, alpha=0.4, step="pre")
                ax_fill.plot([interval.l, interval.r], [0, 0], color="k", linewidth=0.8)
                ax_fill.text((interval.l + interval.r) / 2, 0.55, f"d={dist}", ha="center", va="center", fontsize=8)
            ax_fill.set_yticks([])
            ax_fill.set_xlabel("x")
            ax_fill.set_xlim(edges[0], edges[-1])
            ax_fill.set_title("Subdomain distance-to-boundary coloring")
            fig_fill.tight_layout()
            out_fill = savepath / "subdomain_distance_fill.png"
            fig_fill.savefig(out_fill)
            print(f"Saved distance-colored spans to {out_fill}")
            plt.close(fig_fill)


    def filter_points(self, points):
        grouped_points = [None] * len(self.intervals)
        self.split_points_num = [0 for _ in range(self.num_subdomains)]
        for i in range(len(self.intervals)):
            interval = self.intervals[i]
            inside_idx = interval.inside(points)
            grouped_points[i] = points[inside_idx]
            self.split_points_num[i] = grouped_points[i].shape[0]

        return grouped_points

    def gen_frame_points(self, num_each_part=2, savepath=None):
        boundary_groups = []
        for interval in self.intervals:
            boundary_pts = interval.uniform_boundary_points(num_each_part)
            boundary_groups.append(boundary_pts)
        merged = np.vstack(boundary_groups).astype(dtype=dde.config.real(np))
        if savepath is not None:
            savepath.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(8, 1.8))
            ax.scatter(merged[:, 0], np.zeros_like(merged[:, 0]), marker="|", s=80)
            ax.set_yticks([])
            ax.set_xlabel("x (boundary points)")
            ax.set_title("Subdomain boundary points (1D)")
            out = savepath / "subdomain_boundary_points.png"
            fig.savefig(out)
            print(f"Saved subdomain boundary points plot to {out}\n")
            plt.close(fig)

        self.frame_points = merged

    def gen_inside_anchors(self, num_each_part=10, savepath=None):
        anchor_groups = []
        for interval in self.intervals:
            anchor_pts = interval.uniform_points(num_each_part, boundary=False)
            anchor_groups.append(anchor_pts)

        self.inside_anchors = anchor_groups


class Pointset2D:
    def __init__(self, geom, num_subdomains: list[int,int]):
        self.geom = geom
        self.num_subdomains = num_subdomains
        self.rectangles = None
        self.distance_to_boundary = None
        self.split_points_num = [0 for _ in range(num_subdomains[0] * num_subdomains[1])]
        self.frame_points = None
        self.inside_anchors = None

    def split(self, savepath=None, col_points=None):
        x_min, x_max = self.geom.xmin[0], self.geom.xmax[0]
        y_min, y_max = self.geom.xmin[1], self.geom.xmax[1]
        x_edges = np.linspace(x_min, x_max, self.num_subdomains[0] + 1)
        y_edges = np.linspace(y_min, y_max, self.num_subdomains[1] + 1)
        rectangles = []
        self.split_points_num = []
        for i in range(len(x_edges) - 1):
            for j in range(len(y_edges) - 1):
                rect = Rectangle([x_edges[i], y_edges[j]], [x_edges[i + 1], y_edges[j + 1]])
                rectangles.append(rect)
        self.rectangles = rectangles
        n_x, n_y = self.num_subdomains
        self.distance_to_boundary = []
        for i in range(n_x):
            for j in range(n_y):
                dist = min(i, n_x - 1 - i, j, n_y - 1 - j)
                self.distance_to_boundary.append(dist)

        if savepath is not None:
            savepath.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            total_parts = self.num_subdomains[0] * self.num_subdomains[1]
            cmap_parts = plt.cm.get_cmap("turbo", total_parts)
            color_order = np.random.default_rng(0).permutation(total_parts)
            if col_points is not None:
                for i, pts in enumerate(col_points):
                    ax.scatter(pts[:, 0], pts[:, 1], s=1, color=cmap_parts(color_order[i]), label=f"col part {i}")
            else:
                for i, rect in enumerate(self.rectangles):
                    pts = rect.uniform_points(100, boundary=True)
                    ax.scatter(pts[:, 0], pts[:, 1], s=1, color=cmap_parts(color_order[i]), label=f"part {i}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Subdomain spans (2D)")
            out = savepath / "subdomain_splits.png"
            fig.savefig(out)
            print(f"Saved subdomain splits plot to {out}\n")
            plt.close(fig)

            # Color rectangles by distance to the outer boundary. / 按到外边界的距离给矩形子域着色。
            fig_fill, ax_fill = plt.subplots(figsize=(5, 5))
            unique_dist = sorted(set(self.distance_to_boundary))
            cmap = plt.cm.get_cmap("tab20", len(unique_dist))
            idx = 0
            for i in range(n_x):
                for j in range(n_y):
                    rect = self.rectangles[idx]
                    dist = self.distance_to_boundary[idx]
                    color = cmap(unique_dist.index(dist))
                    x0, y0 = rect.xmin
                    x1, y1 = rect.xmax
                    ax_fill.fill_between([x0, x1], y0, y1, color=color, alpha=0.4, step="post")
                    ax_fill.text((x0 + x1) / 2, (y0 + y1) / 2, f"d={dist}", ha="center", va="center", fontsize=7)
                    idx += 1
            ax_fill.set_xlim(x_min, x_max)
            ax_fill.set_ylim(y_min, y_max)
            ax_fill.set_xlabel("x")
            ax_fill.set_ylabel("y")
            ax_fill.set_title("Subdomain distance-to-boundary coloring (2D)")
            out_fill = savepath / "subdomain_distance_fill.png"
            fig_fill.savefig(out_fill)
            print(f"Saved distance-colored spans to {out_fill}\n")
            plt.close(fig_fill)


    def filter_points(self, points):
        grouped_points = [None] * len(self.rectangles)
        self.split_points_num = [0 for _ in range(len(self.rectangles))]
        for i in range(len(self.rectangles)):
            rectangle = self.rectangles[i]
            inside_idx = rectangle.inside(points)
            grouped_points[i] = points[inside_idx]
            self.split_points_num[i] = grouped_points[i].shape[0]
        return grouped_points

    def gen_frame_points(self, num_each_part=20, savepath=None):
        boundary_groups = []
        for rect in self.rectangles:
            boundary_pts = rect.uniform_boundary_points(num_each_part)
            boundary_groups.append(boundary_pts)
        merged = np.vstack(boundary_groups).astype(dtype=dde.config.real(np))
        if savepath is not None:
            savepath.parent.mkdir(parents=True, exist_ok=True)
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.scatter(merged[:, 0], merged[:, 1], s=8)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_title("Subdomain boundary points")
            out = savepath / "subdomain_boundary_points.png"
            fig.savefig(out)
            print(f"Saved subdomain boundary points plot to {out}\n")
            plt.close(fig)

        self.frame_points = merged

    def gen_inside_anchors(self, num_each_part=10, savepath=None):
        anchor_groups = []
        for rec in self.rectangles:
            anchor_pts = rec.uniform_points(num_each_part, boundary=False)
            anchor_groups.append(anchor_pts)

        self.inside_anchors = anchor_groups
