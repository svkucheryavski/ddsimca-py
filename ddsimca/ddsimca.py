import math
import itertools
import pandas as pd
import numpy as np
import numpy.typing as npt
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from matplotlib.axes import Axes
from typing import TypeVar
from scipy.stats.distributions import chi2, norm, binom

# maps indices to names of roles and decisions
ind_to_roles = ['regular', 'extreme', 'outlier', 'alien', 'external']
ind_to_decisions = ['out', 'in']

# annotations for correct typing
Array2D = npt.NDArray[np.float64]

# default plot colors and markers
MAIN_COLOR = "tab:blue"

COLORS_FOMS = {
   "eff":   "tab:purple",
   "sens":  "tab:blue",
   "spec":  "tab:cyan",
   "sel":   "tab:cyan",
   "acc":   "gray"
}

COLORS_ROLES = {
   "regular":  "tab:blue",
   "extreme":  "tab:orange",
   "outlier":  "tab:red",
   "alien":    "tab:blue",
   "external": "tab:red"
}

COLORS_DECISIONS = {
   "in":    "tab:blue",
   "out":  "tab:red",
}

MARKERS = ['o', 's', '^', 'd', '>', 'h', 'p', 'v']

# a small number to add to sd
EPSILON = 1e-12



#######################################
# Auxillary methods for plots         #
#######################################

def get_group_colors(groups:list) -> dict:
    """
    Returns dictionary with dedicated color for each group name.
    """
    n = len(groups)

    if n == 1:
        colors = ['tab:blue']
    elif n == 2:
        colors = ['tab:blue', 'tab:red']
    elif n == 3:
        colors = ['tab:blue', 'tab:orange', 'tab:red']
    elif n <= 10:
        cmap = plt.get_cmap("tab10", n)
        colors = [cmap(i) for i in range(n)]
    elif n <= 20:
        cmap = plt.get_cmap("tab20", n)
        colors = [cmap(i) for i in range(n)]
    else:
        raise ValueError("Number of groups is too large (>20) for distingushing them with colors.")

    return dict(zip(groups, colors))


def plot_axes(ax:Axes, type:str = "p") -> None:
    """
    Add horizontal and vertical (only if `type = "p"`) lines crossing (0,0) origin to a plot.
    """
    ax.axhline(0, color="#a0a0a0", linewidth = 0.75, zorder = 1)
    if type == "p":
        ax.axvline(0, color="#a0a0a0", linewidth = 0.75, zorder = 1)


def plot_grid(ax:Axes) -> None:
    """
    Add lightgray grid to a plot and place it behind other plot elements.
    """
    ax.set_axisbelow(True)
    ax.grid(True, linestyle = "--", color = "#e0e0e0")


def plot_labels(ax:Axes, x:npt.NDArray, y:npt.NDArray, labels:npt.NDArray, dy:float = 0 ) -> None:
    """
    Show labels on top of data points with coordinates (x, y)
    """
    for i in range(0, len(x)):
        ax.text(x[i], y[i] + dy, labels[i], color="gray", ha = "center")


def plot_compstats(ax:Axes, y:npt.NDArray, color:str = MAIN_COLOR, marker:str = "o", label:str = "") -> None:
    """
    Show plot with a statistic vs number of components.
    """
    ncomp = len(y)
    comp_seq = list(range(1, ncomp + 1))
    ax.plot(comp_seq, y, color = color, markeredgecolor = color, markerfacecolor = "#ffffff",
        marker = marker, label = label)
    ax.set_xlim((0, ncomp + 1))
    ax.set_xlabel("Number of PCs")
    plot_grid(ax)



#######################################
# Auxillary methods for calculations  #
#######################################

def get_limits(u0:float, Nu:float, CLe:float = 0.95, CLo:float = 0.9983) -> tuple[float, float]:
    """
    Compute statistical limits for extreme objects and outliers based on chi-square distribution.
    """
    qe = chi2.ppf(CLe, Nu)
    qo = chi2.ppf(CLo, Nu)
    return (float(qe) * u0 / Nu, float(qo) * u0 / Nu)


def get_distparams(U: Array2D, type:str = 'classic') -> tuple[npt.NDArray[np.float64],
      npt.NDArray[np.float64]]:
    """
    Computes parameters of a scaled chi-square distribution that approximate the distribution of the distance values
    using the method of moments.

    Parameters
    ----------
    U : NDArray
        A matrix (2D NumPy array, nrows x ncomp) of distances to compute the distribution parameters for.

    Returns
    -------
    tuple

        u0 : NDArray
            The scalar values for each column of U.
        Nu : NDArray
            The estimated number of degrees of freedom for each column of U.

    Raises
    ------
    ValueError
        If the input array is empty.

    """

    if len(U.shape) != 2:
        raise ValueError("Argument U must be a matrix (2D array).")

    if type == 'classic':
        u0 = U.mean(axis = 0)
        vu = U.std(axis = 0, ddof=1.)**2

        u02 = u0 ** 2
        u02[u02 < EPSILON] = EPSILON
        vu[vu < EPSILON] = EPSILON

        Nu = np.round(np.divide(2 * u02, vu))
        Nu[Nu < 1] = 1
        Nu[Nu > 250] = 250

        return (u0, Nu)

    Mu = np.median(U, axis = 0)
    Su = np.quantile(U, 0.75, axis = 0) - np.quantile(U, 0.25, axis = 0)
    RM = np.divide(Su, Mu)

    Nu = np.zeros(len(RM))
    Nu[RM > 2.685592117] = 1
    Nu[RM < 0.194565995] = 100

    ind = (RM >= 0.194565995) & (RM <= 2.685592117)
    Nu[ind] = np.round(np.exp((1.380948 * np.log(2.68631 / RM[ind])) ** 1.185785))

    tQ2 = chi2.ppf(0.50, Nu)
    tIQR = chi2.ppf(0.75, Nu) - chi2.ppf(0.25, Nu)
    u0 = 0.5 * Nu * (Mu / tQ2 + Su / tIQR)

    return (u0, Nu)


def process_members(f:npt.NDArray[np.float64], eCrit:float, oCrit:float, roles:npt.NDArray[np.int16],
    ind:npt.NDArray[np.bool_]|None) -> tuple[int, int]:
    """
    Process objects as target class members by assigning corresponding roles.

    Parameters
    ----------
    f : NDArray
        Vector with full distance values for each object.
    eCrit : float
        Critical f-value for extreme objects.
    oCrit : float
        Critical f-value for outliers.
    roles : NDArray
        Vector (1D Array) with numeric indices for roles (will be changed in place).
    ind : NDArray
        Vector (1D Array) with logical values which point on target class members.

    Returns
    -------
    Tuple with number of true positives (TP) and false negatives (FN). In addition to that the function modifies
    list with roles by assigning `0` to regular objects, `1` to extreme and `2` to outliers.
    """
    if ind is None or ind.sum() < 1:
        return (0, 0)


    reg_ind = ind & (f <= eCrit)
    out_ind = ind & (f > oCrit)
    ext_ind = ind & (f > eCrit) & (f < oCrit)

    roles[reg_ind] = 0
    roles[ext_ind] = 1
    roles[out_ind] = 2

    TP = reg_ind.sum()
    FN = ind.sum() - TP
    return (TP, FN)


def process_strangers(f:npt.NDArray[np.float64], k:float, eCrit:float, roles:npt.NDArray[np.int16],
    ind:npt.NDArray[np.bool_]|None) -> tuple[int, int, float, float, float, float, float, float, float, float]:
    """
    Process objects as non-members by assigning corresponding roles.

    Parameters
    ----------
        f : NDArray
            Vector with full distance values for each object.
        k : float
            Number of degrees of freedom for f (also known as Nf).
        eCrit : float
            Critical f-value for extreme objects.
        roles : NDArray
            Vector (1D Array) with numeric indices for roles (will be changed in place).
        ind : NDArray
            Vector (1D Array) with logical values which point on alternative class members.

    Returns
    -------
    Tuple with following values

        TN
            Number of true negatives.
        FP
            Number of false positives.
        beta
            Probability to make Type II error.
        s, f0, hz, Mz, Sz, k, m
            Parameters used to fit a non-central chi-square distribution.


    In addition to that, the function modifies list with roles by assigning `3` to alien objects and `4` to external objects.
    """

    if ind is None or ind.sum() < 1:
        return (0, 0, 0., 0., 0., 0., 0., 0., 0., 0.)

    ind_in = ind & (f <= eCrit)
    ind_out = ind & (f > eCrit)
    TN = ind_out.sum()
    FP = ind_in.sum()

    # by default all non-members are aliens
    roles[ind] = 3

    # Step 1. Sort all distances
    fs = f[ind]
    indv = fs.argsort()
    fp = fs[indv]
    ind_num = np.where(ind)[0]

    # Step 2. Try to fit the non-central chi-square
    I = len(indv)
    Disc = -1
    n = 0
    m = 0
    d = 0
    M1 = 0

    while Disc < 0:
        if n > 0:
            # sample with largest f does not fit, so we change its rolw to "external"
            # and amend the number of aliens and externals
            roles[ind_num[indv[I - 1]]] = 4

            # then we remove this sample from the temporary vector and
            # assess the next biggest
            I = I - 1
            indv = indv[:-1]
            fp = fp[:-1]

        # compute parameters for moments squared equation and
        # its discriminant
        m = fp.mean()
        d = fp.var(ddof = 1)
        M1 = d / (m * m)
        Disc = 4 - 2 * k * M1
        n = n + 1

    # Step 3. Calculate  x  by Eq. (18)
    x = (2 + math.sqrt(Disc)) / M1

    # Calculate  s and  f'0 by Eq. (19)
    s = (x - k)
    f0 = m / x


    # Calculate: z, hz, r, p, Mz, Sz by Eq. (20)
    z = eCrit / f0
    hz = 1 - 2 * (k + s) * (k + 3 * s) / (3 * (k + 2 * s)**2)
    r = (hz - 1) * (1 - 3 * hz)
    p = (k + 2 * s) / (k + s)**2
    Mz = 1 + hz * p * (hz - 1 - 0.5 * (2 - hz) * r * p)
    Sz = hz * math.sqrt(2 * p) * (1 + 0.5 * r * p)

    # Step 4.	If α is given then calculate β by Eq. (21)
    beta = float(norm.cdf((math.pow(z / (k + s), hz) - Mz) / Sz))
    return (TN, FP, beta, s, f0, hz, Mz, Sz, k, m)


####################################
# DDSIMCARes class                 #
####################################

class DDSIMCARes:
    """
    A class to hold and process results from the DDSIMCA model predictions. Do not use it manually,
    it is used by the 'predict' method from the DDSIMCA class.

    Class methods
    -------------
    `select_ncomp()`
        Sets (selects) optimal number of components.
    `summary()`
        Prints a summary with main figures of merits and corresponding statistics.
    `as_df()`
        Returns data frame with role, decision and distances for every object.
    `plotFoM()`
        Show a plot with selected figure of merit vs. number of components.
    `plotDistance()`
        Shows a plot with score, orthogonal or full distance vs object index.
    `plotAcceptance()`
        Shows the acceptance plot.
    `plotScores()`
        Shows a scores plot.
    `plotExtremes()`
        Shows extremes plot.
    `plotAliens()`
        Shows aliens plot.
    """
    def __init__(self, target_class, hParams, qParams, fParams, center, scale, alpha, gamma,
        lim_type, I, H, Q, T, E, classes, labels, ncomp_selected):

        nrows, ncomp = Q.shape
        h0, Nh = hParams
        q0, Nq = qParams
        f0, Nf = fParams

        self.nrows = nrows
        self.ncomp = ncomp
        self.ncomp_selected = ncomp_selected
        self.center = center
        self.scale = scale

        self.T = T
        self.E = E
        self.Q = Q
        self.H = H
        self.F = (H / h0) * Nh + (Q / q0) * Nq

        self.hParams = hParams
        self.qParams = qParams
        self.fParams = fParams
        self.labels = labels

        self.has_classes = classes is not None and len(classes) > 0
        self.classes = np.unique(classes) if self.has_classes else []
        self.class_labels = classes
        self.target_class = target_class

        # confidence levels and critical limits
        self.alpha = alpha
        self.gamma = gamma
        self.lim_type = lim_type

        self.CLe = 1 - alpha
        self.CLo = (1 - gamma)**(1.0 / I)

        if self.has_classes:
            ind_members = self.class_labels == target_class
            ind_strangers = ~ind_members
            ind_unknowns = np.full(nrows, False)
            num_members = sum(ind_members)
            num_strangers = sum(ind_strangers)
            num_unknowns = 0
        else:
            ind_members = np.full(nrows, False)
            ind_strangers = np.full(nrows, False)
            ind_unknowns = np.full(nrows, True)
            num_members = 0
            num_strangers = 0
            num_unknowns = nrows

        # outcomes
        outcomes = [None] * ncomp
        self.R = np.zeros((self.nrows, self.ncomp), dtype=np.int16) # matrix with roles
        self.D = np.full((self.nrows, self.ncomp), False) # matrix with decisions


        for a in range(ncomp):
            eCrit, oCrit = get_limits(f0[a], Nf[a], self.CLe, self.CLo)
            f = self.F[:, a]
            roles = np.zeros(nrows, dtype=np.int16)
            decisions = f < eCrit

            TP, FN = process_members(f, eCrit, oCrit, roles, ind_members)
            TN, FP, beta, s, f0t, hz, Mz, Sz, k, m = process_strangers(f, Nf[a], eCrit, roles, ind_strangers)

            if num_unknowns > 0:
                _ = process_strangers(f, Nf[a], eCrit, roles, ind_unknowns)

            if num_members > 0:
                sens = TP / (TP + FN)
            else:
                sens = 0

            if num_strangers > 0:
                spec = TN / (TN + FP)
                sel  = 1 - beta
            else:
                spec = 0
                sel = 0

            if num_strangers > 0 and num_members > 0:
                eff = math.sqrt(sens * spec)
                acc = (TP + TN) / nrows
            else:
                eff = 0
                acc = 0

            num_in = np.sum(decisions)
            num_out = nrows - num_in
            outcomes[a] = {"PCs": a + 1, "eCrit": eCrit, "oCrit": oCrit, "TP": TP, "FN": FN, "TN": TN, "FP": FP,
                           "beta": beta, "s": s, "f0t": f0t, "hz": hz, "Mz": Mz, "Sz": Sz, "k": k, "m": m,
            "in": num_in, "out": num_out, "sens": sens, "spec": spec, "sel": sel, "acc": acc,
            "eff": eff}

            self.R[:, a] = roles
            self.D[:, a] = decisions

        self.num_members = num_members
        self.num_strangers = num_strangers
        self.num_unknowns = num_unknowns

        self.ind_members = ind_members
        self.ind_strangers = ind_strangers
        self.ind_unknowns = ind_unknowns

        self.outcomes = pd.DataFrame(outcomes)


    def select_ncomp(self, ncomp:int):
        """
        Change (select) optimal number of components.

        Parameters
        ----------
            ncomp : int
                Number of components to set.
        """
        if ncomp < 1 or ncomp > self.ncomp:
            raise ValueError(f"Wrong value for 'ncomp' parameter (must be between 1 and {self.ncomp}")
        self.ncomp_selected = ncomp


    def summary(self):
        """
        Prints a summary with general information about the dataset
        used to create this result ibject as well as classification statistic
        and figures of merit for each number of components in the model.
        """

        print('\033[1m', end = "")
        print("DDSIMCA results:\n")
        print('\033[0m', end = "")

        print(f"- number of components (total): {self.ncomp}")
        print(f"- number of components (selected): {self.ncomp_selected}")
        print(f"- limit type: {self.lim_type}")
        print(f"- alpha: {self.alpha:.3f}")
        print(f"- gamma: {self.gamma:.3f}\n")

        if (self.num_unknowns > 0):
            print( "- class labels: not provided")
            print(f"- number of objects: {self.nrows}")
        else:
            print( "- class labels: provided")
            print(f"- number of objects: {self.nrows}")
            print(f"- number of members: {self.num_members}")
            print(f"- number of strangers: {self.num_strangers}")

        print("")

        colnames = ["PCs", "eCrit", "oCrit", "in", "out"]

        if self.num_members > 0:
            colnames.extend(["TP", "FN", "sens"])
        if self.num_strangers > 0:
            colnames.extend(["TN", "FP", "spec", "sel"])
        if self.num_members > 0 and self.num_strangers > 0:
            colnames.extend(["acc", "eff"])

        out = self.outcomes[colnames].round(3)
        print(out.to_string(index = False))


    def as_df(self, ncomp:int|None = None) -> pd.DataFrame:
        """
        Returns data frame with distance values, decisions and roles for each object.

        Parameters
        ----------
            ncomp : int, optional
                Number of components to use for creating the data frame. If not specified the pre-selected
                optimal number will be used

        """

        ncomp_ind = self.ncomp_selected - 1 if ncomp is None else ncomp - 1

        return pd.DataFrame({
            "class": self.class_labels,
            "decision": ["in" if v else "out" for v in self.D[:, ncomp_ind]],
            "role": [ind_to_roles[int(v)] for v in self.R[:, ncomp_ind]],
            "h": self.H[:, ncomp_ind],
            "q": self.Q[:, ncomp_ind],
            "f": self.F[:, ncomp_ind]
        }, index = self.labels)


    def plotFoM(self, ax:Axes, fom = "sens", color:str|None = None, marker:str = 'o', label:str|None = None,
        show_ci:bool = False):
        """
        Shows plots with selected figure of merit vs number of PCs.

        Parameters
        ----------
            ax : Axes
                Matplotlib's Axis object.
            fom : str, optional
                Name of FoM to display (can be: `"sens"`, `"spec"`, `"eff"`, and `"acc"`).
            color : str, optional
                Color value for the plot.
            marker : str, optional
                Marker symbol for plot points.
            label : str, optional
                Label for the plot (to be used in legend).
            show_ci : bool, optional
                Show or not rectangle with confidence interval for sensitivity (works only with `fom = "sens"`)

        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.
        """

        if not self.has_classes:
            raise ValueError("This results object does not have figures of merit as reference class labels were not provided.")

        if fom not in ["sens", "spec", "sel", "acc", "eff"]:
            raise ValueError("Wrong value for parameter 'fom'.")

        if color is None:
            color = COLORS_FOMS[fom]

        if label is None:
            label = fom

        plot_compstats(ax, np.array(self.outcomes[fom]), color = color, marker = marker, label = label)

        # show confidence interval for sensitivity
        if (fom == "sens") and show_ci:
            n = self.nrows
            p = 1 - self.alpha
            lo = binom.ppf(0.025, n, p) / n
            up = binom.ppf(0.975, n, p) / n
            rect = mpatches.Rectangle((0, lo), self.ncomp + 1, up - lo, facecolor = "#00000020")
            ax.add_patch(rect)

        ax.set_title("Figures of merit")
        ax.set_ylim((0, 1.1))
        ax.set_ylabel('')
        ax.legend()


    def plotDistance(self, ax:Axes, ncomp:int|None = None, distance:str="q",
        colors:dict|None = None, show_labels:bool = False, show_crit:bool = True):
        """
        Plots the specified type of distance for the data points.

        Parameters
        ----------
        ax : Axes
            Matplotlib's Axis object.
        ncomp : int, optional
            Number of PCs to show the plot for (if not specified, pre-selected optimal number will be used).
        do_log : bool, optional
            Whether to plot the original distances or log-transformed (log(1 + u)).
        show_labels : bool, optional
            Logical, show or not object labels on top of each data point.
        show_set : str, optional
            Which set to show ("members", "strangers", or "all"), if not specified, will be autoselected.

        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.

        """

        if distance not in ("q", "h", "f"):
            raise ValueError("Invalid distance type specified. Choose 'q', 'h', or 'f'.")

        if ncomp is not None and ((ncomp < 1) or (ncomp > self.ncomp)):
            raise ValueError(f"Wrong value for parameter 'ncomp' (should be between 1 and {self.ncomp}).")

        ncomp_ind = self.ncomp_selected - 1 if ncomp is None else ncomp - 1


        if self.has_classes:
            nclasses = len(self.classes)
            if colors is None:
                colors = get_group_colors(list(self.classes))
            elif len(colors) < nclasses:
                raise ValueError(f"Colors for each of the {nclasses} must be provided.")

        distances = {
            'q': self.Q[:, ncomp_ind],
            'h': self.H[:, ncomp_ind],
            'f': self.F[:, ncomp_ind]
        }[distance]

        title_map = {'q': "Residual", 'h': "Score", 'f': "Full"}

        x = np.arange(len(distances))
        y = distances
        lbs = self.labels
        dy = np.max(distances) * 0.05

        if self.has_classes and colors is not None:
            g = self.class_labels
            lbs = self.labels
            gu = np.unique(g)
            for i, c in enumerate(gu):
                class_points = [(x[j], y[j], lbs[j]) for j, gl in enumerate(g) if gl == c]
                if not class_points:
                    continue
                cx, cy, cl = zip(*class_points)
                cx = np.asarray(cx)
                cy = np.asarray(cy)
                ax.bar(cx, cy, color = colors[c], label = c)
                if show_labels:
                    cl = np.asarray(cl)
                    plot_labels(ax, cx, cy, cl, dy)
        else:
            ax.bar(x, y, color = MAIN_COLOR)

        if show_labels:
            plot_labels(ax, x, distances, self.labels, dy)

        if show_crit and distance == "f":
            fCrit = self.outcomes["eCrit"][ncomp_ind]
            ax.axhline(fCrit, color="#a0a0a0", linewidth = 0.75, zorder = 1)

        ax.legend()
        ax.set_title(f"{title_map[distance]} distance (A = {ncomp_ind + 1})")
        ax.set_ylabel(f"{distance}-distance")
        ax.set_xlabel("Objects")


    def plotAcceptance(self, ax:Axes, ncomp:int|None=None, do_log:bool=False,
        show_labels:bool = False, show_set:str = ""):
        """
        Plots an acceptance graph showing scaled explained and residual distances and the decision boundary.

        Parameters
        ----------
        ax : Axes
            Matplotlib's Axis object.
        ncomp : int, optional
            Number of PCs to show the plot for (if not specified, pre-selected optimal number will be used).
        do_log : bool, optional
            Whether to plot the original distances or log-transformed (log(1 + u)).
        show_labels : bool, optional
            Logical, show or not object labels on top of each data point.
        show_set : str, optional
            Which set to show ("members", "strangers", or "all"), if not specified, will be autoselected.

        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.

        """

        if show_set == "":
            if self.num_unknowns > 0 or (self.num_members > 0 and self.num_strangers > 0):
                show_set = "all"
            elif self.num_members > 0:
                show_set = "members"
            else:
                show_set = "strangers"

        if show_set not in ["all", "members", "strangers"]:
            raise ValueError("Wrong value for parameter 'show_set', use: 'all', 'members' or 'strangers'")

        if ncomp is not None and ((ncomp < 1) or (ncomp > self.ncomp)):
            raise ValueError(f"Wrong value for parameter 'ncomp' (should be between 1 and {self.ncomp}).")

        ncomp_ind = self.ncomp_selected - 1 if ncomp is None else ncomp - 1

        marker = "s" if self.num_unknowns > 0 else "o"

        h0, Nh = self.hParams
        q0, Nq = self.qParams
        f0, Nf = self.fParams

        roles = [ind_to_roles[int(i)] for i in self.R[:, ncomp_ind]]
        decisions = [ind_to_decisions[int(i)] for i in self.D[:, ncomp_ind]]

        h0 = h0[ncomp_ind]
        q0 = q0[ncomp_ind]
        f0 = f0[ncomp_ind]

        Nh = Nh[ncomp_ind]
        Nq = Nq[ncomp_ind]
        Nf = Nf[ncomp_ind]

        h = self.H[:, ncomp_ind]
        q = self.Q[:, ncomp_ind]

        h_scaled = np.log1p(h / h0) if do_log else h / h0
        q_scaled = np.log1p(q / q0) if do_log else q / q0

        if show_set == "all" or self.num_unknowns > 0:
            x = h_scaled
            y = q_scaled
            g = self.class_labels if self.num_unknowns == 0 else decisions
            lbs = self.labels
            gu = np.unique(g)
            col = get_group_colors(list(self.classes)) if self.has_classes else COLORS_DECISIONS
            show_outliers_boundary = False
        elif show_set == "members" or (self.num_members > 0 and self.num_strangers == 0):
            x = h_scaled[self.ind_members]
            y = q_scaled[self.ind_members]
            g = list(itertools.compress(roles, self.ind_members))
            lbs = list(itertools.compress(self.labels, self.ind_members))
            gu = np.unique(g)
            col = COLORS_ROLES
            show_outliers_boundary = True
        else:
            x = h_scaled[self.ind_strangers]
            y = q_scaled[self.ind_strangers]
            g = list(itertools.compress(roles, self.ind_strangers))
            lbs = list(itertools.compress(self.labels, self.ind_strangers))
            gu = np.unique(g)
            col = COLORS_ROLES
            show_outliers_boundary = False

        dy = (np.max(y) - np.min(y)) * 0.05

        for i, c in enumerate(gu):
            class_points = [(x[j], y[j], lbs[j]) for j, gl in enumerate(g) if gl == c]
            if not class_points:
                continue
            cx, cy, cl = zip(*class_points)
            cx = np.asarray(cx)
            cy = np.asarray(cy)
            cl = np.asarray(cl)
            ax.scatter(cx, cy, label=c, marker=marker, edgecolors=col[c], facecolors='none')
            if show_labels:
                plot_labels(ax, cx, cy, cl, dy)

        # show decision and outliers boundaries
        fCritE = self.outcomes["eCrit"][ncomp_ind]
        xqeMax = fCritE / Nh
        xqe = np.linspace(0, xqeMax, 200)
        yqe = (fCritE - xqe * Nh) / Nq

        fCritO = self.outcomes["oCrit"][ncomp_ind]
        xqoMax = fCritO / Nh
        xqo = np.linspace(0, xqoMax, 200)
        yqo = (fCritO - xqo * Nh) / Nq

        if do_log:
            xqe = np.log1p(xqe)
            yqe = np.log1p(yqe)
            xqo = np.log1p(xqo)
            yqo = np.log1p(yqo)

        ax.plot(xqe, yqe, 'k--', linewidth=0.5)

        if show_outliers_boundary:
            ax.plot(xqo, yqo, 'k:', linewidth=0.5)

        plot_grid(ax)
        ax.legend()
        ax.set_title(f"Acceptance plot (A = {ncomp_ind + 1})")

        if do_log:
            ax.set_xlabel("Explained distance, log(1 + h/h0)")
            ax.set_ylabel("Residual distance, log(1 + q/q0)")
        else:
            ax.set_xlabel("Explained distance, h/h0")
            ax.set_ylabel("Residual distance, q/q0")

        y_max = np.max([np.max(y), np.max(yqo)])
        ax.set_ylim((0, y_max * 1.15))


    def plotScores(self, ax:Axes, comp:tuple = (1,), type = "p", color:str = 'tab:blue',
        marker:str = 'o', show_labels:bool = False, label:str|None = None):
        """
        Shows scores plot.

        Parameters
        ----------
        ax : Axes
            Matplotlib's Axis object.
        comp : tuple, optional
            Number of PCs to show the plot for, tuple with two values for scatter plot and one value for line plot.
        type: str, optional
            Plot type - `"p"` for scatter plot (e.g. PC2 vs PC1), `"l"` for line plot (e.g. PC1 scores vs object index).
        color: str, optional
            Color of the plot elements (markers or lines).
        marker: str, optional
            Marker symbol in case of scatter plot.
        show_labels : bool, optional
            Logical, show or not object labels on top of each data point.
        label : str, optional
            Label for the plot (to be used in legend).

        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.
        """


        if not isinstance(comp, tuple):
            comp = (comp, )
        if (len(comp) < 1) or (len(comp) > 2):
            raise ValueError("Parameter 'comp' should be a tuple with 1 or 2 values.")

        if type == "p":

            if  len(comp) > 1:
                x = self.T[:, comp[0] - 1]
                y = self.T[:, comp[1] - 1]
                xlab = f"PC{comp[0]}"
                ylab = f"PC{comp[1]}"
            else:
                x = np.arange(1, self.nrows + 1, dtype=np.int16)
                y = self.T[:, comp[0] - 1]
                xlab = "Objects"
                ylab = f"PC{comp[0]}"

            ax.plot(x, y,linestyle='None', marker = marker, markeredgecolor = color,
                markerfacecolor = "#ffffff00", label = label)

        elif type == "l":
            if label is None:
                label = f"PC{comp[0]}"
            x = np.arange(1, self.nrows + 1, dtype=np.int16)
            y = self.T[:, comp[0] - 1]
            xlab = "Objects"
            ylab = f"PC{comp[0]}"
            ax.plot(x, y, color = color, label = label)

        elif type == "h":
            if label is None:
                label = f"PC{comp[0]}"
            x = np.arange(1, self.nrows + 1, dtype=np.int16)
            y = self.T[:, comp[0] - 1]
            xlab = "Objects"
            ylab = f"PC{comp[0]}"
            ax.bar(x, y, color = color, label = label)
        else:
            raise ValueError("Wrong value for parameter 'type'.")

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        plot_grid(ax)
        plot_axes(ax, type = type)

        if show_labels:
            dy = (np.max(y) - np.min(y)) * 0.05
            plot_labels(ax, x, y, self.labels, dy)
            ylim = ax.get_ylim()
            ax.set_ylim((ylim[0], ylim[1] + dy * 2))
        ax.set_title("Scores")


    def plotExtremes(self, ax:Axes, ncomp:int|None = None, label:str = "", show_ellipse:bool = True,
        color = "#ffffff00", edgecolors:str = MAIN_COLOR, marker:str = "o"):
        """
        Show extremes plot (number of observed extremes vs expected) for class members.

        Parameters
        ----------
        ax : Axes
            Matplotlib's Axis object.
        ncomp : int, optional
            Number of PCs to show the plot for (if not specified, pre-selected optimal number will be used).
        label : str, optional
            Label for the plot (to be used in legend).
        show_ellipse : bool, optional
            Whether to show the confidence ellipse or not.
        color : str, optional
            Color value for the markers' face.
        edgecolors : str, optional
            Color value for the markers' edge.
        marker : str, optional
            Marker symbol for plot points.


        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.
        """

        if self.num_members < 1:
            raise ValueError("This result object does not have target class members.")

        if ncomp is not None and ((ncomp < 1) or (ncomp > self.ncomp)):
            raise ValueError(f"Wrong value for parameter 'ncomp' (should be between 1 and {self.ncomp}).")

        ncomp_ind = self.ncomp_selected - 1 if ncomp is None else ncomp - 1


        f = self.F[self.ind_members, ncomp_ind]
        n = len(f)

        # remove excluded values if any
        expected = np.arange(1, n + 1)

        # compute and show the tolerance ellipse
        x = np.arange(1, n + 1)
        alpha = x / n
        D = 2 * np.sqrt(x * (1 - alpha))
        Nm = x - D
        Np = x + D

        alpha = expected / n
        Nf = self.fParams[1][ncomp_ind]
        q = 1 - chi2.cdf(f, Nf)
        observed = np.sum(q[:, None] < alpha[None, :], axis=0)


        if show_ellipse:
            line_color = "#00000010"
            ax.plot(x, Nm, color = line_color)
            ax.plot(x, Np, color = line_color)
            ax.plot(x, x, color = line_color)
            plt.vlines(x, Nm, Np, color = line_color)


        ax.scatter(expected, observed, marker = marker, color = color, edgecolors = edgecolors,
            label = label)

        ax.set_xlabel("Number of extremes (expected)")
        ax.set_ylabel("Number of extremes (observed)")
        ax.set_title(f"Extremes (A = {ncomp_ind + 1})")
        plot_grid(ax)


    def plotAliens(self, ax:Axes, ncomp:int|None = None, label:str = "", show_ellipse:bool = True,
        color = "#ffffff00", edgecolors:str = MAIN_COLOR, marker:str = "o"):
        """
        Show aliens plot (number of observed extremes vs expected) for class strangers.

        Parameters
        ----------
        ax : Axes
            Matplotlib's Axis object.
        ncomp : int, optional
            Number of PCs to show the plot for (if not specified, pre-selected optimal number will be used).
        label : str, optional
            Label for the plot (to be used in legend).
        show_ellipse : bool, optional
            Whether to show the confidence ellipse or not.
        color : str, optional
            Color value for the markers' face.
        edgecolors : str, optional
            Color value for the markers' edge.
        marker : str, optional
            Marker symbol for plot points.


        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.
        """

        if self.num_strangers < 1:
            raise ValueError("This result object does not have objects from non-target classes.")

        if ncomp is not None and ((ncomp < 1) or (ncomp > self.ncomp)):
            raise ValueError(f"Wrong value for parameter 'ncomp' (should be between 1 and {self.ncomp}).")

        ncomp_ind = self.ncomp_selected - 1 if ncomp is None else ncomp - 1


        roles = self.R[:, ncomp_ind]
        f = self.F[(self.ind_strangers) & (roles == 3), ncomp_ind]
        n = len(f)

        # remove excluded values if any
        expected = np.arange(1, n + 1)

        # compute and show the tolerance ellipse
        x = np.arange(1, n + 1)
        beta = x / n
        D = 2 * np.sqrt(x * (1 - beta))
        Nm = x - D
        Np = x + D

        m  =  self.outcomes["m"][ncomp_ind]
        Mz = self.outcomes["Mz"][ncomp_ind]
        hz = self.outcomes["hz"][ncomp_ind]
        Sz = self.outcomes["Sz"][ncomp_ind]
        beta = expected / n
        zb = norm.ppf(beta)

        eCrit = m * np.pow(Sz * zb + Mz, 1.0 / hz)
        observed = np.sum(f[:, None] < eCrit[None, :], axis=0)


        if show_ellipse:
            line_color = "#00000010"
            ax.plot(x, Nm, color = line_color)
            ax.plot(x, Np, color = line_color)
            ax.plot(x, x, color = line_color)
            plt.vlines(x, Nm, Np, color = line_color)


        ax.scatter(expected, observed, marker = marker, color = color, edgecolors = edgecolors,
            label = label)

        ax.set_xlabel("Number of aliens (expected)")
        ax.set_ylabel("Number of aliens (observed)")
        ax.set_title(f"Aliens (A = {ncomp_ind + 1})")
        plot_grid(ax)


    def plotSelectivity(self, ax:Axes, ncomp:int|None = None, color:str = MAIN_COLOR, label:str = ""):
        """
        Show plot where selectivity values are plotted agains expected sensitivity.

        Parameters
        ----------
        ax : Axes
            Matplotlib's Axis object.
        ncomp : int, optional
            Number of PCs to show the plot for (if not specified, pre-selected optimal number will be used).
        label : str, optional
            Label for the plot (to be used in legend).
        color : str, optional
            Color value for the markers' face.


        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.
        """

        if self.num_strangers < 1:
            raise ValueError("This result object does not have objects from non-target classes.")

        if ncomp is not None and ((ncomp < 1) or (ncomp > self.ncomp)):
            raise ValueError(f"Wrong value for parameter 'ncomp' (should be between 1 and {self.ncomp}).")

        ncomp_ind = self.ncomp_selected - 1 if ncomp is None else ncomp - 1


        s = self.outcomes["s"][ncomp_ind]
        f0t = self.outcomes["f0t"][ncomp_ind]
        k = self.outcomes["k"][ncomp_ind]
        Mz = self.outcomes["Mz"][ncomp_ind]
        hz = self.outcomes["hz"][ncomp_ind]
        Sz = self.outcomes["Sz"][ncomp_ind]

        norm1 = 1. / (k + s)
        norm2 = 1. / Sz

        alpha = np.arange(0, 1, 0.001)
        fcrit = chi2.ppf(1 - alpha, k)
        z = fcrit / f0t
        beta = norm.cdf((np.pow(z * norm1, hz) - Mz) * norm2)
        sel = np.sum(beta * 0.001)
        auc = 1 - sel

        plot_grid(ax)
        ax.plot(beta, 1 - alpha, label = label, color = color)
        ax.set_xlabel("1 - selectivity, β")
        ax.set_ylabel("Expected sensitivity, 1 - ɑ")
        ax.set_title(f"Selectivity (A = {ncomp_ind + 1}, AUC = {auc:.4f})")


####################################
# DDSIMCA class                    #
####################################

class DDSIMCA:
    """
    A Data Driven SIMCA model class.

    Class methods
    -------------
    `predict()`:
        Applies DDSIMCA model to a dataset.
    `select_ncomp()`:
        Set (selects) optimal number of components.
    `summary()`:
        Shows summary information about the model.
    `plotLoadings()`:
        Shows loadings plot.
    `plotDoF()`:
        Show plot with degrees of freedom vs PCs.
    `plotEigenvals()`:
        Shows plot with eigenvalues vs PCs.
    """

    def __init__(self, target_class:str):
        """
        Creates an instance of DDD-SIMCA model class.

        Parameters
        ----------
        target_class : str
            The name of the target class the model should be trained for.
        """
        # set main model parameters
        self.target_class = target_class
        self.status = "init"


    def get_distances(self, X:Array2D) -> tuple[Array2D, Array2D, Array2D, Array2D]:
        """
        Project values from X to PC space and computes score, orthogonal and full distances
        for each number of components.

        Parameters
        ----------
        X : NDArray
            Matrix (2D array) with data values.

        Returns
        -------
        tuple with four matrices (2D arrays, nrows x ncomp)

            H : NDArray
                score distances.
            Q : NDArray
                orthogonal distances.
            T : NDArray
                scores.
            E : NDArray
                Residuals.
        """

        X = (X - self.center_values) / (self.scale_values + EPSILON)
        T = X @ self.V

        H = np.zeros((X.shape[0], self.ncomp))
        Q = np.zeros((X.shape[0], self.ncomp))

        # a = 1
        X_hat = T[:, :1] @ self.V[:, :1].T
        E = X - X_hat
        H[:, :1] = T[:, :1] * T[:, :1] / self.eigenvals[0]
        Q[:, 0] = (E ** 2).sum(axis=1)

        # a > 1
        for a in range(2, self.ncomp+1):
            X_hat = T[:, :a] @ self.V[:, :a].T
            E = X - X_hat
            H[:, a - 1] = H[:, a - 2] + T[:, a - 1] * T[:, a - 1] / self.eigenvals[a - 1]
            Q[:, a - 1] = (E ** 2).sum(axis=1)

        return (H, Q, T, E)


    def train(self, data:pd.DataFrame, ncomp:int, center:bool = True, scale:bool = False):
        """
        Train DDSIMCA model.

        Parameters
        ----------
        data : pd.DataFrame
            Data frame with training set, first column should contain target class label.
        ncomp : int
            Number of components to compute (optimal number can be selected later).
        center : bool
            Mean center or not data variables.
        scale: bool
            Standardize or not data variables.

        Raises
        ------
            ValueError:
                If data frame has wrong dimension, no column with class labels or wrong values for this column.

        """

        class_labels = data.iloc[:, 0]
        classes = sorted(class_labels.unique())
        if len(classes) != 1 or classes[0] != self.target_class:
            raise ValueError(f"First column of data frame must content target class name ('{self.target_class}').")

        X = data.iloc[:, 1:].values.astype(np.float64)
        nrows = X.shape[0]
        ncols = X.shape[1]

        if ncomp < 1 or ncomp > ncols or ncomp > nrows - 1:
            raise ValueError(f"Dataset size {nrows}x{ncols} does not match the number of components ({self.ncomp}).")

        self.ncomp = ncomp
        self.center_values = X.mean(axis=0) if center else np.zeros(X.shape[1])
        self.scale_values = X.std(axis=0, ddof = 1.) if scale else np.ones(X.shape[1]) - EPSILON

        self.varlabels = data.columns[1:]

        varvalues = pd.to_numeric(list(self.varlabels), errors="coerce")
        self.varvalues = varvalues if not np.isnan(varvalues).any() else np.arange(1, ncols + 1)

        _, s, V = np.linalg.svd((X - self.center_values) / (self.scale_values + EPSILON), full_matrices=False)
        self.V = np.transpose(V)[:, :ncomp]
        self.eigenvals  = s[:ncomp]**2 / (nrows - 1)

        H, Q, _, _ = self.get_distances(X)

        h0c, Nhc = get_distparams(H, type = "classic")
        q0c, Nqc = get_distparams(Q, type = "classic")
        h0r, Nhr = get_distparams(H, type = "robust")
        q0r, Nqr = get_distparams(Q, type = "robust")

        Nfc = Nhc + Nqc
        Nfr = Nhr + Nqr
        f0c = Nfc
        f0r = Nfr

        self.center = center
        self.scale = scale
        self.ncomp_selected = self.ncomp

        self.nrows = nrows
        self.ncols = ncols
        self.hParams = {"classic": (h0c, Nhc), "robust": (h0r, Nhr)}
        self.qParams = {"classic": (q0c, Nqc), "robust": (q0r, Nqr)}
        self.fParams = {"classic": (f0c, Nfc), "robust": (f0r, Nfr)}
        self.status = "trained"


    def predict(self, data:pd.DataFrame, lim_type:str = "classic", alpha:float=0.05, gamma:float=0.01) -> DDSIMCARes:
        """
        Apply DDSIMCA model to a new dataset.

        Parameters
        ----------
        data : pd.DataFrame
            Data frame with data values (can be with or without class labels).
        lim_type : str, optional
            Type of estimator to use for computing critical limits ("classic", or "robust")
        alpha : float, optional
            Significance level to define expected sensitivity of the model.
        gamma : float, optional
            Significance level for detection of outliers.

        Returns
        -------
        DDSIMCARes
            A DDSIMCARes object containing the predictions and statistical analysis.
        """

        if alpha < 0.00001 or alpha > 0.999999:
            raise ValueError("Wrong value for parameter 'alpha' (must be between 0.00001 and 0.999999).")

        if data.shape[1] < self.ncols or data.shape[1] > self.ncols + 1:
            raise ValueError(f"Wrong number of columns in the data frame (expected to be {self.ncols + 1} if first column contains class labels or {self.ncols} if not).")

        has_classes = data.shape[1] == self.ncols + 1

        if has_classes:
            classes = data.iloc[:, 0]
            X = data.iloc[:, 1:].values.astype(np.float64)
        else:
            classes = None
            X = data.values.astype(np.float64)

        labels = list(data.index)
        H, Q, T, E = self.get_distances(X)


        return DDSIMCARes(
            self.target_class,
            self.hParams[lim_type],
            self.qParams[lim_type],
            self.fParams[lim_type],
            self.center,
            self.scale,
            alpha,
            gamma,
            lim_type,
            self.nrows,
            H, Q, T, E,
            classes,
            labels,
            self.ncomp_selected
        )

    def select_ncomp(self, ncomp:int):
        """
        Change (select) optimal number of components

        Parameters
        ----------
            ncomp : int
                Number of components to set.
        """
        if ncomp < 1 or ncomp > self.ncomp:
            raise ValueError(f"Wrong value for 'ncomp' parameter (must be between 1 and {self.ncomp}")
        self.ncomp_selected = ncomp


    def summary(self, lim_type:str = "classic"):
        """
        Show summary information for given type of limits estimator ('classic' or 'robust').
        """

        if (lim_type != 'classic' and lim_type != 'robust'):
            raise ValueError("Wrong value for 'lim_type' parameter (must be either 'classic' or 'robust'")

        print('\033[1m', end = "")
        print("DDSIMCA model:\n")
        print('\033[0m', end = "")

        if self.status == "init":
            print("- model has not been trained yet.")
            return

        scaling_ind = int(self.center) * 2 + int(self.scale)
        scaling_str = ["none", "standardization", "mean centering", "autoscaling (mean centering + standardization)"]

        print(f"- target class: {self.target_class}")
        print(f"- number of components (total): {self.ncomp}")
        print(f"- number of components (optimal): {self.ncomp_selected}")
        print(f"- number of training samples: {self.nrows}")
        print(f"- number of variables: {self.ncols}")
        print(f"- preprocessing: {scaling_str[scaling_ind]}")

        print(f"\nParameters for {lim_type} estimators:\n")

        print('\033[4m', end = "")
        print("PCs    Nh    Nq   eigenvals")
        print('\033[0m', end = "")

        _, Nq = self.qParams[lim_type]
        _, Nh = self.hParams[lim_type]
        for a in range(self.ncomp):
            if a == self.ncomp_selected - 1:
                print('\033[1m', end = "")
            print(f"{(a + 1):3d} {Nh[a]:5.0f} {Nq[a]:5.0f} {self.eigenvals[a]:11.3f}")
            if a == self.ncomp_selected - 1:
                print('\033[0m', end = "")


    def plotLoadings(self, ax:Axes, comp:tuple = (1,), type = "p", color:str = 'tab:blue',
        marker:str = 'o', show_labels:bool = False):
        """
        Shows loadings plot.

        Parameters
        ----------
        ax : Axes
            Matplotlib's Axis object.
        comp : tuple, optional
            Number of PCs to show the plot for, tuple with two values for scatter plot and one value for line plot.
        type: str, optional
            Plot type - `"p"` for scatter plot (e.g. PC2 vs PC1), `"l"` for line plot (e.g. PC1 scores vs object index).
        color: str, optional
            Color of the plot elements (markers or lines).
        marker: str, optional
            Marker symbol in case of scatter plot.
        show_labels : bool, optional
            Logical, show or not object labels on top of each data point.

        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.
        """


        if not isinstance(comp, tuple):
            comp = (comp, )
        if (len(comp) < 1) or (len(comp) > 2):
            raise ValueError("Parameter 'comp' should be a tuple with 1 or 2 values.")

        if type == "p":

            if  len(comp) > 1:
                x = self.V[:, comp[0] - 1]
                y = self.V[:, comp[1] - 1]
                xlab = f"PC{comp[0]}"
                ylab = f"PC{comp[1]}"
            else:
                x = self.varvalues
                y = self.V[:, comp[0] - 1]
                xlab = "Variables"
                ylab = f"PC{comp[0]}"

            ax.grid(True, zorder=-1, linestyle = "--", color = "#e0e0e0")
            ax.plot(x, y,linestyle='None', marker = marker, markeredgecolor = color, markerfacecolor = "#ffffff00")

        elif type == "l":
            x = self.varvalues
            y = self.V[:, comp[0] - 1]
            xlab = "Variables"
            ylab = f"PC{comp[0]}"
            ax.plot(x, y, color = color, label = f"PC{comp[0]}")

        elif type == "h":
            x = np.arange(1, self.ncols + 1, dtype=np.int16)
            y = self.V[:, comp[0] - 1]
            xlab = "Variables"
            ylab = f"PC{comp[0]}"
            ax.bar(x, y, color = color, label = f"PC{comp[0]}")
        else:
            raise ValueError("Wrong value for parameter 'type'")

        ax.set_xlabel(xlab)
        ax.set_ylabel(ylab)

        plot_grid(ax)
        plot_axes(ax, type = type)

        if show_labels:
            plot_labels(ax, x, y, np.asarray(self.varlabels), 0.01)

        ax.set_title("Loadings")


    def plotDoF(self, ax, dof:str="Nh", lim_type:str="classic", color:str|None = None, marker:str = 'o',
        label:str|None = None):
        """
        Shows plot with degrees of freedom (Nh, Nq or Nf) vs number of components.

        ax : Axes
            Matplotlib's Axis object.
        lim_type : tuple, optional
            Which estimator ("classic" or "robust") to use.
        color: str, optional
            Color of the plot elements (markers or lines).
        marker: str, optional
            Marker symbol in case of scatter plot.
        label : str, optional
            Label for the plot series (to be used for legend).

        Raises
        ------
        ValueError
            If any parameter does not meet the requirements.
        """

        colors = {"Nh": "tab:purple", "Nq": "tab:blue", "Nf": "tab:cyan"}

        if dof not in ["Nh", "Nq", "Nf"]:
            raise ValueError("Wrong value for parameter 'dof'.")

        if color is None:
            color = colors[dof]

        if label is None:
            label = dof

        if dof == "Nh":
            y = self.hParams[lim_type][1]
        elif dof == "Nq":
            y = self.qParams[lim_type][1]
        else:
            y = self.fParams[lim_type][1]

        plot_compstats(ax, y, color = color, marker = marker, label = label)
        ax.set_title("Degrees of freedom")
        ax.set_ylabel('')
        ax.legend()


    def plotEigenvals(self, ax, do_log:bool = False, color:str = 'tab:blue', marker:str = 'o'):
        """
        Shows plot with eigenvalus vs number of components.

        Shows plot with degrees of freedom (Nh, Nq or Nf) vs number of components.

        ax : Axes
            Matplotlib's Axis object.
        do_log : bool, optional
            Apply log transformation or not.
        color: str, optional
            Color of the plot elements (markers or lines).
        marker: str, optional
            Marker symbol in case of scatter plot.
        """

        if do_log:
            y = np.log10(self.eigenvals)
            ylab = "log10(λ)"
        else:
            y = self.eigenvals
            ylab = "λ"

        plot_compstats(ax, y, color = color, marker = marker, label = '')
        ax.set_title("Eigenvalues")
        ax.set_ylabel(ylab)



def ddsimca(data:pd.DataFrame, ncomp:int, center:bool=True, scale:bool=False):
    """
    Train Data Driven SIMCA model

    Parameters
    ----------
    data : pd.DataFrame
        Data frame with training set, first column should contain target class label.
    ncomp : int
        Number of components to compute (optimal number can be selected later).
    center : bool, optional
        Logical, mean center or not data variables.
    scale : bool, optional
        Logical, standardize or not data variables.

    The model will compute distance parameters based on the training set for both classic and
    robust estimators. You can select which estimator to use later, when apply model to a new
    dataset via method `predict()`.
    """

    class_labels = data.iloc[:, 0]
    classes = sorted(class_labels.unique())
    if len(classes) != 1:
        raise ValueError("First column of data frame must content only target class name")

    target_class = classes[0]
    m = DDSIMCA(target_class)
    m.train(data, ncomp, center = center, scale = scale)
    return m
