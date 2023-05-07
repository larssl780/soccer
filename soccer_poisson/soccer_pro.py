import numpy as np
from scipy import optimize
from scipy.stats import skellam
import pandas as pd
from thin_wrappers import grid_runner as gr
from thin_wrappers.utils import find_all_indicies
import re
# import pdb


def make_pretty(styler, cap=''):
    styler.set_caption(cap)
    # styler.format(precision=3)
    return styler


def odds_ratio_func(odds_array, c):
    """
    odds_array contains the odds with margin on top
    c is odds_ratio
    """
    assert len(odds_array) == 3, "Only handles 3 elements!"
    x = 1. / odds_array[0]
    y = 1. / odds_array[1]
    z = 1. / odds_array[2]
    return 1 - x / (c + x - c * x) - y / (c + y - c * y) - z / (c + z - c * z)


def odds_ratio_solver(odds_array):
    return optimize.brentq(lambda c: odds_ratio_func(odds_array, c), 1, 20)


def fair_odds_from_odds_ratio(odds_w_margin, odds_ratio):
    prob_w_margin = 1. / odds_w_margin
    return 1. / (prob_w_margin / (odds_ratio + prob_w_margin - odds_ratio * prob_w_margin))


def log_expo_func(odds_array, n):
    sumz = 0
    for o in odds_array:
        sumz += pow(1. / o, 1. / n)
    return 1 - sumz


def log_expo_solver(odds_array):
    return optimize.brentq(lambda n: log_expo_func(odds_array, n), 0.1, 0.99)


def fair_odds_from_log_expo(odds_w_margin, expo):
    prob_w_margin = 1. / odds_w_margin
    prob_wo_margin = pow(prob_w_margin, 1. / expo)
    return 1. / prob_wo_margin


def mov_probs_one_arg2(odds_and_potentially_extra_arg):
    home_odds, *rest = odds_and_potentially_extra_arg
    draw_odds = rest[0]
    away_odds = rest[1]
    transform_to_norm = True
    if len(rest) > 2:
        transform_to_norm = rest[2]

    probs = mov_probs_from_poission_reg_np(
        home_odds, draw_odds, away_odds, transform_to_norm=transform_to_norm)
    return probs


def mov_probs_skellam(home_odds, draw_odds, away_odds):
    # [h, d, a] = remove_overround([home_odds, draw_odds, away_odds])
    win = 1 / home_odds
    draw = 1 / draw_odds
    lose = 1 / away_odds
    # remove the overround:
    k = optimize.brentq(lambda x: win**x + draw**x + lose**x - 1, 1, 2)
    # p_win = win**k
    p_draw = draw**k
    p_lose = lose**k

    def f(p):
        return (skellam.pmf(0, p[0], p[1]) - p_draw)**2 + (skellam.cdf(-1, p[0], p[1]) - p_lose)**2

    def c_1(p):
        return p[0]

    def c_2(p):
        return p[1]

    solution = optimize.minimize(f, np.array([2, 2]), constraints=(
        {'type': 'ineq', 'fun': c_1}, {'type': 'ineq', 'fun': c_2}))

    (mu1, mu2) = solution.x

    clean_probs = {}

    mov_probs = dict(
        zip(np.arange(-20, 20), skellam.pmf(np.arange(-20, 20), mu1, mu2)))
    for mov, prob in mov_probs.items():
        if mov < -5:
            if -np.inf in clean_probs:
                clean_probs[-np.inf] += prob
            else:
                clean_probs[-np.inf] = prob
        elif mov > 5:
            if np.inf in clean_probs:
                clean_probs[np.inf] += prob
            else:
                clean_probs[np.inf] = prob
        else:
            if mov in clean_probs:
                clean_probs[mov] += prob
            else:
                clean_probs[mov] = prob

    return clean_probs, mu1, mu2


def mov_probs_one_arg3(odds_and_potentially_extra_arg):
    home_odds, *rest = odds_and_potentially_extra_arg
    draw_odds = rest[0]
    away_odds = rest[1]
    transform_to_norm = True
    if len(rest) > 2:
        transform_to_norm = rest[2]

    probs = mov_probs_from_poission_reg_rmse(
        home_odds, draw_odds, away_odds, transform_to_norm=transform_to_norm)
    return probs


def mov_probs_from_poission_reg_np(home_odds, draw_odds, away_odds, transform_to_norm=True):

    [h, d, a] = remove_overround([home_odds, draw_odds, away_odds])
    res = poisson_loss_function(1 / h, 1 / d, 1 / a)
    lambda_1, lambda_2, lambda_1_2 = res.x
    # print('lambda_1 = %f' % lambda_1)
    # print('lambda_2 = %f' % lambda_2)
    poisson_1, poisson_2 = double_poisson_regression_np(
        lambda_1, lambda_2, lambda_1_2)
    # poisson_1, poisson_2 = res.x
    poisson_matrix = to_poisson_matrix_np(poisson_1, poisson_2)
    mov_probs = {}
    for home_goals in range(poisson_matrix.shape[0]):
        for away_goals in range(poisson_matrix.shape[1]):
            mov = home_goals - away_goals
            if mov not in mov_probs:
                mov_probs[mov] = poisson_matrix[home_goals, away_goals].item()
            else:
                mov_probs[mov] += poisson_matrix[home_goals, away_goals].item()

    if transform_to_norm:
        clean_probs = {}
        for mov, prob in mov_probs.items():
            if mov < -5:
                if -np.inf in clean_probs:
                    clean_probs[-np.inf] += prob
                else:
                    clean_probs[-np.inf] = prob
            elif mov > 5:
                if np.inf in clean_probs:
                    clean_probs[np.inf] += prob
                else:
                    clean_probs[np.inf] = prob
            else:
                if mov in clean_probs:
                    clean_probs[mov] += prob
                else:
                    clean_probs[mov] = prob

        return clean_probs

    return mov_probs


def mov_probs_from_poission_reg_rmse(home_odds, draw_odds, away_odds, transform_to_norm=True):

    [h, d, a] = remove_overround([home_odds, draw_odds, away_odds])
    res = poisson_loss_function_rmse(1 / h, 1 / d, 1 / a)
    lambda_1, lambda_2, lambda_1_2 = res.x
    # print('lambda_1 = %f' % lambda_1)
    # print('lambda_2 = %f' % lambda_2)
    poisson_1, poisson_2 = double_poisson_regression_np(
        lambda_1, lambda_2, lambda_1_2)
    # poisson_1, poisson_2 = res.x
    poisson_matrix = to_poisson_matrix_np(poisson_1, poisson_2)
    mov_probs = {}
    for home_goals in range(poisson_matrix.shape[0]):
        for away_goals in range(poisson_matrix.shape[1]):
            mov = home_goals - away_goals
            if mov not in mov_probs:
                mov_probs[mov] = poisson_matrix[home_goals, away_goals].item()
            else:
                mov_probs[mov] += poisson_matrix[home_goals, away_goals].item()

    if transform_to_norm:
        clean_probs = {}
        for mov, prob in mov_probs.items():
            if mov < -5:
                if -np.inf in clean_probs:
                    clean_probs[-np.inf] += prob
                else:
                    clean_probs[-np.inf] = prob
            elif mov > 5:
                if np.inf in clean_probs:
                    clean_probs[np.inf] += prob
                else:
                    clean_probs[np.inf] = prob
            else:
                if mov in clean_probs:
                    clean_probs[mov] += prob
                else:
                    clean_probs[mov] = prob

        return clean_probs

    return mov_probs


def poisson_loss_function(p_normal_time_win, p_normal_time_draw, p_normal_time_loss):
    """
    50x faster than the gradient bull-shit...
    """
    def get_loss_for_single_prediction(lambdas):
        lambda_1, lambda_2, lambda_1_2 = lambdas
        return cross_entropy_loss_np(single_prediction_np(lambda_1, lambda_2, lambda_1_2), (p_normal_time_win, p_normal_time_draw, p_normal_time_loss))

    return optimize.minimize(get_loss_for_single_prediction, (1, 1, 0.5))


def cross_entropy_loss_np(prediction, target):
    return -np.mean(np.stack(target, axis=-1) * np.log(np.stack(prediction, axis=-1)), axis=-1)


def double_poisson_regression_np(country_1, country_2, country_1_2):
    goals = np.arange(20)
    factorials = np.array([np.math.factorial(goal) for goal in goals])
    lambda_1 = np.expand_dims(country_1 + country_1_2, axis=-1)
    lambda_2 = np.expand_dims(country_2 - country_1_2, axis=-1)

    poisson_1 = lambda_1**goals * np.exp(-lambda_1) / factorials
    poisson_2 = lambda_2**goals * np.exp(-lambda_2) / factorials
    return poisson_1, poisson_2


def to_poisson_matrix_np(poisson_1, poisson_2):
    poisson_matrix = np.dot(np.expand_dims(
        poisson_1, axis=1), np.expand_dims(poisson_2, axis=0))
    poisson_matrix /= np.sum(poisson_matrix)
    return poisson_matrix


def single_prediction_np(lambda_1, lambda_2, lambda_1_2):
    poisson_1, poisson_2 = double_poisson_regression_np(
        lambda_1, lambda_2, lambda_1_2)
    poisson_matrix = to_poisson_matrix_np(poisson_1, poisson_2)
    p_normal_time_win, p_normal_time_draw, p_normal_time_loss = p_normal_time(
        poisson_matrix)
    return p_normal_time_win, p_normal_time_draw, p_normal_time_loss


def p_normal_time(poisson_matrix):
    ones_like_matrix = np.ones(shape=(20, 20))
    win_mask = np.tril(ones_like_matrix, k=-1)
    draw_mask = np.eye(ones_like_matrix.shape[0])
    loss_mask = np.triu(ones_like_matrix, k=1)
    p_normal_time_win = np.sum(poisson_matrix * win_mask)
    p_normal_time_draw = np.sum(poisson_matrix * draw_mask)
    p_normal_time_loss = np.sum(poisson_matrix * loss_mask)
    return p_normal_time_win, p_normal_time_draw, p_normal_time_loss


def poisson_loss_function_rmse(p_normal_time_win, p_normal_time_draw, p_normal_time_loss):
    def get_loss_for_single_prediction(lambdas):
        lambda_1, lambda_2, lambda_1_2 = lambdas
        return rmse_loss(single_prediction_np(lambda_1, lambda_2, lambda_1_2), (p_normal_time_win, p_normal_time_draw, p_normal_time_loss))

    return optimize.minimize(get_loss_for_single_prediction, (1, 1, 0.5))


def rmse_loss(prediction, target):
    return rmse(np.array(prediction), np.array(target))


def dict_prob_mov_less(dic, mov):
    valbro = np.array(list(dic.values()))
    return valbro.T[np.less(list(dic.keys()), mov)].sum()


def dict_prob_mov_greater_equal(dic, mov):
    valbro = np.array(list(dic.values()))
    return valbro.T[np.greater_equal(list(dic.keys()), mov)].sum()


def asian_quarter_handicaps():
    return np.arange(-7.75, 7.75 + 1e-4, 0.5)
    # [-2.75, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75]:


def asian_non_quarter_handicaps():
    return np.arange(-8.5, 7.5 + 1e-4, 0.5)


def asian_handicaps():
    raw = list(asian_quarter_handicaps()) + list(asian_non_quarter_handicaps())
    raw.sort()
    return raw


def _offsetting_odds(odds=None):
    """
    P(Home team win by at least 1)
    P(away team lose by at most 1) = p(home team win by at most 1) = 1- P(Home team win by at least 1) => by def have to sum to 1!

    P(home-1)
    we need to show that the following holds:
    p(away+1) = 1 - p(home-1)
    p(away+1) = P(away team lose by at most 1)
    P(away team lose by at most 1) = p(home team win by at most 1)
    p(home team win by at most 1) = 1- P(Home team win by at least 1)
    ergo, p(away+1) = 1 - p(home-1)

    Eq. we have odds for home +0.5, we can then work out odds for away -0.5
    P(away-0.5) = 1 - p(home+0.5)
    p(home+0.5) = 1/odds(home+0.5)
    P(away-0.5) = 1 - 1/odds(home+0.5)
    odds(away-0.5) = 1/(1 - 1/odds(home+0.5))

    We use this as a short cut in the code, but it's also useful for parsing oddsportal data, which only lists Asian handicaps for
    the home team... (but there we have it written out).

    So say we have listed Home +0.5 as betfair 1 = 1.61, 2 = 2.46, the way to read that is then that 2 means away -0.5!

    This assumes there's zero overround, does that matter for our calcs?
    """
    return 1.0 / (1 - 1.0 / odds)


def dict_prob_mov_equal(dic, mov):
    valbro = np.array(list(dic.values()))
    return valbro.T[np.equal(list(dic.keys()), mov)].sum()


def dict_prob_mov_greater(dic, mov):
    valbro = np.array(list(dic.values()))
    return valbro.T[np.greater(list(dic.keys()), mov)].sum()


def clubelo_expected_mov(probs):
    vals = range(-20, 20)
    ev = 0
    for v in vals:
        ev += v * mov_pmf(v, probs)
    return ev


def mov_pmf(x, probs):
    # found = False
    # while not found:
    # bins = list(probs.keys())
    if x in probs:
        return probs[x]
    elif x < -5:
        return probs[-np.inf]
    elif x > 5:
        return probs[np.inf]


def calculate_fair_odds_and_loss_prob(probs, hc):

    try:
        lp = loss_calc(hc=hc, probs=probs)
    except:
        lp = np.nan
    try:
        fo = fair_asian_odds_clubelo(probs, hc)
    except:
        # pdb.set_trace()
        fo = np.nan

    return hc, fo, lp


def calculate_betfair_odds_and_loss_prob(hc, probs=None, commission=0.02):

    hc, fo, lp = calculate_fair_odds_and_loss_prob(probs, hc)

    return betfair_equivalent_odds(fo, commission=commission), lp


def fair_asian_odds_clubelo(mov_probs, home_team_handicap):
    """
    return the odds for which the epnl is 0
    """

    return optimize.brentq(lambda odds: asian_expected_pnl_clubelo(home_team_handicap, odds, mov_probs), 1, 15)


def betfair_net_odds(nom_odds=None, commission=0.02):
    return (nom_odds - 1) * (1 - commission) + 1


def betfair_equivalent_odds(net_odds=None, commission=0.02):
    """Given some theoretical fair odds - what's the odds that we have to see on betfair in order to make money?
    Ie. if fair odds is fo, then bf equivalent odds would be fo *(1+eps), eps>0


    """
    return (net_odds - 1) / (1 - commission) + 1


def loss_calc(home='', dt='', hc=None, probs=None, movs=None, verbose=False):
    if probs is None:
        raise Exception("You have to provide probabilities!")

    add_inf = True
    if movs is None:
        movs = np.arange(-5, 6)
    else:
        add_inf = False

    # for hc -0.5, if mov is less than 1, we've lost
    nqh = asian_non_quarter_handicaps()
    qh = asian_quarter_handicaps()
    if hc == -0.5:
        cutoff = 1
    elif hc == -.75:
        cutoff = 1
    elif hc in [0.5, 0.75]:
        cutoff = 0
    elif hc in nqh:
        if hc < 0:
            if int(hc) == hc:
                # eg. -2, -3 etc
                cutoff = -hc
            else:
                # eg -2.5 , -1.5 etc
                cutoff = -hc + 0.5
        else:
            if int(hc) == hc:
                # eg +2, +3 etc
                cutoff = -hc
            else:
                # eg +2.5
                cutoff = -hc + 0.5
    elif hc in qh:
        # pdb.set_trace()
        if hc < 0:
            cutoff = np.ceil(-hc)
        else:
            cutoff = -np.floor(hc)
            # if int(hc) == hc:
            # print('hej')
            # else:

    # pdb.set_trace()
    try:
        loss_movs = list(movs[np.less(movs, cutoff)])
    except:
        raise Exception("Failed to work out loss MoVs!")

    if add_inf:
        loss_movs.append(-np.inf)

    if verbose:
        print("We will lose money for these MoVs:")
        print(loss_movs)
    prob = 0

    for el in loss_movs:
        prob += probs[el]
    return prob


def rmse(targets, predictions):
    return np.sqrt(((targets - predictions)**2).mean())


def asian_expected_pnl_clubelo(hc, odds, mov_probs):
    """
    mov_probs = {-np.inf:xx, -5:xx,.... 5:xx, np.inf: xx }
    """

    # if hc in [-2.75, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2, 2.25]:
    if hc in asian_quarter_handicaps():
        lower = hc - 0.25
        higher = hc + 0.25
        # pdb.set_trace()
        if int(lower) == lower:
            low_even = True
        else:
            low_even = False
        if hc < 0:
            if low_even:
                return -1 * dict_prob_mov_less(mov_probs, -lower) + 0.5 * (odds - 1) * dict_prob_mov_equal(mov_probs, -lower) + (odds - 1) * dict_prob_mov_greater(mov_probs, -lower)
            else:
                return -1 * dict_prob_mov_less(mov_probs, -higher) - 0.5 * dict_prob_mov_equal(mov_probs, -higher) + (odds - 1) * dict_prob_mov_greater_equal(mov_probs, (-lower + 0.5))
        else:
            if low_even:
                return -1 * dict_prob_mov_less(mov_probs, (-higher + 0.5)) + 0.5 * (odds - 1) * dict_prob_mov_equal(mov_probs, -lower) + (odds - 1) * dict_prob_mov_greater(mov_probs, -lower)
            else:
                return -1 * dict_prob_mov_less(mov_probs, -higher) - 0.5 * dict_prob_mov_equal(mov_probs, -higher) + (odds - 1) * dict_prob_mov_greater(mov_probs, -higher)
                # p_make_money = clf.prob_mov_greater(elo_arg, -higher)

    elif hc in asian_non_quarter_handicaps():
        return -1 * dict_prob_mov_less(mov_probs, -hc) + (odds - 1) * dict_prob_mov_greater(mov_probs, -hc)

    else:
        raise NotImplementedError("Handicap = %2.2f not handled yet" % hc)


def remove_overround(odds, verbose=False, kind='equal'):
    """
    kind in {equal, proportional, odds_ratio, log}

    equal assumes equal margin across bets:


    odds_r = (1-marg)/p_r , r in {H, D, A}

    marg = 1 - 1/sum_r(o_r^-1)
    """

    rv = 0
    for odd in odds:
        if isinstance(odd, str):
            raise TypeError("odds should be float, not str!")
        rv += 1.0 / odd
    if verbose:
        print('Overround/margin is %f' % rv)
    if kind == 'equal':

        return np.array(odds) * rv
    elif kind == 'proportional':
        fair_odds = []

        for odd in odds:
            # pdb.set_trace()
            fair_odds.append(len(odds) * odd / (len(odds) - (rv - 1) * odd))
        return fair_odds
    elif kind == 'odds_ratio':
        assert len(odds) == 3, "Only 3-ways handled so far!"
        orat = odds_ratio_solver(odds)
        fair_odds = []
        for o in odds:
            fair_odds.append(fair_odds_from_odds_ratio(o, orat))
        return fair_odds
    elif kind == 'log':
        expo = log_expo_solver(odds)
        fair_odds = []
        for o in odds:
            fair_odds.append(fair_odds_from_log_expo(o, expo))
        return fair_odds
    else:
        raise NotImplementedError("kind = '%s' not implemented yet" % kind)


class poisson_calculator:
    def __init__(self, home_odds=None, away_odds=None, draw_odds=None, commission=0.02, workers=1, max_loss_prob=0.3):
        self._home_odds = home_odds
        self._draw_odds = draw_odds
        self._away_odds = away_odds
        self._workers = workers
        self._max_loss_prob = max_loss_prob

    @property
    def max_loss_prob(self):
        return self._max_loss_prob

    @max_loss_prob.setter
    def max_loss_prob(self, value):
        self._max_loss_prob = value

    @property
    def home_odds(self):
        return self._home_odds

    @property
    def draw_odds(self):
        return self._draw_odds

    @property
    def away_odds(self):
        return self._away_odds

    @home_odds.setter
    def home_odds(self, value):
        self._home_odds = value

    @draw_odds.setter
    def draw_odds(self, value):
        self._draw_odds = value

    @away_odds.setter
    def away_odds(self, value):
        self._away_odds = value

    @property
    def workers(self):
        return self._workers

    @workers.setter
    def workers(self, value):
        self._workers = value

    @property
    def clean_odds(self):
        ho, do, ao = remove_overround(
            [self.home_odds, self.draw_odds, self.away_odds])
        return ho, do, ao

    @property
    def probs(self):
        """
        the overround gets removed inside the function:
        """
        return mov_probs_one_arg2([self.home_odds, self.draw_odds, self.away_odds])

    @property
    def probs_rmse(self):
        return mov_probs_one_arg3([self.home_odds, self.draw_odds, self.away_odds])

    @property
    def market_implied_probs(self):
        return 1 / np.array(self.clean_odds)

    def fitted_probs(self, use_rmse=False):
        if not use_rmse:
            probs = self.probs
        else:
            probs = self.probs_rmse
        hw = dict_prob_mov_greater(probs, 0)
        dw = dict_prob_mov_equal(probs, 0)
        aw = dict_prob_mov_less(probs, 0)
        return np.array([hw, dw, aw])

    @property
    def expected_mov(self):

        return clubelo_expected_mov(self.probs)

    def predicted_spread(self, **kwargs):
        home_odds = kwargs.pop('home_odds', None)
        draw_odds = kwargs.pop('draw_odds', None)
        away_odds = kwargs.pop('away_odds', None)

        if (home_odds is not None) and (draw_odds is not None) and (away_odds is not None):
            self.home_odds = home_odds
            self.draw_odds = draw_odds
            self.away_odds = away_odds
        return -1 * self.expected_mov

    def grid(self, apply_max_loss=True, use_rmse=False):
        probs = self.probs
        if use_rmse:
            probs = self.probs_rmse
        if self.workers > 1:
            grid_bf, odds_bf = gr.grid_evaluator(
                calculate_betfair_odds_and_loss_prob, grid=asian_handicaps(), args=(probs,), workers=self.workers)
        else:
            # tmp = map(calculate_betfair_odds_and_loss_prob, asian_handicaps(), (probs,))
            odds_bf = [calculate_betfair_odds_and_loss_prob(
                hc, probs) for hc in asian_handicaps()]
            grid_bf = asian_handicaps()
            # grid_bf = np.array([x[0] for x in tmp])
            # odds_bf = np.array([x[1] for x in tmp])
            # fetto = list(tmp)

            # grid_bf, odds_bf = map(calculate_betfair_odds_and_loss_prob, asian_handicaps(), (probs,))
        # pdb.set_trace()
        agrid = np.array(grid_bf).reshape(len(grid_bf), 1)
        raw_grid = pd.DataFrame(np.hstack([agrid, np.array((odds_bf))]), columns=[
                                'hc', 'odds', 'loss']).dropna()
        if not apply_max_loss:
            return raw_grid

        mlp = self.max_loss_prob  # NOQA: F841
        return raw_grid.query("loss <= @mlp")

    def loss_prob(self, handicap, **kwargs):

        home_odds = kwargs.pop('home_odds', None)
        draw_odds = kwargs.pop('draw_odds', None)
        away_odds = kwargs.pop('away_odds', None)

        if (home_odds is not None) and (draw_odds is not None) and (away_odds is not None):
            self.home_odds = home_odds
            self.draw_odds = draw_odds
            self.away_odds = away_odds
        return loss_calc(hc=handicap, probs=self.probs)

    def recommended_bet(self, use_rmse=False, **kwargs):

        home_odds = kwargs.pop('home_odds', None)
        draw_odds = kwargs.pop('draw_odds', None)
        away_odds = kwargs.pop('away_odds', None)
        loss_probi = kwargs.pop('cutoff_lp', None)

        if (home_odds is not None) and (draw_odds is not None) and (away_odds is not None) and (loss_probi is not None):
            self.home_odds = home_odds
            self.draw_odds = draw_odds
            self.away_odds = away_odds
            self.max_loss_prob = loss_probi
        df = self.grid(use_rmse=use_rmse)
        tmp = df[df.odds == df.odds.max()].iloc[0]

        fo = np.ceil(100 * tmp.odds) / 100
        hc = tmp.hc
        if int(hc) == hc:
            return '%d@%.2f' % (hc, fo)

        return '%.2f@%.2f' % (hc, fo)

    def opponent_grid(self, use_rmse=False):
        """
        what's the best bet from the opponent's perspective?
        """
        _grid = self.grid(apply_max_loss=False, use_rmse=use_rmse)
        # to avoid divide by zero:
        _grid = _grid.query("odds != 1")
        oppo_hcs = []
        oppo_fodds = []
        lossis = []
        for _tuple in _grid.itertuples():
            # remove the betfair commission:
            offi = betfair_equivalent_odds(
                _offsetting_odds(betfair_net_odds(_tuple.odds)))
            oppo_hc = -1 * _tuple.hc
            oppo_hcs.append(oppo_hc)
            oppo_fodds.append(offi)
            lossis.append(1 - _tuple.loss)
        _grid['opp_odds'] = oppo_fodds
        _grid['opp_hc'] = oppo_hcs
        _grid['opp_lp'] = lossis

        return _grid[['opp_hc', 'opp_odds', 'opp_lp']]

    def opponent_recommended_bet(self, loss_limit=0.3, use_rmse=False, **kwargs):

        home_odds = kwargs.pop('home_odds', None)
        draw_odds = kwargs.pop('draw_odds', None)
        away_odds = kwargs.pop('away_odds', None)

        if (home_odds is not None) and (draw_odds is not None) and (away_odds is not None):
            self.home_odds = home_odds
            self.draw_odds = draw_odds
            self.away_odds = away_odds
        _grid = self.opponent_grid(use_rmse=use_rmse)
        eligible = _grid.query("opp_lp <= @loss_limit")
        if eligible.empty:
            return 'N/A'
        tmp = eligible.nlargest(1, 'opp_odds').iloc[0]

        fo = np.ceil(100 * tmp['opp_odds']) / 100

        hc = tmp.opp_hc
        if int(hc) == hc:
            return '%d@%.2f' % (hc, fo)

        return '%.2f@%.2f' % (hc, fo)


class skellam_calculator(poisson_calculator):
    def __init__(self, home_odds=None, away_odds=None, draw_odds=None, commission=0.02, workers=1, max_loss_prob=0.3):
        super().__init__(home_odds=home_odds, away_odds=away_odds, draw_odds=draw_odds,
                         commission=commission, workers=1, max_loss_prob=max_loss_prob)

        self._mu1 = None
        self._mu2 = None

    @property
    def mu1(self):
        return self._mu1

    @mu1.setter
    def mu1(self, value):
        self._mu1 = value

    @property
    def mu2(self):
        return self._mu2

    @mu2.setter
    def mu2(self, value):
        self._mu2 = value

    @property
    def probs(self):
        """
        the overround gets removed inside the function:
        """
        probs, mu1, mu2 = mov_probs_skellam(
            self.home_odds, self.draw_odds, self.away_odds)

        self.mu1 = mu1
        self.mu2 = mu2
        return probs

    @property
    def inverted_probs(self):
        out = {}
        for k, v in self.probs.items():
            out[-1 * k] = v
        return out

    @property
    def expected_mov(self):
        # print(self.probs)
        return self.mu1 - self.mu2

    def report(self):
        grid = self.grid(apply_max_loss=False)
        grid['fair_odds'] = grid.odds
        oppo_hcs = []
        oppo_fodds = []
        for _tuple in grid.itertuples():
            try:
                offi = betfair_equivalent_odds(
                    _offsetting_odds(betfair_net_odds(_tuple.fair_odds)))
            except:
                oppo_hc = -1 * _tuple.hc
                oppo_hcs.append(oppo_hc)
                oppo_fodds.append(np.nan)
                continue
            oppo_hc = -1 * _tuple.hc
            oppo_hcs.append(oppo_hc)
            oppo_fodds.append(offi)
        grid['opp_odds'] = oppo_fodds
        grid['opp_hc'] = oppo_hcs
        
        max_loss_prob = self.max_loss_prob * 100
        back_home = grid.query(
            "loss.mul(100) <= @max_loss_prob and fair_odds >1.14", engine='python').copy()
        back_away = grid.query(
            "loss.mul(100) >= (100-@max_loss_prob) and opp_odds > 1.14", engine='python').copy()
        back_away.sort_values(['loss', 'opp_odds'],
                              ascending=[1, 0], inplace=True)
        # pdb.set_trace()
        bets = []

        back_home = back_home.iloc[:1]
        for _t in back_home.itertuples():
            raw_odds = np.ceil(_t.fair_odds * 100) / 100

            # pdb.set_trace()
            epnl = asian_expected_pnl_clubelo(
                _t.hc, raw_odds, self.probs) * 1e4
            bets.append(["%.2f @ %.3f (%.2f lp, %.0f ep)" %
                        (_t.hc, raw_odds, 100 * _t.loss, epnl), epnl])

            back_away = back_away.iloc[:1]
        for _t in back_away.itertuples():
            raw_odds = np.ceil(_t.opp_odds * 100) / 100
            epnl = asian_expected_pnl_clubelo(
                _t.opp_hc, raw_odds, self.inverted_probs) * 1e4
            bets.append(["%.2f @ %.3f (%.2f lp, %.0f ep)" %
                        (_t.opp_hc, raw_odds, 100 * (1 - _t.loss), epnl), epnl])

        idx = np.concatenate(
            [np.repeat('H', len(back_home)), np.repeat('A', len(back_away))])
        do_it = pd.DataFrame(bets, columns=['bet', 'epnl'], index=pd.Index(idx))
        do_it.sort_values('epnl', ascending=False, inplace=True)
        del do_it['epnl']
        ho, do, ao = self.clean_odds
        styler = do_it.style.pipe(make_pretty, 'Bets %s (%.2f-%.2f-%.2f: %.2f)' %
                                  (pd.to_datetime('now').strftime('%H:%M'), ho, do, ao, self.predicted_spread()))
        return do_it, styler


def find_all(line='', tag='', case=False, return_unique=False):
    if not case:
        if not return_unique:
            return [m.group() for m in re.finditer(tag, line, re.IGNORECASE)]
        return list(set([m.group() for m in re.finditer(tag, line, re.IGNORECASE)]))
    else:
        if not return_unique:
            return [m.group() for m in re.finditer(tag, line)]
        return list(set([m.group() for m in re.finditer(tag, line)]))


def us_to_eu_odds(in_odds):
    if in_odds > 0:
        return 1 + in_odds / 100
    else:
        return 1 + 100 / abs(in_odds)


def parse_oddsportal_page(text=None, skip_finished=True):
    """
    copy table-element from Inspect
    """

    idxs = find_all_indicies(text, r'-\w{8}/')
    tags = find_all(text, r'-\w{8}/')
    decimal_query = r'[+-]?([0-9]+\.?[0-9]*|\.[0-9]+)'

    out = []
    for i in range(len(idxs)):
        idx = idxs[i]
        _id = tags[i][1:-1]
        if not (re.search('[a-z]', _id) and re.search('[A-Z]', _id)):
            continue
        try:
            next_idx = idxs[i + 1]
        except:
            next_idx = -1

        search_text = text[idx:next_idx]

        if skip_finished:
            if 'table-score' in search_text:
                continue

        desc_idx = search_text.find('/">')
        end_idx = search_text.find('</a>')

        odds_idxs = find_all_indicies(text[idx:], 'odds_text')[:3]
        odds = []

        time_idx = text[:idx].rfind('table-time datet')
        time_text = re.search(r'\d{2}\:\d{2}', text[time_idx:])

        if time_text is not None:
            time_disp = time_text.group()
        else:
            # pdb.set_trace()
            time_disp = 'n/a'
        for oid in odds_idxs:
            odds_txt = re.search(decimal_query, text[idx + oid:]).group()

            if odds_txt.startswith('+') or odds_txt.startswith('-'):
                if odds_txt.startswith('+'):
                    this_odds = us_to_eu_odds(int(odds_txt.replace('+', '')))
                else:
                    this_odds = us_to_eu_odds(int(odds_txt))
            else:
                this_odds = float(odds_txt)
            odds.append(this_odds)
        row = [_id, search_text[desc_idx + 3:end_idx], time_disp] + odds

        if '<span class' in row[1]:
            continue
        out.append(row)
    return pd.DataFrame(out, columns=['game_id', 'match', 'time', 'home', 'draw', 'away'])
