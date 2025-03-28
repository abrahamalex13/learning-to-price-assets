import numpy as np
from functools import partial
from learning_to_price_assets.consumption_based_model import (
    transform_wellbeing_power_form, 
    transform_wellbeing_power_form_dwdc, 
    infer_price, 
    transform_lifetime_wellbeing
    )

# because a client/user tests these functions by their numeric results,
# hard-code test case numeric results below.

def test_transform_wellbeing_power_form():
    """
    Benchmark against hard-coded power form model, per _Asset Pricing_ Section 1.1.
    """

    result = transform_wellbeing_power_form(consumption=100, strength_of_diminishing=0.75)
    # direct from power form model: ( 1 / (1-0.75) ) * 100**(1-0.75)
    # a user/client would test this function via output value.
    expected = 4 * 3.1623

    assert np.isclose(expected, result, atol=0.001)

def test_transform_wellbeing_power_form_dwdc():

    consumption=10
    gamma=0.75

    expected = consumption**(-1 * gamma)
    # interpretation: wellbeing rate of change at tested consumption level
    # if equal to 1, then wellbeing improves with consumption at 1:1 rate
    assert np.isclose(expected, 0.17783, atol=0.001)

    result = transform_wellbeing_power_form_dwdc(consumption, gamma)

    assert np.isclose(expected, result, atol=0.001)

def test_infer_price():
    """
    When future consumption expected higher versus today,
    less willing to reduce today's consumption in exchange for future payoffs.
    So, asset price decreases.
    """

    beta = 1 / 1.05
    strength_of_diminishing = 0.75
    consumption_today = 50
    consumption_tomorrow = 100
    payoff_asset = 100

    dwdc_tomorrow = transform_wellbeing_power_form_dwdc(consumption_tomorrow, strength_of_diminishing)
    assert np.isclose(dwdc_tomorrow, 0.03162, atol=0.001)
    dwdc_today = transform_wellbeing_power_form_dwdc(consumption_today, strength_of_diminishing)
    assert np.isclose(dwdc_today, 0.053183, atol=0.001)

    expected_price = beta * dwdc_tomorrow / dwdc_today * payoff_asset
    assert np.isclose(expected_price, 56.624, atol=0.01)

    wellbeing_dwdc_curve = partial(transform_wellbeing_power_form_dwdc, strength_of_diminishing=strength_of_diminishing)
    result_price = infer_price(consumption_today, consumption_tomorrow, wellbeing_dwdc_curve, payoff_asset, beta)

    assert np.isclose(expected_price, result_price, atol=0.001)

def test_transform_lifetime_wellbeing():

    beta = 1 / 1.05
    strength_of_diminishing = 0.5
    consumption_today = 424
    consumption_tomorrow = 424

    wellbeing_today = transform_wellbeing_power_form(consumption_today, strength_of_diminishing)
    assert np.isclose(wellbeing_today, 41.183, atol=0.01)
    wellbeing_discounted_tomorrow = beta * transform_wellbeing_power_form(consumption_tomorrow, strength_of_diminishing)
    assert np.isclose(wellbeing_discounted_tomorrow, 39.221, atol=0.01)
    expected = wellbeing_today + wellbeing_discounted_tomorrow

    wellbeing_func = partial(transform_wellbeing_power_form, strength_of_diminishing=strength_of_diminishing)
    result = transform_lifetime_wellbeing(wellbeing_func, consumption_today, consumption_tomorrow, beta)

    assert np.isclose(expected, result, atol=0.001)

def test_plot_points():
    """
    When plotting 3D Surface colors, and labeling in the tooltip -- discovered plotly library bug,
    which swaps plotted axes versus the request via api (https://github.com/plotly/plotly.js/issues/5003).
    Therefore, it's critical to test plotted coordinate versus formula results.
    """

    # visible when interacting with plot
    consumption_today_plot = 111
    consumption_tomorrow_plot = 646
    lifetime_wellbeing_plot = 69.5

    beta = 1 / 1.05
    strength_of_diminishing = 0.5

    wellbeing_curve = partial(transform_wellbeing_power_form, strength_of_diminishing=strength_of_diminishing)

    lifetime_wellbeing_formula_result = (
        wellbeing_curve(consumption=consumption_today_plot) + 
        beta * wellbeing_curve(consumption=consumption_tomorrow_plot)
        )
    
    assert np.isclose(lifetime_wellbeing_formula_result, lifetime_wellbeing_plot, atol=0.1)
