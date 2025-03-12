import numpy as np
from functools import partial
from learning_to_price_assets.consumption_based_model import (
    transform_wellbeing_power_form, 
    transform_wellbeing_power_form_dwdc, 
    infer_price, 
    transform_lifetime_wellbeing
    )

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

    # direct from power form model: consumption**(-1 * gamma)
    expected = 0.17783
    
    result = transform_wellbeing_power_form_dwdc(consumption, gamma)

    assert np.isclose(expected, result, atol=0.001)

def test_infer_price():
    """Hard-coded model from _Asset Pricing_ Section 1.1."""

    beta = 1 / 1.05
    strength_of_diminishing = 0.75
    consumption_today = 50
    consumption_tomorrow = 100
    payoff_asset = 100

    dwdc_tomorrow = transform_wellbeing_power_form_dwdc(consumption_tomorrow, strength_of_diminishing)
    dwdc_today = transform_wellbeing_power_form_dwdc(consumption_today, strength_of_diminishing)

    expected_price = beta * dwdc_tomorrow / dwdc_today * payoff_asset

    wellbeing_dwdc_curve = partial(transform_wellbeing_power_form_dwdc, strength_of_diminishing=strength_of_diminishing)
    result_price = infer_price(consumption_today, consumption_tomorrow, wellbeing_dwdc_curve, payoff_asset, beta)

    assert np.isclose(expected_price, result_price, atol=0.001)

def test_transform_lifetime_wellbeing():

    beta = 1 / 1.05
    strength_of_diminishing = 0.75
    consumption_today = 50
    consumption_tomorrow = 100

    wellbeing_today = transform_wellbeing_power_form(consumption_today, strength_of_diminishing)
    wellbeing_discounted_tomorrow = beta * transform_wellbeing_power_form(consumption_tomorrow, strength_of_diminishing)
    expected = wellbeing_today + wellbeing_discounted_tomorrow

    wellbeing_func = partial(transform_wellbeing_power_form, strength_of_diminishing=strength_of_diminishing)
    result = transform_lifetime_wellbeing(wellbeing_func, consumption_today, consumption_tomorrow, beta)

    assert np.isclose(expected, result, atol=0.001)
