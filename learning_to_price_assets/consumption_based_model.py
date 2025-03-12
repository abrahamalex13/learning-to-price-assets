from collections.abc import Callable

def transform_wellbeing_power_form(consumption, strength_of_diminishing: float): 
    """
    Wellbeing also known as 'utility', in economics jargon.
    Wooldridge associates the two terms in _Introductory Econometrics_ (Section 1-2).
    Cochrane's _Asset Pricing_ (Section 1.1) presents this convenient functional form.

    Expect strength_of_diminishing in [0, 1).
    strength_of_diminishing=0 implies, constant wellbeing gain with additional consumption.
    """

    wellbeing = ( 1 / (1 - strength_of_diminishing) ) * consumption**(1-strength_of_diminishing)

    return wellbeing

def transform_wellbeing_power_form_dwdc(consumption, strength_of_diminishing: float):
    """ 'dwdc': derivative of wellbeing with respect to consumption. """

    deriv = consumption**(-1 * strength_of_diminishing)
    
    return deriv

def infer_price(
    consumption_today, 
    consumption_tomorrow, 
    wellbeing_dwdc: Callable, 
    payoff_asset, 
    subjective_discount_factor
    ):
    """
    To avoid this function's arguments proliferation, 
    request wellbeing_dwdc with pre-specified arguments (besides consumption). 
    """

    price = (
        subjective_discount_factor * 
        (wellbeing_dwdc(consumption_tomorrow) / wellbeing_dwdc(consumption_today)) *
        payoff_asset
        )

    return price

def transform_lifetime_wellbeing(
    wellbeing_func: Callable,
    consumption_today, 
    consumption_expected_tomorrow, 
    subjective_discount_factor: float
    ):
    """
    Wellbeing also known as 'utility', in economics jargon.
    Wooldridge associates the two terms in _Introductory Econometrics_ (Section 1-2).

    Cochrane's _Asset Pricing_ presents a convenient functional form.
    """

    wellbeing_today = wellbeing_func(consumption_today)
    wellbeing_expected_tomorrow = subjective_discount_factor * wellbeing_func(consumption_expected_tomorrow)
    lifetime_wellbeing = wellbeing_today + wellbeing_expected_tomorrow

    return lifetime_wellbeing