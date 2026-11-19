"""Placeholder smoke test: checks that the package and its modules import."""


def test_package_imports():
    import volforecast
    from volforecast import data, estimators

    assert volforecast is not None
    assert set(data.TICKERS) == {"SPY", "TLT", "FXE", "GLD", "USO"}
    assert callable(estimators.daily_variance)
