def test_import():
    import opencount_ci
    assert hasattr(opencount_ci, "count_objects")
