def make_click_callback(function):
    """
    Given a `function`, returns a callback function that can be used for a
    click option's `callback=` to apply `function` to the value before passing
    the argument to the command

    """
    def callback(ctx, param, value):
        return function(value)

    return callback
