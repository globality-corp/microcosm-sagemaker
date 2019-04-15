def make_click_callback(function):
    def callback(ctx, param, value):
        return function(value)

    return callback
