import json


def add_extra_config(key):
    """
    Creates a click callback which will store the value into an "extra_config"
    parameter at the specified key, where the key can contain '.' to create
    nested dicts within the config.

    """
    key_components = key.split('.')
    parents, key = key_components[:-1], key_components[-1]

    def callback(ctx, param, value):
        extra_config = ctx.params.setdefault('extra_config', dict())
        for parent in parents:
            extra_config = extra_config.setdefault(parent, dict())
        extra_config[key] = value

    return callback


def load_extra_config(ctx, param, value):
    """
    A click callback to fill an "extra_config" parameter with the contents of a
    json file.  If any other params also modify "extra_config", they will take
    precedence.

    """
    extra_config = ctx.params.get('extra_config', dict())

    configuration = (
        json.load(value)
        if value else dict()
    )

    configuration.update(extra_config)

    ctx.params['extra_config'] = configuration
