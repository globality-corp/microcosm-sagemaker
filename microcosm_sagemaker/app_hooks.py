from pkg_resources import DistributionNotFound, iter_entry_points


class AppHooks:
    @staticmethod
    def create_train_graph(*args, **kwargs):
        for name, factory in self._get_factories():
            if name == "train_graph":
                return factory(*args, **kwargs)
        return None

    @staticmethod
    def create_serve_graph(*args, **kwargs):
        for name, factory in self._get_factories():
            if name == "serve_graph":
                return factory(*args, **kwargs)
        return None

    @staticmethod
    def _get_factories():
        for entry_point in iter_entry_points(group="microcosm-sagemaker.hooks"):
            try:
                factory = entry_point.load()
                yield entry_point.name, factory
            except DistributionNotFound:
                continue
        yield from []
