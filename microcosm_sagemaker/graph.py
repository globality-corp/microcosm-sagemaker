def get_component_name(graph, component):
    return next(
        key
        for key, possible_component in graph
        if possible_component == component
    )
