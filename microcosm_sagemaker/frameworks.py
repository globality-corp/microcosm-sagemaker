class FrameworkRegistry:
    def __init__(self, graph):
        self.graph = graph
        self.frameworks = []

    def register_framework(self, framework):
        self.frameworks.append(framework)

    def init(self):
        for framework in self.frameworks:
            framework.init()
