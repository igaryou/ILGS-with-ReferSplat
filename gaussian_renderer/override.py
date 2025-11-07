
class temp_override:
    def __init__(self, g, new_feat): self.g, self.new, self.old = g, new_feat, None
    def __enter__(self):
        self.old = self.g.get_semantic_features()  # 実装に合わせて取得名
        self.g.set_semantic_features(self.new)
    def __exit__(self, *exc): self.g.set_semantic_features(self.old)
